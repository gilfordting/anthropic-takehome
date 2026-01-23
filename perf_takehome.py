"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

We recommend you look through problem.py next.
"""

from collections import defaultdict
from dataclasses import dataclass
import random
from typing import Optional
import unittest

from problem import (
    Engine,
    Instruction,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)


CONST_PREFIX = "const "
VECTOR_PREFIX = "v"


# Symbolic structures use names, instead of concrete registers.
# We can safely instantiate one of these like val = val + 1 and it will automatically add `val` to `frees`, triggering behavior to use different registers.
# Semantically, we allocate new registers for dest as a separate mapping, free registers and remove from current mapping, then merge in dest mappings into current mapping.
@dataclass
class SymbolicInstructionSlot:
    batch: int
    round: int
    engine: Engine
    op: str
    # dependencies are all here, minus ones that start with "const_"
    arg_names: tuple[str, ...]
    # same role as `defines`, but explicitly separated from arg list
    dest: Optional[str] = None
    # always frees these, and adds them to freelist
    frees: Optional[set[str]] = None
    # this is used to offset into a vector reg, if defined
    vector_offsets: Optional[dict[str, int]] = None
    comment: str = None

    def is_vector_var(self, name: str) -> bool:
        if name.startswith(VECTOR_PREFIX):
            return name not in self.vector_offsets  # this will turn it into a scalar
        return name.startswith(CONST_PREFIX) and name[len(CONST_PREFIX) :].startswith(
            VECTOR_PREFIX
        )

    def __post_init__(self):
        if self.frees is None:
            self.frees = set()
        if self.vector_offsets is None:
            self.vector_offsets = {}
        # If `dest` reuses a symbolic register, use different registers for input and output.
        # In order to trigger this behavior, add `dest` to `frees` if not already present.
        if self.dest is not None and self.dest in self.arg_names:
            self.frees.add(self.dest)
        # you can't free constant registers
        assert all(not name.startswith(CONST_PREFIX) for name in self.frees), (
            "cannot free constant registers"
        )
        # and you can only free registers you actually used
        assert all(reg in self.arg_names for reg in self.frees), (
            "cannot free registers you didn't use; got: "
            + ", ".join(self.frees)
            + " but only have: "
            + ", ".join(self.arg_names)
        )
        # vector_offsets must specify for actually present things
        assert all(
            name in self.arg_names or name == self.dest for name in self.vector_offsets
        )
        # vector offsets must be 0 to 7
        assert all(0 <= offset < VLEN for offset in self.vector_offsets.values()), (
            "vector offsets must be 0 to 7"
        )

        # check that all vector slots get vector arguments/destinations
        operands = self.arg_names
        match self.engine:
            case "alu":
                if self.dest is not None:
                    operands += (self.dest,)
                assert len(operands) == 3, "alu must have 2 arguments and 1 destination"
                assert all(not self.is_vector_var(name) for name in operands), (
                    "alu arguments/destinations must be scalar, instead got: "
                    + ", ".join(operands)
                )
            case "valu":
                if self.dest is not None:
                    operands += (self.dest,)
                if self.op == "multiply_add":
                    assert len(operands) == 4, (
                        "multiply_add must have 3 arguments and 1 destination"
                    )
                else:
                    assert len(operands) == 3, (
                        "valu must have 2 arguments and 1 destination; got: "
                        + ", ".join(operands)
                        + " for op: "
                        + self.op
                    )
                assert all(self.is_vector_var(name) for name in operands), (
                    "valu arguments/destinations must be vector, instead got: "
                    + ", ".join(operands)
                )
            case "load":
                assert len(operands) == 1, "load must have 1 address"
                assert all(not self.is_vector_var(name) for name in operands), (
                    "load arg must be a scalar address"
                )
                assert self.dest is not None, "load must have a destination"
                assert self.is_vector_var(self.dest) == self.is_vector, (
                    "load type must match destination type"
                )
            case "store":
                assert len(operands) == 2, (
                    "store must have 2 arguments: value and address"
                )
                assert not self.is_vector_var(operands[0]), (
                    "store address must be scalar"
                )
                assert self.is_vector_var(operands[1]) == self.is_vector, (
                    "store value must match destination type"
                )

    def to_concrete(
        self,
        in_mapping: dict[str, int],
        out_mapping: dict[str, int],
        consts: dict[str, int],
    ) -> (Engine, tuple):
        # Used for inputs. Will throw error if mapping not found.
        def name_to_reg(name: str, mapping: dict[str, int]):
            if name.startswith(CONST_PREFIX):
                return consts[name[len(CONST_PREFIX) :]]
            assert name in mapping, (
                f"name {name} not found in mapping for round {self.round} and batch {self.batch}; comment: {self.comment};"
            )

            reg = mapping[name]
            if name in self.vector_offsets:
                reg += self.vector_offsets[name]
            assert reg < 1536, (
                "register is out of bounds; name is: "
                + name
                + "; reg is: "
                + str(reg)
                + "vector offsets: "
                + str(self.vector_offsets[name])
            )
            return reg

        full_args = [name_to_reg(name, in_mapping) for name in self.arg_names]
        if self.dest is not None:
            full_args = [name_to_reg(self.dest, out_mapping)] + full_args
        return self.engine, (self.op, *full_args)

    @property
    def is_vector(self) -> bool:
        match self.engine:
            case "alu":
                return False
            case "valu":
                return True
            case "load":
                return self.op == "vload"
            case "store":
                return self.op == "vstore"
            case "flow":
                return self.op == "vselect"
            case "debug":
                return False
            case _:
                assert False, "invalid engine; got: " + self.engine

    @property
    def is_scalar(self) -> bool:
        return not self.is_vector


def make_slot(
    batch: int,
    round: int,
    engine: Engine,
    op: str,
    arg_names: list[str],
    dest: Optional[str] = None,
    frees: Optional[set[str]] = None,
    vector_offsets: Optional[dict[str, int]] = None,
    comment: str = None,
) -> SymbolicInstructionSlot:
    return SymbolicInstructionSlot(
        batch=batch,
        round=round,
        engine=engine,
        op=op,
        dest=dest,
        arg_names=tuple(arg_names),
        frees=frees,
        vector_offsets=vector_offsets,
        comment=comment,
    )


@dataclass
class SymbolicBundle:
    slots: list[SymbolicInstructionSlot]

    def __post_init__(self):
        # destination names must be unique
        dests = [slot.dest for slot in self.slots if slot.dest is not None]
        assert len(dests) == len(set(dests)), "destination names must be unique"

    def to_concrete(
        self,
        using: dict[str, int],
        scalar_freelist: set[int],
        vector_freelist: set[int],
        consts: dict[str, int],
    ) -> Instruction:
        # make the out mapping: take stuff from freelist and use it for destinations
        # also keep track of which slot it came from
        out_mapping = {}
        for i, slot in enumerate(self.slots):
            if slot.dest is not None:
                # Must make this compatible with vector offsets
                if slot.dest in slot.vector_offsets:
                    assert slot.op == "load", (
                        "vector offsets can only be used with load dest"
                    )
                    # If loading to a vector offset, we don't load to a new register.
                    # We must use an existing register.
                    assert slot.dest in using, (
                        "If loading to vector offset, dest must already exist"
                    )
                    out_mapping[slot.dest] = (
                        using[slot.dest],
                        i,
                    )
                    continue
                freelist = scalar_freelist if slot.is_scalar else vector_freelist
                assert len(freelist) > 0, (
                    f"freelist is empty for slot {i}, which is batch {slot.batch} and round {slot.round}; it is scalar: {slot.is_scalar}"
                )

                out_mapping[slot.dest] = (freelist.pop(), i)
        assert all(reg < 1536 for reg, _ in out_mapping.values()), (
            "register is out of bounds"
        )
        assert all(reg < 1536 for reg in using.values()), (
            "using register is out of bounds"
        )
        # entire instruction
        inst = defaultdict(list)
        for slot in self.slots:
            engine, args = slot.to_concrete(
                using, {name: reg for name, (reg, _) in out_mapping.items()}, consts
            )
            inst[engine].append(args)

        # after this, we can safely free registers; aggregate first so we don't try to free the same name twice, though
        # get the set of all freed variables
        all_frees = set()
        for slot in self.slots:
            if slot.frees is not None:
                all_frees |= slot.frees
        # check that all the registers we're trying to free are actually available to be freed
        assert all(name in using for name in all_frees), (
            "cannot free registers that are not available"
        )
        for name in all_frees:
            assert not name.startswith(CONST_PREFIX), "cannot free constant registers"
            if name.startswith(VECTOR_PREFIX):
                assert using[name] + 7 < 1536, "vector register is out of bounds"
                vector_freelist.add(using.pop(name))
            else:
                assert using[name] < 1536, "scalar register is out of bounds"
                scalar_freelist.add(using.pop(name))

        # check that we have a valid instruction
        for engine in inst:
            assert len(inst[engine]) <= SLOT_LIMITS[engine], (
                f"engine {engine} exceeded slot limit"
            )

        for name, (reg, src_slot_i) in out_mapping.items():
            # if we are loading to vector offset, should not redefine dest; it already exists
            if name in self.slots[src_slot_i].vector_offsets:
                continue
            assert name not in using, (
                f"variable {name} is being redefined, in round {self.slots[src_slot_i].round} and batch {self.slots[src_slot_i].batch}"
            )
            using[name] = reg

        return inst


@dataclass
class SymbolicProgram:
    bundles: list[SymbolicBundle]

    def to_concrete(
        self,
        using: dict[str, int],
        scalar_freelist: set[int],
        vector_freelist: set[int],
        consts: dict[str, int],
    ) -> list[Instruction]:
        old_scalar_freelist = {i for i in scalar_freelist}
        old_vector_freelist = {i for i in vector_freelist}
        insts = [
            bundle.to_concrete(using, scalar_freelist, vector_freelist, consts)
            for bundle in self.bundles
        ]
        # we should be back to the same
        assert len(vector_freelist) == len(old_vector_freelist)
        assert all(i in old_vector_freelist for i in vector_freelist)
        # some error margin allowed for scalar freelist (addresses)
        assert abs(len(scalar_freelist) - len(old_scalar_freelist)) < 4
        assert all(i in old_scalar_freelist for i in scalar_freelist)

        return insts


def infer_engine(op: str) -> Engine:
    engine = None
    if "load" in op or "const" in op:
        engine = "load"
    elif "store" in op:
        engine = "store"
    elif op in ("+", "-", "*", "//", "cdiv", "^", "&", "|", "<<", ">>", "%", "<", "=="):
        engine = "alu"
    elif op == "multiply_add":
        engine = "valu"
    elif "valu" in op:
        engine = "valu"
        op = op[4:]
    elif op in (
        "select",
        "add_imm",
        "vselect",
        "halt",
        "pause",
        "trace_write",
        "cond_jump",
        "cond_jump_rel",
        "jump",
        "jump_indirect",
        "coreid",
    ):
        engine = "flow"
    else:
        engine = "debug"

    return engine, op


# Makes a single bundle (1 VLIW instruction) that has a single occupied slot.
# Use this factory for everything. Will infer engine from name.
def single_bundle(
    op: str,
    arg_names: list[str],
    dest: Optional[str] = None,
    frees: Optional[set[str]] = None,
    vector_offsets: Optional[dict[str, int]] = None,
    batch: int = None,
    round: int = None,
    comment: str = None,
) -> SymbolicBundle:
    engine, op = infer_engine(op)
    return SymbolicBundle(
        slots=[
            make_slot(
                batch,
                round,
                engine,
                op,
                arg_names,
                dest,
                frees,
                vector_offsets,
                comment,
            )
        ]
    )


# Makes a single bundle that has multiple occupied slots. This merges multiple bundles together.
def multi_bundle(*args: SymbolicBundle) -> SymbolicBundle:
    return SymbolicBundle(slots=sum([bundle.slots for bundle in args], []))


def const(name: str) -> str:
    return f"{CONST_PREFIX}{name}"


def vector(name: str) -> str:
    return f"{VECTOR_PREFIX}{name}"

def vconst(name: str) -> str:
    return const(vector(name))


# localize a name to a batch and round
def localize(name: str, batch: int, round: int) -> str:
    return f"{name}_{batch}_{round}"


def batch_localize(name: str, batch: int) -> str:
    return f"{name}_{batch}"


# all symbolic instructions needs to provide this info
# if a name is used for dest, does not need to be provided in `frees`
# op: str,
# arg_names: list[str],
# dest: Optional[str] = None,
# frees: Optional[set[str]] = None,

def calculate_shifts(batch: int) -> list[int]:
    factor = batch // VLEN
    shift = 0
    shifts = []
    while factor > 0:
        if factor % 2 == 1:
            shifts.append(shift)
        factor //= 2
        shift += 1
    return shifts


def make_initial_load(
    curr_addr_name: str, val_name: str, batch: int, round: int
) -> list[SymbolicBundle]:
    bundles = []
    assert batch % VLEN == 0, "batch must be a multiple of VLEN"

    tmp_name = batch_localize("addr_calc_tmp", batch)

    shifts = calculate_shifts(batch)
    if len(shifts) == 0:
        bundles.append(
            single_bundle(
                batch=batch,
                round=round,
                op="vload",
                arg_names=[const("inp_values_p")],
                dest=val_name,
            )
        )
        return bundles

    first_shift, rest = shifts[0], shifts[1:]
    single_bundles = [
        single_bundle(
            batch=batch,
            round=round,
            op="<<",
            arg_names=[const("1"), const(str(first_shift))],
            dest=curr_addr_name,
            comment="first shift",
        )
    ]
    # can do two shifts in parallel on the first cycle
    if len(rest) > 0:
        single_bundles.append(
            single_bundle(
                batch=batch,
                round=round,
                op="<<",
                arg_names=[const("1"), const(str(rest[0]))],
                dest=tmp_name,
                comment="first of rest shift",
            )
        )
    bundles.append(multi_bundle(*single_bundles))
    for shift in rest[1:]:
        # curr_addr += tmp; tmp = 1 << i
        bundles.append(
            multi_bundle(
                single_bundle(
                    batch=batch,
                    round=round,
                    op="+",
                    arg_names=[curr_addr_name, tmp_name],
                    dest=curr_addr_name,
                    frees={tmp_name},
                ),
                single_bundle(
                    batch=batch,
                    round=round,
                    op="<<",
                    arg_names=[const("1"), const(str(shift))],
                    dest=tmp_name,
                ),
            )
        )
    # add tmp, multiply by VLEN, add inp_values_p, and vload
    if len(rest) > 0:
        bundles.append(
            single_bundle(
                batch=batch,
                round=round,
                op="+",
                arg_names=[curr_addr_name, tmp_name],
                dest=curr_addr_name,
                frees={tmp_name},
                comment="1",
            )
        )
    bundles.append(
        single_bundle(
            batch=batch,
            round=round,
            op="*",
            arg_names=[curr_addr_name, const("s_vlen")],
            dest=curr_addr_name,
            comment="2",
        )
    )
    bundles.append(
        single_bundle(
            batch=batch,
            round=round,
            op="+",
            arg_names=[curr_addr_name, const("inp_values_p")],
            dest=curr_addr_name,
            comment="3",
        )
    )
    bundles.append(
        single_bundle(
            batch=batch,
            round=round,
            op="vload",
            arg_names=[curr_addr_name],
            dest=val_name,
            comment="4",
        )
    )
    return bundles

# in_name is provided, and we reuse it for a lot of stuff
def make_hash(
    in_name: str, out_name: str, batch: int, round: int
) -> list[SymbolicBundle]:
    bundles = []
    curr = in_name
    # looks like:
    # valu: multiply_add, curr, curr, const vhash_mult0, const vhash_add0
    # @parallel
    #   valu: op1, tmp1, curr, const vhash_add0; frees: curr
    #   valu: op3, tmp2, curr, const vhash_mult0; frees: curr
    # valu: op2, curr, tmp1, tmp2; frees: tmp1, tmp2
    # ...
    # valu: op2, out_name, tmp1, tmp2; frees: tmp1, tmp2

    for i, (op1, _, op2, op3, _) in enumerate(HASH_STAGES):
        op1, op2, op3 = (f"valu{op}" for op in (op1, op2, op3))
        if i in (0, 2, 4):
            bundles.append(
                single_bundle(
                    batch=batch,
                    round=round,
                    op="multiply_add",
                    arg_names=[
                        curr,
                        const(f"vhash_mult{i}"),
                        const(f"vhash_add{i}"),
                    ],
                    dest=curr,
                )
            )
            continue
        tmp1 = vector(localize(f"hash_tmp1_{i}", batch, round))
        tmp2 = vector(localize(f"hash_tmp2_{i}", batch, round))
        bundles.append(
            multi_bundle(
                single_bundle(
                    batch=batch,
                    round=round,
                    op=op1,
                    arg_names=[curr, const(f"vhash_add{i}")],
                    dest=tmp1,
                    frees={curr},
                ),
                single_bundle(
                    batch=batch,
                    round=round,
                    op=op3,
                    arg_names=[curr, const(f"vhash_mult{i}")],
                    dest=tmp2,
                    frees={curr},
                ),
            )
        )
        if i == len(HASH_STAGES) - 1:
            curr = out_name
        bundles.append(
            single_bundle(
                batch=batch,
                round=round,
                op=op2,
                arg_names=[tmp1, tmp2],
                dest=curr,
                frees={tmp1, tmp2},
            )
        )
    return bundles


def make_round0(
    val_name: str, idx_name: str, treeval_name: str, batch: int, round: int
) -> list[SymbolicBundle]:
    bundles = []
    in0 = vector(localize("in0", batch, round))
    bundles.append(
        single_bundle(
            batch=batch,
            round=round,
            op="valu^",
            arg_names=[val_name, vconst("treeval0")],
            dest=in0,
            frees={val_name},
        )
    )
    bundles.extend(make_hash(in0, val_name, batch, round))
    parity0 = vector(localize("parity0", batch, round))
    bundles.append(
        single_bundle(
            batch=batch,
            round=round,
            op="valu%",
            arg_names=[val_name, vconst("2")],
            dest=parity0,
        )
    )
    bundles.append(
        multi_bundle(
            single_bundle(
                batch=batch,
                round=round,
                op="valu+",
                arg_names=[parity0, vconst("1")],
                dest=idx_name,
                frees={parity0},
            ),
            single_bundle(
                batch=batch,
                round=round,
                op="multiply_add",
                arg_names=[parity0, vconst("diff21"), vconst("treeval1")],
                dest=treeval_name,
                frees={parity0},
            ),
        )
    )
    return bundles


def make_round1(
    val_name: str, idx_name: str, treeval_name: str, batch: int, round: int
) -> list[SymbolicBundle]:
    bundles = []
    in1 = vector(localize("in1", batch, round))
    parity1 = vector(localize("parity1", batch, round))
    upperbit1 = vector(localize("upperbit1", batch, round))
    diff1 = vector(localize("diff1", batch, round))
    diff2 = vector(localize("diff2", batch, round))
    ddiff = vector(localize("ddiff", batch, round))
    bundles.append(
        single_bundle(
            batch=batch,
            round=round,
            op="valu^",
            arg_names=[val_name, treeval_name],
            dest=in1,
            frees={val_name, treeval_name},
        )
    )
    bundles.extend(make_hash(in1, val_name, batch, round))
    bundles.append(
        single_bundle(
            batch=batch,
            round=round,
            op="multiply_add",
            arg_names=[vconst("2"), idx_name, vconst("1")],
            dest=idx_name,
        )
    )
    bundles.append(
        multi_bundle(
            single_bundle(
                batch=batch,
                round=round,
                op="valu%",
                arg_names=[val_name, vconst("2")],
                dest=parity1,
            ),
            single_bundle(
                batch=batch,
                round=round,
                op="valu-",
                arg_names=[idx_name, vconst("3")],
                dest=upperbit1,
            ),
        )
    )
    bundles.append(
        multi_bundle(
            single_bundle(
                batch=batch,
                round=round,
                op="multiply_add",
                arg_names=[parity1, vconst("diff43"), vconst("treeval3")],
                dest=diff1,
                frees={parity1},
            ),
            single_bundle(
                batch=batch,
                round=round,
                op="multiply_add",
                arg_names=[parity1, vconst("diff65"), vconst("treeval5")],
                dest=diff2,
                frees={parity1},
            ),
            single_bundle(
                batch=batch,
                round=round,
                op="valu+",
                arg_names=[idx_name, parity1],
                dest=idx_name,
                frees={parity1},
            ),
        )
    )
    bundles.append(
        multi_bundle(
            single_bundle(
                batch=batch,
                round=round,
                op="valu-",
                arg_names=[diff2, diff1],
                dest=ddiff,
                frees={diff2},
            ),
            single_bundle(
                batch=batch,
                round=round,
                op="valu>>",
                arg_names=[upperbit1, vconst("1")],
                dest=upperbit1,
            ),
        )
    )
    bundles.append(
        single_bundle(
            batch=batch,
            round=round,
            op="multiply_add",
            arg_names=[upperbit1, ddiff, diff1],
            dest=treeval_name,
            frees={upperbit1, ddiff, diff1},
        )
    )
    return bundles


def make_wraparound_round(
    val_name: str, idx_name: str, treeval_name: str, batch: int, round: int
) -> list[SymbolicBundle]:
    bundles = []
    in_wrap = vector(localize("in_wrap", batch, round))
    bundles.append(
        single_bundle(
            batch=batch,
            round=round,
            op="valu^",
            arg_names=[val_name, treeval_name],
            dest=in_wrap,
            frees={val_name, treeval_name},
        )
    )
    bundles.extend(make_hash(in_wrap, val_name, batch, round))
    return bundles


def make_last_round(
    val_name: str,
    curr_addr_name: str,
    treeval_name: str,
    batch: int,
    round: int,
) -> list[SymbolicBundle]:
    bundles = []
    in_last = vector(localize("in_last", batch, round))
    bundles.append(
        single_bundle(
            batch=batch,
            round=round,
            op="valu^",
            arg_names=[val_name, treeval_name],
            dest=in_last,
            frees={val_name, treeval_name},
        )
    )
    bundles.extend(make_hash(in_last, val_name, batch, round))
    arg_names = [curr_addr_name if batch != 0 else const("inp_values_p"), val_name]
    frees = {val_name}
    if batch != 0:
        frees.add(curr_addr_name)
    bundles.append(
        single_bundle(
            batch=batch,
            round=round,
            op="vstore",
            arg_names=arg_names,
            frees=frees,
        )
    )
    return bundles


def make_mid_round(
    val_name: str,
    idx_name: str,
    treeval_name: str,
    batch: int,
    round: int,
    rounds: int,
) -> list[SymbolicBundle]:
    bundles = []
    inX = vector(localize("inX", batch, round))
    parityX = vector(localize("parityX", batch, round))
    # this is a scalar register!
    tmp_addr = batch_localize("tmp_addr", batch)
    bundles.append(
        single_bundle(
            batch=batch,
            round=round,
            op="valu^",
            arg_names=[val_name, treeval_name],
            dest=inX,
            frees={val_name},  # don't free treeval, because it's used for gather
        )
    )
    bundles.extend(make_hash(inX, val_name, batch, round))
    bundles.append(
        multi_bundle(
            single_bundle(
                batch=batch,
                round=round,
                op="valu%",
                arg_names=[val_name, vconst("2")],
                dest=parityX,
            ),
            single_bundle(
                batch=batch,
                round=round,
                op="multiply_add",
                arg_names=[vconst("2"), idx_name, vconst("1")],
                dest=idx_name,
            ),
        )
    )
    bundles.append(
        multi_bundle(
            single_bundle(
                batch=batch,
                round=round,
                op="valu+",
                arg_names=[idx_name, parityX],
                dest=idx_name,
                frees={parityX},
            ),
        )
    )
    # gather
    # first, calc address: forest_values_p + idx[0]
    bundles.append(
        single_bundle(
            batch=batch,
            round=round,
            op="+",
            arg_names=[const("forest_values_p"), idx_name],
            dest=tmp_addr,
            vector_offsets={idx_name: 0},
        )
    )
    # if next round is wraparound or last round, free idx
    forest_height = 10
    for i in range(1, VLEN):
        frees = set()
        next_round_is_wraparound = (round + 2) % (forest_height + 1) == 0
        next_round_is_last = round + 1 == rounds - 1
        if (next_round_is_wraparound or next_round_is_last) and i == VLEN - 1:
            frees = {idx_name}
        bundles.append(
            multi_bundle(
                single_bundle(
                    batch=batch,
                    round=round,
                    op="+",
                    arg_names=[const("forest_values_p"), idx_name],
                    dest=tmp_addr,
                    vector_offsets={idx_name: i},
                    frees=frees,
                ),
                # Load from tmp_addr into offset vector reg
                single_bundle(
                    batch=batch,
                    round=round,
                    op="load",
                    arg_names=[tmp_addr],
                    dest=treeval_name,
                    vector_offsets={treeval_name: i - 1},
                    frees={tmp_addr},
                ),
            )
        )
    bundles.append(
        single_bundle(
            batch=batch,
            round=round,
            op="load",
            arg_names=[tmp_addr],
            dest=treeval_name,
            vector_offsets={treeval_name: VLEN - 1},
            frees={tmp_addr},
        )
    )
    return bundles

def make_batch_insts(
    batch: int, rounds: int, forest_height: int
) -> list[SymbolicBundle]:
    bundles = []
    curr_addr_name = batch_localize("curr_addr", batch)
    val_name = vector(batch_localize("val", batch))
    idx_name = vector(batch_localize("idx", batch))
    treeval_name = vector(batch_localize("treeval", batch))

    # Load initial value for first batch; load inp_values_p.
    bundles.extend(make_initial_load(curr_addr_name, val_name, batch, 0))
    for round in range(rounds):
        # round 0
        if round % (forest_height + 1) == 0:
            bundles.extend(make_round0(val_name, idx_name, treeval_name, batch, round))
        # round 1
        elif round % (forest_height + 1) == 1:
            bundles.extend(make_round1(val_name, idx_name, treeval_name, batch, round))
        # wraparound round
        elif (round + 1) % (forest_height + 1) == 0:
            bundles.extend(
                make_wraparound_round(val_name, idx_name, treeval_name, batch, round)
            )
        # last round
        elif round == rounds - 1:
            bundles.extend(
                make_last_round(
                    val_name,
                    curr_addr_name,
                    treeval_name,
                    batch,
                    round,
                )
            )
        else:
            bundles.extend(
                make_mid_round(val_name, idx_name, treeval_name, batch, round, rounds)
            )
    return bundles

class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def make_bundle(self, **kwargs):
        vliw_inst = {
            "alu": kwargs.get("alu", []),
            "valu": kwargs.get("valu", []),
            "load": kwargs.get("load", []),
            "store": kwargs.get("store", []),
            "flow": kwargs.get("flow", []),
            "debug": kwargs.get("debug", []),
        }
        assert all(
            len(vliw_inst[engine]) <= SLOT_LIMITS[engine] for engine in vliw_inst
        ), "instruction exceeded slot limit"
        return vliw_inst

    def add_bundle(self, **kwargs):
        self.instrs.append(self.make_bundle(**kwargs))

    # This uses an entire cycle.
    def add(self, engine, slot):
        assert engine in ("flow", "debug"), "only flow and debug can be entire cycles"
        self.instrs.append({engine: [slot]})

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def build_const_mapping(self):
        const_mapping = {}

        # Input is list of (name, concrete value).
        # We allocate scratch space, load with const, and register in const_mapping.
        # Returns the loads we need to perform.
        def build_scalar_constants(scalars: list[tuple[str, int]]):
            loads = []
            for name, val in scalars:
                reg = self.alloc_scratch()
                loads.append(("const", reg, val))
                assert name not in const_mapping, f"Constant {name} already registered"
                const_mapping[name] = reg
            return loads

        num_loads = build_scalar_constants([(str(i), i) for i in range(8)])
        vlen_loads = build_scalar_constants([("s_vlen", VLEN)])
        hash_add_loads = build_scalar_constants(
            [(f"hash_add{i}", imm) for i, (_, imm, _, _, _) in enumerate(HASH_STAGES)]
        )
        hash_mult_loads = build_scalar_constants(
            [
                (f"hash_mult{i}", 1 + 2**shift if i in (0, 2, 4) else shift)
                for i, (_, _, _, _, shift) in enumerate(HASH_STAGES)
            ]
        )

        # Input is list of names and register holding address to load from.
        # Allocate scratch space, load with vload, and register individual lanes in const_mapping.
        # Returns the vload to perform.
        def vload_const_batch(names: list[str], addr_reg: int):
            assert len(names) <= VLEN, "Too many names for vload_const_batch"
            base = self.alloc_scratch(length=VLEN)
            for i, name in enumerate(names):
                const_mapping[name] = base + i
            return ("vload", base, addr_reg)

        init_vars = [
            "rounds",
            "n_nodes",
            "batch_size",
            "forest_height",
            "forest_values_p",
            "inp_indices_p",
            "inp_values_p",
        ]
        init_vars_vload = vload_const_batch(init_vars, const_mapping["0"])
        treevals_vload = vload_const_batch(
            [f"treeval{i}" for i in range(VLEN)], const_mapping["forest_values_p"]
        )

        # Input is list of (name, reg holding scalar val).
        # We allocate scratch space, broadcast the scalar, and register in const_mapping.
        # Returns the broadcasts we need to perform.
        def build_vector_constants(vectors: list[tuple[str, int]]):
            broadcasts = []
            for name, reg in vectors:
                vbase_reg = self.alloc_scratch(length=VLEN)
                const_mapping[name] = vbase_reg
                broadcasts.append(("vbroadcast", vbase_reg, reg))
            return broadcasts

        num_vbroadcasts = build_vector_constants(
            [(f"v{i}", const_mapping[str(i)]) for i in range(4)]
        )
        treeval_vbroadcasts = build_vector_constants(
            [(f"vtreeval{i}", const_mapping[f"treeval{i}"]) for i in range(VLEN)]
        )
        hash_add_vbroadcasts = build_vector_constants(
            [
                (f"vhash_add{i}", const_mapping[f"hash_add{i}"])
                for i in range(len(HASH_STAGES))
            ]
        )
        hash_mult_vbroadcasts = build_vector_constants(
            [
                (f"vhash_mult{i}", const_mapping[f"hash_mult{i}"])
                for i in range(len(HASH_STAGES))
            ]
        )

        # valu's for diffs
        vdiff21, vdiff43, vdiff65 = [self.alloc_scratch(length=VLEN) for i in range(3)]
        vdiff_info = list(zip([vdiff21, vdiff43, vdiff65], [2, 4, 6], [1, 3, 5]))
        vdiff_valus = [
            ("-", vreg, const_mapping[f"vtreeval{i1}"], const_mapping[f"vtreeval{i2}"])
            for vreg, i1, i2 in vdiff_info
        ]
        for vreg, i1, i2 in vdiff_info:
            const_mapping[f"vdiff{i1}{i2}"] = vreg

        # --- Now how to pack these? ---
        # Longest dependency chain:
        # load 0 --> load init vars --> vload treevals --> vbroadcast treevals --> valu diffs

        # Taking a step back:
        # Scalar const loads: num_loads (4), vlen_loads (3), hash_add_loads (6), hash_mult_loads (6); there are 19 of these, and they don't have dependencies.
        # Scalar vloads:
        # - init_vars_vload: dependency on loading 0 const.
        # - treevals_vload: dependency on loading forest_values_p (init_vars_vload).
        # vbroadcasts:
        # - num_vbroadcasts: 4, dependency on loading num_loads.
        # - treeval_vbroadcasts: 8, dependency on scalar treevals (treevals_vload).
        # - hash_add_vbroadcasts: 6, dependecy on scalars (hash_add_loads).
        # - hash_mult_vbroadcasts: 6, dependency on scalars (hash_mult_loads).
        # valus:
        # - vdiff_valus: 3, dependency on vector treevals (treeval_vbroadcasts).

        # Cycle 0: load 0, 1 scalar constants.
        # Cycle 1: vload init vars, 1 vlen load, vbroadcast 0 and 1.
        # Cycle 2: vload treevals,
        MAX_CYCLES = 20
        load_bundles = [[] for _ in range(MAX_CYCLES)]
        valu_bundles = [[] for _ in range(MAX_CYCLES)]
        load_bundles[0].extend(num_loads[:2])
        load_bundles[1].extend([init_vars_vload, num_loads[2]])
        load_bundles[2].extend([treevals_vload, num_loads[3]])
        load_bundles[3].extend(num_loads[4:6])
        load_bundles[4].extend(num_loads[6:8])
        # cycles 3 and beyond: the 6 hash_add loads, then the 6 hash_mult loads, then valu
        for i in range(3):
            load_bundles[5 + i].extend(hash_add_loads[i * 2 : i * 2 + 2])
        for i in range(3):
            load_bundles[8 + i].extend(hash_mult_loads[i * 2 : i * 2 + 2])
        load_bundles[11].extend(vlen_loads)

        # load 0: 0 and 1
        # load 1: vload init vars, 2
        # load 2: vload treevals, 3
        # load 3: 4 and 5
        # load 4: 6 and 7
        # load 5: hashadds 0 and 1
        # load 6: hashadds 2 and 3
        # load 7: hashadds 4 and 5
        # load 8: hashmults 0 and 1
        # load 9: hashmults 2 and 3
        # load 10: hashmults 4 and 5
        # load 11: vlen

        valu_bundles[1].extend(num_vbroadcasts[0:2])
        # outstanding: 2
        valu_bundles[2].append(num_vbroadcasts[2])
        # outstanding: 3, 8 treevals
        valu_bundles[3].extend([treeval_vbroadcasts[i] for i in (1, 2, 3, 4, 5, 6)])
        # outstanding: 345, treevals 0 and 7
        valu_bundles[4].extend(
            vdiff_valus
            + [treeval_vbroadcasts[i] for i in (0, 7)]
            + [num_vbroadcasts[3]]
        )
        # outstanding: 4567
        valu_bundles[5].extend(num_vbroadcasts[4:8])
        for i in range(3):
            valu_bundles[6 + i].extend(hash_add_vbroadcasts[i * 2 : i * 2 + 2])
        # outstanding: hashmults 0-1
        for i in range(3):
            valu_bundles[9 + i].extend(hash_mult_vbroadcasts[i * 2 : i * 2 + 2])

        setup_cycle_count = 0
        for load_bundle, valu_bundle in zip(load_bundles, valu_bundles):
            if load_bundle or valu_bundle:
                self.add_bundle(load=load_bundle, valu=valu_bundle)
                setup_cycle_count += 1
        # print(f"setup cycle count: {setup_cycle_count}")
        return const_mapping

    # Make scratch registers. Modifies scratch allocator but does not issue instructions.
    def make_freelists(self):
        # Scalar: 8 numerical constants, 1 vlen, 12 hash constants
        # Vector regs: 1 init vars, 1 tree vals
        # Broadcast: 8 numerical, 8 treevals, 12 hash constants
        # vDiffs: 3
        # 285 constant registers.
        N_CONST_REG = (8 + 1 + 12) + 8 * (1 + 1) + 8 * (8 + 8 + 12) + 8 * 3
        # 2 scalar registers per batch: curr_addr, and one scratch
        N_SCALAR_REG = 32 * 2
        # currently 148; 4.625 vec registers per batch. probably ok
        N_VECTOR_REG = (SCRATCH_SIZE - N_CONST_REG - N_SCALAR_REG) // VLEN

        # Make the freelist. Vector and scalar are separated; we must reconcile this difference.
        scalar_freelist = set(self.alloc_scratch() for _ in range(N_SCALAR_REG))
        vector_freelist = set(
            self.alloc_scratch(length=VLEN) for _ in range(N_VECTOR_REG)
        )

        return scalar_freelist, vector_freelist

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        # makes constant mappings, as well as instructions for setup phase
        # this is only 12 cycles; simpler to just have it fully separate. not much overhead
        consts = self.build_const_mapping()

        # make register freelists. this is free (no cycles)
        scalar_freelist, vector_freelist = self.make_freelists()
        assert all(reg < 1536 for reg in scalar_freelist), (
            "scalar register is out of bounds"
        )
        assert all(reg < 1536 for reg in vector_freelist), (
            "vector register is out of bounds"
        )

        # required to match with reference
        self.add("flow", ("pause",))
        # ignored by simulator
        self.add("debug", ("comment", "Starting loop"))

        # ---- Main program ----
        bundles = []

        for batch in range(0, batch_size, VLEN):
            bundles.extend(make_batch_insts(batch, rounds, forest_height))
        prog = SymbolicProgram(bundles)
        self.instrs.extend(
            prog.to_concrete({}, scalar_freelist, vector_freelist, consts)
        )
        # Required to match with the yield in reference_kernel2
        self.instrs.append({"flow": [("pause",)]})


BASELINE = 147734


def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    # print(kb.instrs)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])
        # Updating these in memory isn't required, but you can enable this check for debugging
        # assert machine.mem[inp_indices_p:inp_indices_p+len(inp.indices)] == ref_mem[inp_indices_p:inp_indices_p+len(inp.indices)]

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    # def test_kernel_correctness(self):
    #     for batch in range(1, 3):
    #         for forest_height in range(3):
    #             do_kernel_test(
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
    #             )

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256, prints=False)


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    unittest.main()
