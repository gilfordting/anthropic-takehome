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
from itertools import pairwise
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

# Variable naming stuff.
# Const has precedence over vector. Vector constants are like const vX.
CONST_PREFIX = "const "
VECTOR_PREFIX = "v"


def const(name: str) -> str:
    return f"{CONST_PREFIX}{name}"


def is_const(name: str) -> bool:
    return name.startswith(CONST_PREFIX)


def is_vector(name: str) -> bool:
    return (
        name.startswith(VECTOR_PREFIX)
        or name.startswith(CONST_PREFIX)
        and name[len(CONST_PREFIX) :].startswith(VECTOR_PREFIX)
    )


def vector(name: str) -> str:
    return f"{VECTOR_PREFIX}{name}"


def vconst(name: str) -> str:
    return const(vector(name))


# localize a name to a batch and round
def localize(name: str, batch: int, round: int) -> str:
    return f"{name}_{batch}_{round}"


def batch_localize(name: str, batch: int) -> str:
    return f"{name}_{batch}"


# Must be hashable to be added to DiGraph
@dataclass(frozen=True)
class SymbolicInstructionSlot2:
    engine: Engine
    op: str
    # dependencies are all here, minus ones that start with "const_"
    arg_names: tuple[str, ...]
    # same role as `defines`, but explicitly separated from arg list
    dest: Optional[str] = None
    # always frees these, and adds them to freelist
    # frees: Optional[set[str]] = None
    # this is used to offset into a vector reg, if defined
    vector_offsets: Optional[frozendict[str, int]] = None
    # batch: int
    # round: int
    # comment: str = None

    @property
    def dependencies(self) -> set[str]:
        return {arg for arg in self.arg_names if not is_const(arg)}

    @property
    def has_def(self) -> bool:
        return self.dest is not None


class ComputationGraph:
    # Allows us to construct a portion of a computation graph that will be merged into a larger graph later.
    # TODO: handle vector offsets.
    def __init__(self):
        self.graph = nx.DiGraph()

        # Temporary staging while we build the graph.
        self.inner_defs: dict[str, SymbolicInstructionSlot2] = {}

        # Keeps track of external dependencies. Which slots depend on a given outside variable?
        self.ext_deps: dict[str, list[SymbolicInstructionSlot2]] = defaultdict(list)
        # Defines what variables we export.
        self.exports: dict[str, SymbolicInstructionSlot2] = {}

    def add_edge(
        self, from_node: SymbolicInstructionSlot2, to_node: SymbolicInstructionSlot2
    ):
        assert from_node in self.graph, f"originating node {from_node} must be in graph"
        assert to_node in self.graph, f"destination node {to_node} must be in graph"
        self.graph.add_edge(from_node, to_node)

    # Must add slots in dependency order.
    # The last slot added to the graph must be marked. This defines the export.
    def add_slot(self, slot: SymbolicInstructionSlot2, exports=False):
        self.graph.add_node(slot)

        # Check that dependencies satisfied, and draw incoming edges
        for arg in slot.dependencies:
            # Inner defs take precedence over upper defs
            if arg in self.inner_defs:
                # Draw edge from inner def's node to this one
                # We don't need to mark with var name, because it's always the same as the originating node's `dest`
                self.add_edge(self.inner_defs[arg], slot)
                continue
            # Otherwise, we need this variable from outside our graph. Add to external dependencies; we will draw the edge later. TODO
            self.ext_deps[arg].append(slot)

        # if this slot defines a new variable, add to inner_defs
        # inner variables must be unique!
        if slot.has_def:
            assert slot.dest not in self.inner_defs, (
                f"inner variable {slot.dest} is already defined"
            )
            self.inner_defs[slot.dest] = slot

        if exports:
            self.exports[slot.dest] = slot

def make_hash_graph()

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
                for name in operands:
                    if self.is_vector_var(name):
                        assert name in self.vector_offsets, (
                            "if vector variable passed to scalar alu, var must be in vector offsets: "
                            + name
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

    @property
    def is_load(self) -> bool:
        return self.engine == "load"

    @property
    def is_scalar_load(self) -> bool:
        return self.is_load and self.is_scalar

    @property
    def is_store(self) -> bool:
        return self.engine == "store"


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
    # Used for heuristics. These are populated by a higher-level function.
    dist_to_load: int = None
    dist_to_scalar_load: int = None
    dist_to_end: int = None

    def __post_init__(self):
        # destination names must be unique
        dests = [slot.dest for slot in self.slots if slot.dest is not None]
        if len(dests) != len(set(dests)):
            # if a thing is repeated, check that it's to a vector offset that's different
            for dest in set(dests):
                offsets = []
                if dests.count(dest) <= 1:
                    continue
                # this one is repeated
                for slot in self.slots:
                    if slot.dest != dest:
                        continue
                    assert slot.vector_offsets is not None, (
                        "vector offsets must be specified for repeated destinations"
                    )
                    assert dest in slot.vector_offsets, (
                        "destination must be in vector offsets"
                    )
                    offsets.append(slot.vector_offsets[dest])
                assert len(offsets) == len(set(offsets)), (
                    "vector offsets must be different for repeated destinations"
                )

    def check_slot_condition(self) -> bool:
        for engine, limit in SLOT_LIMITS.items():
            slots_for_engine = [slot for slot in self.slots if slot.engine == engine]
            if len(slots_for_engine) > limit:
                return False
        return True

    def merge(self, other: "SymbolicBundle") -> "SymbolicBundle":
        return SymbolicBundle(slots=self.slots + other.slots)

    def can_merge(self, other: "SymbolicBundle") -> bool:
        merged = self.merge(other)
        return merged.check_slot_condition()

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

        # Figure out how to make things compatible with vector offsets?
        # Main use cases: loading to a vector offset for the main scalar loads
        # make mid round: using a specific position within a vector for scalar alu op; loading to a position within an existing vector
        # Goal: allow using scalar engine with vector offsets, for both input and output

        # Generally, this does the following:
        # Allocate *new* registers for outputs.
        # Pass these to individual slots for concrete-izing.
        # Remove freed registers from `using`, and put them on the freelist.
        # Add out_mapping to `using`.

        reg_slots = []
        vector_offset_slots = []
        for i, slot in enumerate(self.slots):
            if slot.dest is not None:
                if slot.dest in slot.vector_offsets:
                    vector_offset_slots.append((i, slot))
                else:
                    reg_slots.append((i, slot))

        for i, slot in reg_slots:
            # Always allocate a new register for the destination, if reg slot
            freelist = scalar_freelist if slot.is_scalar else vector_freelist
            assert len(freelist) > 0, (
                f"freelist is empty for slot {i}, which is batch {slot.batch} and round {slot.round}; it is scalar: {slot.is_scalar}"
            )
            out_mapping[slot.dest] = (freelist.pop(), i)

        vector_offset_dests = {}
        for i, slot in vector_offset_slots:
            if slot.dest not in vector_offset_dests:
                vector_offset_dests[slot.dest] = i
                if slot.dest not in using:
                    freelist = scalar_freelist if slot.is_scalar else vector_freelist
                    out_mapping[slot.dest] = (freelist.pop(), i)
                else:
                    out_mapping[slot.dest] = (using[slot.dest], i)

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

    @property
    def has_load(self) -> bool:
        return any(slot.is_load for slot in self.slots)

    @property
    def has_scalar_load(self) -> bool:
        return any(slot.is_scalar_load for slot in self.slots)

    @property
    def has_store(self) -> bool:
        return any(slot.is_store for slot in self.slots)


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

    def populate_heuristics(self):
        # lower is better
        dist_to_load = float("inf")  # default: -1, no load below
        dist_to_scalar_load = float("inf")
        for i, bundle in enumerate(reversed(self.bundles)):
            # if has load, 0 away! best case. reset counter
            if bundle.has_load:
                bundle.dist_to_load = 0
                dist_to_load = 0
            elif dist_to_load >= 0:
                dist_to_load += 1
                bundle.dist_to_load = dist_to_load
            else:
                bundle.dist_to_load = float("inf")
            # scalar load specifically
            if bundle.has_scalar_load:
                bundle.dist_to_scalar_load = 0
                dist_to_scalar_load = 0
            elif dist_to_scalar_load >= 0:
                dist_to_scalar_load += 1
                bundle.dist_to_scalar_load = dist_to_scalar_load
            else:
                bundle.dist_to_scalar_load = float("inf")
            bundle.dist_to_end = i

    def get_bundle(self, pc: int) -> SymbolicBundle:
        if pc >= len(self.bundles):
            return None
        return self.bundles[pc]


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


@dataclass
class VLIWScheduler:
    progs: list[SymbolicProgram]

    def schedule(
        self,
        using: dict[str, int],
        scalar_freelist: set[int],
        vector_freelist: set[int],
        consts: dict[str, int],
    ) -> list[Instruction]:
        old_scalar_freelist = {i for i in scalar_freelist}
        old_vector_freelist = {i for i in vector_freelist}
        insts = []
        for prog in self.progs:
            prog.populate_heuristics()

        prog_counters = [0] * len(self.progs)
        while any(
            pc < len(prog.bundles) for pc, prog in zip(prog_counters, self.progs)
        ):
            bundle = None
            streams = sorted(
                range(len(self.progs)),
                key=lambda x: (
                    # self.progs[x].dist_to_load,
                    self.progs[x].bundles[prog_counters[x]].dist_to_scalar_load
                    if prog_counters[x] < len(self.progs[x].bundles)
                    else float("inf"),
                    x,  # if same, just schedule in order
                ),
            )

            for i in streams:
                bundle_i = self.progs[i].get_bundle(prog_counters[i])
                if bundle_i is None:
                    continue
                if bundle is None:
                    bundle = bundle_i
                    prog_counters[i] += 1
                elif bundle.can_merge(bundle_i):
                    bundle = bundle.merge(bundle_i)
                    prog_counters[i] += 1
            assert bundle is not None, "no bundle found"
            insts.append(
                bundle.to_concrete(using, scalar_freelist, vector_freelist, consts)
            )

        # we should be back to the same
        assert len(vector_freelist) == len(old_vector_freelist)
        assert all(i in old_vector_freelist for i in vector_freelist)
        assert len(scalar_freelist) == len(old_scalar_freelist)
        assert all(i in old_scalar_freelist for i in scalar_freelist)
        return insts

        # which programs are the most promising?
        # we want to make sure load is utilized, so prioritize those.
        # but we also don't want to get stuck in a state where we just stop after the last load, and then never actually do the store.


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
    single_bundles = []
    if first_shift == 0:
        # don't have 0; 1 << i = 1 << 0 = 1
        single_bundles.append(
            single_bundle(
                batch=batch,
                round=round,
                op="-",
                arg_names=[const("2"), const("1")],
                dest=curr_addr_name,
            )
        )
    else:
        single_bundles.append(
            single_bundle(
                batch=batch,
                round=round,
                op="<<",
                arg_names=[const("1"), const(str(first_shift))],
                dest=curr_addr_name,
            )
        )
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
                        vconst(f"hash_mult{i}"),
                        vconst(f"hash_add{i}"),
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
                    arg_names=[curr, vconst(f"hash_add{i}")],
                    dest=tmp1,
                    frees={curr},
                ),
                single_bundle(
                    batch=batch,
                    round=round,
                    op=op3,
                    arg_names=[curr, vconst(f"hash_mult{i}")],
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


def make_round2(
    val_name: str, idx_name: str, treeval_name: str, batch: int, round: int
) -> list[SymbolicBundle]:
    bundles = []
    # make variables
    hash_in = vector(localize("hash_in", batch, round))
    parity = vector(localize("parity", batch, round))
    norm_idx = vector(localize("norm_idx", batch, round))
    bit0, bit1, bit2 = [vector(localize(f"bit{i}", batch, round)) for i in range(3)]
    lerp87, lerp109, lerp1211, lerp1413 = [
        vector(localize(f"lerp{i}{i - 1}", batch, round)) for i in range(8, 16, 2)
    ]
    ddiff10987, ddiff14131211 = [
        vector(localize(f"ddiff{s}", batch, round)) for s in ("10987", "14131211")
    ]
    lerp10987, lerp14131211 = [
        vector(localize(f"lerp{s}", batch, round)) for s in ("10987", "14131211")
    ]
    ddiff147 = vector(localize("ddiff147", batch, round))

    # algorithm
    bundles.append(
        single_bundle(
            batch=batch,
            round=round,
            op="valu^",
            arg_names=[val_name, treeval_name],
            dest=hash_in,
            frees={val_name, treeval_name},
        )
    )
    bundles.extend(make_hash(hash_in, val_name, batch, round))
    bundles.append(
        multi_bundle(
            single_bundle(
                batch=batch,
                round=round,
                op="multiply_add",
                arg_names=[vconst("2"), idx_name, vconst("1")],
                dest=idx_name,
            ),
            single_bundle(
                batch=batch,
                round=round,
                op="valu%",
                arg_names=[val_name, vconst("2")],
                dest=parity,
            ),
        )
    )
    bundles.append(
        single_bundle(
            batch=batch,
            round=round,
            op="valu+",
            arg_names=[idx_name, parity],
            dest=idx_name,
            frees={parity},
        )
    )
    bundles.append(
        single_bundle(
            batch=batch,
            round=round,
            op="valu-",
            arg_names=[idx_name, vconst("7")],
            dest=norm_idx,
        ),
    )
    bundles.append(
        multi_bundle(
            single_bundle(
                batch=batch,
                round=round,
                op="valu&",
                arg_names=[norm_idx, vconst("1")],
                dest=bit0,
            ),
            single_bundle(
                batch=batch,
                round=round,
                op="valu>>",
                arg_names=[norm_idx, vconst("1")],
                dest=norm_idx,
            ),
        )
    )
    bundles.append(
        multi_bundle(
            single_bundle(
                batch=batch,
                round=round,
                op="multiply_add",
                arg_names=[bit0, vconst("diff87"), vconst("treeval7")],
                dest=lerp87,
            ),
            single_bundle(
                batch=batch,
                round=round,
                op="multiply_add",
                arg_names=[bit0, vconst("diff109"), vconst("treeval9")],
                dest=lerp109,
            ),
            single_bundle(
                batch=batch,
                round=round,
                op="multiply_add",
                arg_names=[bit0, vconst("diff1211"), vconst("treeval11")],
                dest=lerp1211,
                frees={bit0},
            ),
            single_bundle(
                batch=batch,
                round=round,
                op="multiply_add",
                arg_names=[bit0, vconst("diff1413"), vconst("treeval13")],
                dest=lerp1413,
                frees={bit0},
            ),
        )
    )
    bundles.append(
        multi_bundle(
            single_bundle(
                batch=batch,
                round=round,
                op="valu&",
                arg_names=[norm_idx, vconst("1")],
                dest=bit1,
            ),
            single_bundle(
                batch=batch,
                round=round,
                op="valu>>",
                arg_names=[norm_idx, vconst("1")],
                dest=norm_idx,
            ),
            single_bundle(
                batch=batch,
                round=round,
                op="valu-",
                arg_names=[lerp109, lerp87],
                dest=ddiff10987,
                frees={lerp109},
            ),
            single_bundle(
                batch=batch,
                round=round,
                op="valu-",
                arg_names=[lerp1413, lerp1211],
                dest=ddiff14131211,
                frees={lerp1413},
            ),
        )
    )
    bundles.append(
        multi_bundle(
            single_bundle(
                batch=batch,
                round=round,
                op="multiply_add",
                arg_names=[bit1, ddiff10987, lerp87],
                dest=lerp10987,
                frees={bit1, lerp87, ddiff10987},
            ),
            single_bundle(
                batch=batch,
                round=round,
                op="multiply_add",
                arg_names=[bit1, ddiff14131211, lerp1211],
                dest=lerp14131211,
                frees={bit1, lerp1211, ddiff14131211},
            ),
        )
    )
    bundles.append(
        multi_bundle(
            single_bundle(
                batch=batch,
                round=round,
                op="valu&",
                arg_names=[norm_idx, vconst("1")],
                dest=bit2,
                frees={norm_idx},
            ),
            single_bundle(
                batch=batch,
                round=round,
                op="valu-",
                arg_names=[lerp14131211, lerp10987],
                dest=ddiff147,
                frees={lerp14131211},
            ),
        )
    )
    bundles.append(
        single_bundle(
            batch=batch,
            round=round,
            op="multiply_add",
            arg_names=[bit2, ddiff147, lerp10987],
            dest=treeval_name,
            frees={bit2, ddiff147, lerp10987},
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
    tmp_addr1 = batch_localize("tmp_addr1", batch)
    tmp_addr2 = batch_localize("tmp_addr2", batch)
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
    bundles.extend(
        [
            single_bundle(
                batch=batch,
                round=round,
                op="+",
                arg_names=[const("forest_values_p"), idx_name],
                dest=tmp_addr1,
                vector_offsets={idx_name: 0},
            ),
            single_bundle(
                batch=batch,
                round=round,
                op="+",
                arg_names=[const("forest_values_p"), idx_name],
                dest=tmp_addr2,
                vector_offsets={idx_name: 1},
            ),
        ]
    )
    # if next round is wraparound or last round, free idx
    forest_height = 10
    next_round_is_wraparound = (round + 2) % (forest_height + 1) == 0
    next_round_is_last = round + 1 == rounds - 1
    should_free_idx = next_round_is_wraparound or next_round_is_last
    for i in range(2, VLEN, 2):
        frees = set()

        if should_free_idx and i == VLEN - 2:
            frees = {idx_name}
        # always load, regardless
        single_bundles = [
            single_bundle(
                batch=batch,
                round=round,
                op="load",
                arg_names=[tmp_addr1],
                dest=treeval_name,
                vector_offsets={treeval_name: i - 2},
                frees={tmp_addr1},
            ),
            single_bundle(
                batch=batch,
                round=round,
                op="load",
                arg_names=[tmp_addr2],
                dest=treeval_name,
                vector_offsets={treeval_name: i - 1},
                frees={tmp_addr2},
            ),
        ]
        # load tmp_addr for all but last iteration
        # for the last time we load tmp_addr, we release idx
        single_bundles.extend(
            [
                single_bundle(
                    batch=batch,
                    round=round,
                    op="+",
                    arg_names=[const("forest_values_p"), idx_name],
                    dest=tmp_addr1,
                    vector_offsets={idx_name: i},
                    frees=frees,
                ),
                single_bundle(
                    batch=batch,
                    round=round,
                    op="+",
                    arg_names=[const("forest_values_p"), idx_name],
                    dest=tmp_addr2,
                    vector_offsets={idx_name: i + 1},
                    frees=frees,
                ),
            ]
        )

        bundles.append(multi_bundle(*single_bundles))
    # residual load
    bundles.append(
        multi_bundle(
            single_bundle(
                batch=batch,
                round=round,
                op="load",
                arg_names=[tmp_addr1],
                dest=treeval_name,
                vector_offsets={treeval_name: VLEN - 2},
                frees={tmp_addr1},
            ),
            single_bundle(
                batch=batch,
                round=round,
                op="load",
                arg_names=[tmp_addr2],
                dest=treeval_name,
                vector_offsets={treeval_name: VLEN - 1},
                frees={tmp_addr2},
            ),
        )
    )
    return bundles


def make_mid_round_bad(
    val_name: str,
    idx_name: str,
    treeval_name: str,
    batch: int,
    round: int,
    rounds: int,
    forest_height: int,
) -> list[SymbolicBundle]:
    bundles = []
    inX = vector(localize("inX", batch, round))
    parityX = vector(localize("parityX", batch, round))
    val_addrs = vector(localize("val_addrs", batch, round))
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
    # first calc address, vectorized
    frees = set()
    next_round_is_wraparound = (round + 2) % (forest_height + 1) == 0
    next_round_is_last = round + 1 == rounds - 1
    if next_round_is_wraparound or next_round_is_last:
        frees.add(idx_name)
    bundles.append(
        multi_bundle(
            *[
                single_bundle(
                    batch=batch,
                    round=round,
                    op="+",
                    arg_names=[const("forest_values_p"), idx_name],
                    dest=val_addrs,
                    frees=frees,
                    vector_offsets={val_addrs: i, idx_name: i},
                )
                for i in range(VLEN)
            ]
        )
    )
    # then we do loads
    # offset in val_addrs and treeval_name
    # for i in range(0, VLEN, 2):
    #     frees = set()
    #     if i == VLEN - 2:
    #         frees.add(val_addrs)
    #     bundles.append(
    #         multi_bundle(
    #             single_bundle(
    #                 batch=batch,
    #                 round=round,
    #                 op="load",
    #                 arg_names=[val_addrs],
    #                 dest=treeval_name,
    #                 vector_offsets={treeval_name: i, val_addrs: i},
    #                 frees=frees,
    #             ),
    #             single_bundle(
    #                 batch=batch,
    #                 round=round,
    #                 op="load",
    #                 arg_names=[val_addrs],
    #                 dest=treeval_name,
    #                 vector_offsets={treeval_name: i + 1, val_addrs: i + 1},
    #                 frees=frees,
    #             ),
    #         )
    #     )
    for i in range(VLEN):
        frees = set()
        if i == VLEN - 1:
            frees.add(val_addrs)
        bundles.append(
            single_bundle(
                batch=batch,
                round=round,
                op="load",
                arg_names=[val_addrs],
                dest=treeval_name,
                vector_offsets={treeval_name: i, val_addrs: i},
                frees=frees,
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
        # elif round % (forest_height + 1) == 2:
        #     bundles.extend(make_round2(val_name, idx_name, treeval_name, batch, round))
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
                make_mid_round_bad(
                    val_name,
                    idx_name,
                    treeval_name,
                    batch,
                    round,
                    rounds,
                    forest_height,
                )
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

        # The only const registers we need to retain are:
        # scalar: forest_values_p, inp_values_p, 01234, s_vlen
        # vector: 1237, treeval0135791113, diff21, 43, 65, 87, 109, 1211, 1413, hash_mults and hash_adds

        # Notably:
        # - forest_values_p and inp_values_p are part of init_vars, can return the other 6 scalar regs
        # - 7 is loaded in a scalar, we broadcast it, and then can return the scalar reg.
        # - We load treevals 0-15, compute 7 diffs, and then throw away 2, 4, 6, 8, 10, 12, 14.
        # hash_adds and hash_mults are loaded in scalars and broadcasted; we can return the scalar regs.

        # Game plan:
        # Scalar loads: 012347, s_vlen, hash vals
        # Vector loads: init vars, treevals starting 0 and 8
        # Vector broadcasts: 1237, treevals 0 to 14, hash vals
        # vdiffs: diff21, 43, 65, 87, 109, 1211, 1413

        # Can throw away:
        # scalar: 7, init vars except for forest/inp_p, hash vals (?) we might have to come back to this if we want to be able to offload to regular ALU. TODO fix
        # vector: treeval vectors at 0 and 8 (staging area), vtreeval 2, 4, 8, 12, 14

        # Input is list of (name, concrete value).
        # We allocate scratch space, load with const, and register in const_mapping.
        # Returns the loads we need to perform, as a map from name to load instruction.
        def build_scalar_constants(scalars: list[tuple[str, int]]):
            loads = {}
            for name, val in scalars:
                reg = self.alloc_scratch()
                loads[name] = ("const", reg, val)
                assert name not in const_mapping, f"Constant {name} already registered"
                const_mapping[name] = reg
            return loads

        SCALARS = list(range(5)) + [7]

        num_loads = build_scalar_constants([(str(i), i) for i in SCALARS])
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
        treevals0_vload = vload_const_batch(
            [f"treeval{i}" for i in range(VLEN)], const_mapping["forest_values_p"]
        )

        # use 7's reg as scratch after broadcasting and before returning
        treeval8_addr_calc = (
            "+",
            const_mapping["7"],
            const_mapping["forest_values_p"],
            const_mapping["s_vlen"],
        )
        treevals8_vload = vload_const_batch(
            [f"treeval{i}" for i in range(VLEN, 2 * VLEN)], const_mapping["7"]
        )

        # Input is list of (name, reg holding scalar val).
        # We allocate scratch space, broadcast the scalar, and register in const_mapping.
        # Returns the broadcasts we need to perform, as a map from name to broadcast instruction.
        def build_vector_constants(vectors: list[tuple[str, int]]):
            broadcasts = {}
            for name, reg in vectors:
                vbase_reg = self.alloc_scratch(length=VLEN)
                const_mapping[name] = vbase_reg
                broadcasts[name] = ("vbroadcast", vbase_reg, reg)
            return broadcasts

        num_vbroadcasts = build_vector_constants(
            [(f"v{i}", const_mapping[str(i)]) for i in (1, 2, 3, 7)]
        )
        treeval_vbroadcasts = build_vector_constants(
            [
                (f"vtreeval{i}", const_mapping[f"treeval{i}"])
                for i in range(0, VLEN * 2 - 1)
            ]
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
        forest_vals_vbroadcasts = build_vector_constants(
            [("vforest_values_p", const_mapping["forest_values_p"])]
        )

        # valu's for diffs
        vdiff21, vdiff43, vdiff65, vdiff87, vdiff109, vdiff1211, vdiff1413 = [
            self.alloc_scratch(length=VLEN) for i in range(7)
        ]
        vdiff_info = list(
            zip(
                [vdiff21, vdiff43, vdiff65, vdiff87, vdiff109, vdiff1211, vdiff1413],
                [2, 4, 6, 8, 10, 12, 14],
                [1, 3, 5, 7, 9, 11, 13],
            )
        )
        vdiff_valus = {
            f"vdiff{i1}{i2}": (
                "-",
                vreg,
                const_mapping[f"vtreeval{i1}"],
                const_mapping[f"vtreeval{i2}"],
            )
            for vreg, i1, i2 in vdiff_info
        }
        for s, (_, vreg, i1, i2) in vdiff_valus.items():
            const_mapping[s] = vreg

        scalars_to_return = (
            ["0", "7"]
            + [s for s in init_vars if s not in ("forest_values_p", "inp_values_p")]
            + [f"hash_add{i}" for i in range(len(HASH_STAGES))]
            + [f"hash_mult{i}" for i in range(len(HASH_STAGES))]
        )
        scalar_return = {const_mapping.pop(s) for s in scalars_to_return}

        vectors_to_return = ["treeval0", "treeval8"] + [
            f"vtreeval{i}" for i in (2, 4, 8, 12, 14)
        ]
        vector_return = {const_mapping.pop(v) for v in vectors_to_return}
        for i in range(1, VLEN):
            del const_mapping[f"treeval{i}"]
        for i in range(9, VLEN * 2):
            del const_mapping[f"treeval{i}"]

        MAX_CYCLES = 20
        load_bundles = [[] for _ in range(MAX_CYCLES)]
        valu_bundles = [[] for _ in range(MAX_CYCLES)]
        alu_bundles = [[] for _ in range(MAX_CYCLES)]
        # cycle 0: load 0, 7
        load_bundles[0].extend([num_loads["0"], num_loads["7"]])
        # cycle 1: vload init vars, s_vlen; vbroadcast 7
        load_bundles[1].extend([init_vars_vload, vlen_loads["s_vlen"]])
        valu_bundles[1].extend([num_vbroadcasts["v7"]])
        # cycle 2: vload treevals 0-7, load 1 ; alu 7 = forest_values + s_vlen
        load_bundles[2].extend([treevals0_vload, num_loads["1"]])
        alu_bundles[2].extend([treeval8_addr_calc])
        # cycle 3: vload treevals 8-15, load 2; vbroadcast treevals 0 to 5
        load_bundles[3].extend([treevals8_vload, num_loads["2"]])
        valu_bundles[3].extend([treeval_vbroadcasts[f"vtreeval{i}"] for i in range(6)])
        # cycle 4: load hashval0, 1; vbroadcast treevals 6 to 11;
        load_bundles[4].extend([hash_add_loads[f"hash_add{i}"] for i in range(2)])
        valu_bundles[4].extend(
            [treeval_vbroadcasts[f"vtreeval{i}"] for i in range(6, 12)]
        )
        # cycle 5: load hashval2, 3; vbroadcast treevals 12 to 14, valu diff21, 43, 65
        load_bundles[5].extend([hash_add_loads[f"hash_add{i}"] for i in range(2, 4)])
        valu_bundles[5].extend(
            [treeval_vbroadcasts[f"vtreeval{i}"] for i in range(12, 15)]
            + [vdiff_valus[f"vdiff{i + 1}{i}"] for i in range(1, 6, 2)]  # 1 3 5
        )
        # cycle 6: load hashval4, 5; valu diff87, 109, 1211, 1413, vbroadcast 1, 2
        load_bundles[6].extend([hash_add_loads[f"hash_add{i}"] for i in range(4, 6)])
        valu_bundles[6].extend(
            [vdiff_valus[f"vdiff{i + 1}{i}"] for i in range(7, 14, 2)]  # 7 9 11 13
            + [num_vbroadcasts[f"v{i}"] for i in range(1, 3)]
        )
        # cycle 7: load hash_mult 0, 1; vbroadcast hash_add 0-5
        load_bundles[7].extend([hash_mult_loads[f"hash_mult{i}"] for i in range(2)])
        valu_bundles[7].extend(
            [hash_add_vbroadcasts[f"vhash_add{i}"] for i in range(6)]
        )
        # cycle 8: load hash_mult 2, 3; vbroadcast forest_values_p
        load_bundles[8].extend([hash_mult_loads[f"hash_mult{i}"] for i in range(2, 4)])
        valu_bundles[8].extend([forest_vals_vbroadcasts["vforest_values_p"]])
        # cycle 9: load hash_mult 4, 5
        load_bundles[9].extend([hash_mult_loads[f"hash_mult{i}"] for i in range(4, 6)])
        # cycle 10: load 3, 4; vbroadcast hash_mult 0-5
        load_bundles[10].extend([num_loads["3"], num_loads["4"]])
        valu_bundles[10].extend(
            [hash_mult_vbroadcasts[f"vhash_mult{i}"] for i in range(6)]
        )
        # cycle 11: vbroadcast 3
        valu_bundles[11].extend([num_vbroadcasts["v3"]])

        setup_cycle_count = 0
        for load_bundle, valu_bundle, alu_bundle in zip(
            load_bundles, valu_bundles, alu_bundles
        ):
            if load_bundle or valu_bundle:
                self.add_bundle(load=load_bundle, valu=valu_bundle, alu=alu_bundle)
                setup_cycle_count += 1
        return const_mapping, scalar_return, vector_return

    # Make scratch registers. Modifies scratch allocator but does not issue instructions.
    def make_freelists(self, scalar_return, vector_return):
        scratch_left = SCRATCH_SIZE - self.scratch_ptr
        # 2 scalar registers per batch: curr_addr, and one scratch
        N_SCALAR_REG = 32 * 3
        num_extra_scalar = N_SCALAR_REG - len(scalar_return)
        # take the rest for vector regs
        num_vector_alloc = (scratch_left - num_extra_scalar) // VLEN

        # Make the freelist. Vector and scalar are separated; we must reconcile this difference.
        scalar_freelist = set(self.alloc_scratch() for _ in range(num_extra_scalar))
        scalar_freelist |= scalar_return
        vector_freelist = set(
            self.alloc_scratch(length=VLEN) for _ in range(num_vector_alloc)
        )
        vector_freelist |= vector_return
        return scalar_freelist, vector_freelist

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        # makes constant mappings, as well as instructions for setup phase
        # this is only 12 cycles; simpler to just have it fully separate. not much overhead
        consts, scalar_return, vector_return = self.build_const_mapping()

        # make register freelists. this is free (no cycles)
        scalar_freelist, vector_freelist = self.make_freelists(
            scalar_return, vector_return
        )
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
        progs = []
        for batch in range(0, batch_size, VLEN):
            progs.append(
                SymbolicProgram(make_batch_insts(batch, rounds, forest_height))
            )
        scheduler = VLIWScheduler(progs)
        insts = scheduler.schedule({}, scalar_freelist, vector_freelist, consts)
        self.instrs.extend(insts)

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
