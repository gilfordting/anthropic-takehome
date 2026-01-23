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
from typing import Optional, Union
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

class KernelBlock:
    def __init__(self):
        self.instrs = []

    def add_bundle(self, **kwargs):
        vliw_inst = {
            "alu": kwargs.get("alu", []),
            "valu": kwargs.get("valu", []),
            "load": kwargs.get("load", []),
            "store": kwargs.get("store", []),
            "flow": kwargs.get("flow", []),
            "debug": kwargs.get("debug", []),
        }
        self.instrs.append(vliw_inst)

    def append_block(self, other: "KernelBlock"):
        self.instrs.extend(other.instrs)

    # Hijack the first instruction in the block; add the given slots.
    def hijack_first(self, **kwargs):
        for engine in ("alu", "valu", "load", "store", "flow", "debug"):
            if engine in kwargs:
                self.instrs[0][engine].extend(kwargs[engine])
            assert len(self.instrs[0][engine]) <= SLOT_LIMITS[engine], (
                "Hijacking exceeded slot limit"
            )

    def interleave(self, other: "KernelBlock") -> "KernelBlock":
        assert len(self.instrs) == len(other.instrs), (
            "Blocks must have the same number of instructions"
        )
        block = KernelBlock()
        for i in range(len(self.instrs)):
            merged_instr = defaultdict(list)
            for engine in ("alu", "valu", "load", "store", "flow", "debug"):
                merged_instr[engine].extend(self.instrs[i][engine])
                merged_instr[engine].extend(other.instrs[i][engine])
            block.instrs.append(merged_instr)
        return block

# Symbolic structures use names, instead of concrete registers.
# We can safely instantiate one of these like val = val + 1 and it will automatically add `val` to `frees`, triggering behavior to use different registers.
@dataclass(frozen=True)
class SymbolicInstructionSlot:
    engine: Engine
    op: str
    # same role as `defines`, but explicitly separated from arg list
    dest: Optional[str] = None
    # dependencies are all here, minus ones that start with "const_"
    args: tuple[str, ...]
    # always frees these, and adds them to freelist
    frees: Optional[frozenset[str]] = None

    def __post_init__(self):
        if self.frees is None:
            self.frees = set()
        # If `dest` reuses a symbolic register, use different registers for input and output.
        # In order to trigger this behavior, add `dest` to `frees` if not already present.
        if self.dest is not None and self.dest in self.args:
            self.frees.add(self.dest)
        # you can't free constant registers
        assert all(not name.startswith("const_") for name in self.frees), (
            "cannot free constant registers"
        )
        # and you can only free registers you actually used
        assert all(reg in self.args for reg in self.frees), (
            "cannot free registers you didn't use"
        )

    def to_concrete(
        self, in_mapping: dict[str, int], out_mapping: dict[str, int]
    ) -> (Engine, tuple):
        # check that all args are in the mapping
        assert all(name in in_mapping for name in self.args), "arguments not in mapping"
        full_args = [in_mapping[name] for name in self.args]
        # check that dest in mapping
        if self.dest is not None:
            assert self.dest in out_mapping, "non-None dest not in output mapping"
            full_args = [out_mapping[self.dest]] + full_args
        return self.engine, (self.op, *full_args)


def make_slot(
    engine: Engine,
    op: str,
    args: list[str],
    dest: Optional[str] = None,
    frees: Optional[set[str]] = None,
) -> SymbolicInstructionSlot:
    return SymbolicInstructionSlot(
        engine=engine,
        op=op,
        args=tuple(args),
        dest=dest,
        frees=frozenset(frees) if frees is not None else None,
    )


@dataclass
class SymbolicBundle:
    slots: list[SymbolicInstructionSlot]

    def __post_init__(self):
        # destination names must be unique
        dests = [slot.dest for slot in self.slots if slot.dest is not None]
        assert len(dests) == len(set(dests)), "destination names must be unique"

    def to_concrete(self, using: dict[str, int], freelist: set[int]) -> Instruction:
        # make the out mapping: take stuff from freelist and use it for destinations
        out_mapping = {
            slot.dest: freelist.pop() for slot in self.slots if slot.dest is not None
        }
        inst = defaultdict(list)
        for slot in self.slots:
            engine, args = slot.to_concrete(using, out_mapping)
            inst[engine].append(args)
        # after this, we can safely free registers; aggregate first so we don't try to free the same name twice
        all_frees = set()
        for slot in self.slots:
            all_frees |= slot.frees
        # check that all the registers we're trying to free are actually available to be freed
        assert all(name in using for name in all_frees), (
            "cannot free registers that are not available"
        )
        freelist |= {using.pop(name) for name in all_frees}

        # check that we have a valid instruction
        for engine in inst:
            assert len(inst[engine]) <= SLOT_LIMITS[engine], (
                "instruction exceeded slot limit"
            )

        return inst


def single_bundle(
    engine: Engine,
    op: str,
    args: list[str],
    dest: Optional[str] = None,
    frees: Optional[set[str]] = None,
) -> SymbolicBundle:
    return SymbolicBundle(slots=[make_slot(engine, op, args, dest, frees)])


def multi_bundle(*args: SymbolicBundle) -> SymbolicBundle:
    return SymbolicBundle(slots=sum([bundle.slots for bundle in args], []))


# in_name is provided, and we reuse it for a lot of stuff
def make_hash(
    in_name: str, out_name: str, batch: int, round: int
) -> list[SymbolicBundle]:
    bundles = []
    curr = in_name
    for i, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
        # (curr op1 val1) op2 (curr op3 val3)
        if i in (0, 2, 4):
            bundles.append(
                single_bundle(
                    engine="valu",
                    op="multiply_add",
                    args=[curr, f"const_hash_multiply_{i}", f"const_hash_add_{i}"],
                    dest=curr,
                )
            )
            continue
        tmp1 = f"hash_tmp1_{i}_{batch}_{round}"
        tmp2 = f"hash_tmp2_{i}_{batch}_{round}"
        bundles.append(
            multi_bundle(
                single_bundle(
                    engine="valu",
                    op=op1,
                    args=[curr, f"const_hash_add_{i}"],
                    dest=tmp1,
                    frees={curr},
                ),
                single_bundle(
                    engine="valu",
                    op=op3,
                    args=[curr, f"const_hash_multiply_{i}"],
                    dest=tmp2,
                    frees={curr},
                ),
            )
        )
        if i == 5:
            curr = out_name
        bundles.append(
            single_bundle(
                engine="valu", op=op2, args=[tmp1, tmp2], dest=curr, frees={tmp1, tmp2}
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

    def add_bundle(self, **kwargs):
        vliw_inst = {
            "alu": kwargs.get("alu", []),
            "valu": kwargs.get("valu", []),
            "load": kwargs.get("load", []),
            "store": kwargs.get("store", []),
            "flow": kwargs.get("flow", []),
            "debug": kwargs.get("debug", []),
        }
        self.instrs.append(vliw_inst)

    def add_bundle_list(self, vliw_insts: list[Instruction]):
        self.instrs.extend(vliw_insts)

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i):
        slots = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append(("alu", (op1, tmp1, val_hash_addr, self.scratch_const(val1))))
            slots.append(("alu", (op3, tmp2, val_hash_addr, self.scratch_const(val3))))
            slots.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))
            slots.append(
                ("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi)))
            )

        return slots

    def build_kernel_ref(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Like reference_kernel2 but building actual instructions.
        Scalar implementation using only scalar ALU and load/store.
        """
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")
        tmp3 = self.alloc_scratch("tmp3")
        # Scratch space addresses
        init_vars = [
            "rounds",
            "n_nodes",
            "batch_size",
            "forest_height",
            "forest_values_p",
            "inp_indices_p",
            "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))

        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)

        # Pause instructions are matched up with yield statements in the reference
        # kernel to let you debug at intermediate steps. The testing harness in this
        # file requires these match up to the reference kernel's yields, but the
        # submission harness ignores them.
        self.add("flow", ("pause",))
        # Any debug engine instruction is ignored by the submission simulator
        self.add("debug", ("comment", "Starting loop"))

        body = []  # array of slots

        # Scalar scratch registers
        tmp_idx = self.alloc_scratch("tmp_idx")
        tmp_val = self.alloc_scratch("tmp_val")
        tmp_node_val = self.alloc_scratch("tmp_node_val")
        tmp_addr = self.alloc_scratch("tmp_addr")

        for round in range(rounds):
            for i in range(batch_size):
                i_const = self.scratch_const(i)
                # idx = mem[inp_indices_p + i]
                body.append(
                    ("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], i_const))
                )
                body.append(("load", ("load", tmp_idx, tmp_addr)))
                body.append(("debug", ("compare", tmp_idx, (round, i, "idx"))))
                # val = mem[inp_values_p + i]
                body.append(
                    ("alu", ("+", tmp_addr, self.scratch["inp_values_p"], i_const))
                )
                body.append(("load", ("load", tmp_val, tmp_addr)))
                body.append(("debug", ("compare", tmp_val, (round, i, "val"))))
                # node_val = mem[forest_values_p + idx]
                body.append(
                    ("alu", ("+", tmp_addr, self.scratch["forest_values_p"], tmp_idx))
                )
                body.append(("load", ("load", tmp_node_val, tmp_addr)))
                body.append(
                    ("debug", ("compare", tmp_node_val, (round, i, "node_val")))
                )
                # val = myhash(val ^ node_val)
                body.append(("alu", ("^", tmp_val, tmp_val, tmp_node_val)))
                body.extend(self.build_hash(tmp_val, tmp1, tmp2, round, i))
                body.append(("debug", ("compare", tmp_val, (round, i, "hashed_val"))))
                # idx = 2*idx + (1 if val % 2 == 0 else 2)
                body.append(("alu", ("%", tmp1, tmp_val, two_const)))
                body.append(("alu", ("==", tmp1, tmp1, zero_const)))
                body.append(("flow", ("select", tmp3, tmp1, one_const, two_const)))
                body.append(("alu", ("*", tmp_idx, tmp_idx, two_const)))
                body.append(("alu", ("+", tmp_idx, tmp_idx, tmp3)))
                body.append(("debug", ("compare", tmp_idx, (round, i, "next_idx"))))
                # idx = 0 if idx >= n_nodes else idx
                body.append(("alu", ("<", tmp1, tmp_idx, self.scratch["n_nodes"])))
                body.append(("flow", ("select", tmp_idx, tmp1, tmp_idx, zero_const)))
                body.append(("debug", ("compare", tmp_idx, (round, i, "wrapped_idx"))))
                # mem[inp_indices_p + i] = idx
                body.append(
                    ("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], i_const))
                )
                body.append(("store", ("store", tmp_addr, tmp_idx)))
                # mem[inp_values_p + i] = val
                body.append(
                    ("alu", ("+", tmp_addr, self.scratch["inp_values_p"], i_const))
                )
                body.append(("store", ("store", tmp_addr, tmp_val)))

        body_instrs = self.build(body)
        self.instrs.extend(body_instrs)
        # Required to match with the yield in reference_kernel2
        self.instrs.append({"flow": [("pause",)]})

    def build_kernel_draft1(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        # ---- Global constants ----
        init_vars = [
            "rounds",
            "n_nodes",
            "batch_size",
            "forest_height",
            "forest_values_p",
            "inp_indices_p",
            "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        _ = self.alloc_scratch()  # Alignment
        # There are 8 of these, so we can just do a vectorized load once we have a 0 to work with
        zero_const, one_const, two_const, three_const = (
            self.scratch_const(i) for i in range(4)
        )
        # This loads our variables from constant mem into scratch space
        # TODO: bundleize adds, scratch_const, etc. (anything not alloc_scratch)
        self.add("load", ("vload", 0, zero_const))

        # ---- Constants ----
        vlen_const, vlen2_const, vlen3_const = [
            self.scratch_const((i + 1) * VLEN) for i in range(3)
        ]
        vconst0, vconst1, vconst2, vconst3 = (
            self.alloc_scratch(f"vconst{i}", VLEN) for i in range(4)
        )
        self.add_bundle(
            valu=[
                ("vbroadcast", vconst0, zero_const),
                ("vbroadcast", vconst1, one_const),
                ("vbroadcast", vconst2, two_const),
                ("vbroadcast", vconst3, three_const),
            ],
        )

        # Hash constants
        hash_add_consts, vhash_add_consts = {}, {}
        hash_shift_consts, vhash_shift_consts = {}, {}
        broadcast_slots = []
        for i in range(len(HASH_STAGES)):
            _, const, _, _, shift = HASH_STAGES[i]
            hash_add_consts[i] = self.scratch_const(const)
            hash_shift_consts[i] = self.scratch_const(
                1 + 2**shift if i in (0, 2, 4) else shift
            )
            vhash_add_consts[i] = self.alloc_scratch(f"vhash_add_consts{i}", VLEN)
            vhash_shift_consts[i] = self.alloc_scratch(f"vhash_mult_consts{i}", VLEN)
            broadcast_slots.extend(
                [
                    ("vbroadcast", vhash_add_consts[i], hash_add_consts[i]),
                    ("vbroadcast", vhash_shift_consts[i], hash_shift_consts[i]),
                ]
            )
        self.add_bundle(valu=broadcast_slots[:6])
        self.add_bundle(valu=broadcast_slots[6:])

        # Constant diffs for caching + linear interpolation
        vdiff21_const = self.alloc_scratch("vdiff21_const", VLEN)
        vdiff43_const = self.alloc_scratch("vdiff43_const", VLEN)
        vdiff65_const = self.alloc_scratch("vdiff65_const", VLEN)

        # ---- Scratch registers ----
        # One per batch
        N_BATCH = 2

        # Addresses
        tmp_addr1, tmp_addr2, tmp_addr3 = (
            [self.alloc_scratch(f"tmp_addr{i + 1}_batch{b}") for b in range(N_BATCH)]
            for i in range(3)
        )
        # Vector
        vidx = [self.alloc_scratch(f"vidx_batch{b}", VLEN) for b in range(N_BATCH)]
        vtmpidx = [
            self.alloc_scratch(f"vtmpidx_batch{b}", VLEN) for b in range(N_BATCH)
        ]
        vval = [self.alloc_scratch(f"vval_batch{b}", VLEN) for b in range(N_BATCH)]
        vtreeval = [
            self.alloc_scratch(f"vtreeval_batch{b}", VLEN) for b in range(N_BATCH)
        ]
        vparity = [
            self.alloc_scratch(f"vparity_batch{b}", VLEN) for b in range(N_BATCH)
        ]
        vtmp1, vtmp2, vtmp3 = (
            [self.alloc_scratch(f"vtmp{i + 1}_batch{b}", VLEN) for b in range(N_BATCH)]
            for i in range(3)
        )
        # Scratch registers for diffs
        vdiff1, vdiff2 = (
            [self.alloc_scratch(f"vdiff{i}_batch{b}", VLEN) for b in range(N_BATCH)]
            for i in range(2)
        )
        vddiff = [self.alloc_scratch(f"vddiff_batch{b}", VLEN) for b in range(N_BATCH)]

        # ---- Cache population ----

        # Preload the first 8 forest values
        vforest_vals_const = [
            self.alloc_scratch(f"vforest_vals{i}", VLEN) for i in range(VLEN)
        ]
        self.add(
            "load", ("vload", vforest_vals_const[0], self.scratch["forest_values_p"])
        )
        self.add_bundle(
            valu=[
                ("vbroadcast", vforest_vals_const[i], vforest_vals_const[0] + i)
                for i in range(VLEN)
                if i >= VLEN // 2
            ]
        )
        self.add_bundle(
            valu=[
                ("vbroadcast", vforest_vals_const[i], vforest_vals_const[0] + i)
                for i in range(VLEN)
                if i < VLEN // 2
            ]
        )
        # Precompute differences between tree values
        self.add_bundle(
            valu=[
                ("-", vdiff21_const, vforest_vals_const[2], vforest_vals_const[1]),
                ("-", vdiff43_const, vforest_vals_const[4], vforest_vals_const[3]),
                ("-", vdiff65_const, vforest_vals_const[6], vforest_vals_const[5]),
            ]
        )

        # Pause instructions are matched up with yield statements in the reference
        # kernel to let you debug at intermediate steps. The testing harness in this
        # file requires these match up to the reference kernel's yields, but the
        # submission harness ignores them.
        self.add("flow", ("pause",))
        # Any debug engine instruction is ignored by the submission simulator
        self.add("debug", ("comment", "Starting loop"))

        # ---- Main program ----

        # TODO: now, let's try two batches at once!

        # Load initial values for first batch.
        # tmp_addr1 contains current address; used for storing
        # tmp_addr2 contains next address; used for loading
        self.add_bundle(
            alu=[
                (
                    "+",
                    tmp_addr1[0],
                    self.scratch["inp_values_p"],
                    zero_const,
                ),
                (
                    "+",
                    tmp_addr2[0],
                    self.scratch["inp_values_p"],
                    vlen2_const,
                ),
                ("+", tmp_addr1[1], self.scratch["inp_values_p"], vlen_const),
                ("+", tmp_addr2[1], self.scratch["inp_values_p"], vlen3_const),
            ],
            load=[
                ("vload", vval[0], self.scratch["inp_values_p"]),
            ],
        )
        self.add_bundle(load=[("vload", vval[1], tmp_addr1[1])])

        block = KernelBlock()  # block of instructions

        def batch_instrs(i_base, i_sub):
            block = KernelBlock()
            for round in range(rounds):
                # Round 0 is special case
                if round % (forest_height + 1) == 0:
                    # At the top, we use tree0 for vtreeval
                    block.add_bundle(
                        valu=[("^", vval[i_sub], vval[i_sub], vforest_vals_const[0])]
                    )
                    block.append_block(
                        self.vbuild_hash(
                            vval[i_sub],
                            vtmp1[i_sub],
                            vhash_add_consts,
                            vhash_shift_consts,
                        )
                    )
                    block.add_bundle(valu=[("%", vparity[i_sub], vval[i_sub], vconst2)])
                    block.add_bundle(
                        valu=[
                            ("+", vidx[i_sub], vconst1, vparity[i_sub]),
                            (
                                "multiply_add",
                                vtreeval[i_sub],
                                vparity[i_sub],
                                vdiff21_const,
                                vforest_vals_const[1],
                            ),
                        ]
                    )
                    continue
                # Round 1: most complicated
                if round % (forest_height + 1) == 1:
                    block.add_bundle(
                        valu=[("^", vval[i_sub], vval[i_sub], vtreeval[i_sub])]
                    )
                    hash_block = self.vbuild_hash(
                        vval[i_sub],
                        vtmp1[i_sub],
                        vhash_add_consts,
                        vhash_shift_consts,
                    )
                    hash_block.hijack_first(
                        valu=[
                            ("multiply_add", vidx[i_sub], vconst2, vidx[i_sub], vconst1)
                        ]
                    )
                    block.append_block(hash_block)
                    # update parity and tmpidx
                    block.add_bundle(
                        valu=[
                            ("%", vparity[i_sub], vval[i_sub], vconst2),
                            ("-", vtmpidx[i_sub], vidx[i_sub], vconst3),
                        ]
                    )
                    # compute diffs
                    block.add_bundle(
                        valu=[
                            (
                                "multiply_add",
                                vdiff1[i_sub],
                                vparity[i_sub],
                                vdiff43_const,
                                vforest_vals_const[3],
                            ),
                            (
                                "multiply_add",
                                vdiff2[i_sub],
                                vparity[i_sub],
                                vdiff65_const,
                                vforest_vals_const[5],
                            ),
                            ("+", vidx[i_sub], vidx[i_sub], vparity[i_sub]),
                        ]
                    )
                    # compute diff of diffs and get higher bit of tmpidx
                    block.add_bundle(
                        valu=[
                            ("-", vddiff[i_sub], vdiff2[i_sub], vdiff1[i_sub]),
                            (">>", vtmpidx[i_sub], vtmpidx[i_sub], vconst1),
                        ]
                    )
                    block.add_bundle(
                        valu=[
                            (
                                "multiply_add",
                                vtreeval[i_sub],
                                vtmpidx[i_sub],
                                vddiff[i_sub],
                                vdiff1[i_sub],
                            )
                        ]
                    )
                    continue
                # Wraparound round
                if (round + 1) % (forest_height + 1) == 0:
                    block.add_bundle(
                        valu=[("^", vval[i_sub], vval[i_sub], vtreeval[i_sub])]
                    )
                    block.append_block(
                        self.vbuild_hash(
                            vval[i_sub],
                            vtmp1[i_sub],
                            vhash_add_consts,
                            vhash_shift_consts,
                        )
                    )
                    continue
                # Last round
                if round == rounds - 1:
                    # First do the hashing. But then we don't need to load treeval or change idx or anything.
                    block.add_bundle(
                        valu=[("^", vval[i_sub], vval[i_sub], vtreeval[i_sub])]
                    )
                    block.append_block(
                        self.vbuild_hash(
                            vval[i_sub],
                            vtmp1[i_sub],
                            vhash_add_consts,
                            vhash_shift_consts,
                        )
                    )
                    # tmp_addr1 has current address, used for storing; tmp_addr2 has next address, used for loading; update tmp_addr1 and tmp_addr2
                    # only issue load if there is more in batch
                    load_insts = (
                        [("vload", vval[i_sub], tmp_addr2[i_sub])]
                        if i_base + 2 * VLEN < batch_size
                        else []
                    )
                    block.add_bundle(
                        store=[("vstore", tmp_addr1[i_sub], vval[i_sub])],
                        alu=[
                            ("+", tmp_addr1[i_sub], tmp_addr1[i_sub], vlen2_const),
                            ("+", tmp_addr2[i_sub], tmp_addr2[i_sub], vlen2_const),
                        ],
                        load=load_insts,
                    )
                    continue

                # Round 2+: steady-state
                block.add_bundle(
                    valu=[("^", vval[i_sub], vval[i_sub], vtreeval[i_sub])]
                )
                hash_block = self.vbuild_hash(
                    vval[i_sub],
                    vtmp1[i_sub],
                    vhash_add_consts,
                    vhash_shift_consts,
                )
                hash_block.hijack_first(
                    valu=[("multiply_add", vidx[i_sub], vconst2, vidx[i_sub], vconst1)]
                )
                block.append_block(hash_block)
                block.add_bundle(
                    valu=[
                        ("%", vparity[i_sub], vval[i_sub], vconst2),
                    ]
                )
                block.add_bundle(valu=[("+", vidx[i_sub], vidx[i_sub], vparity[i_sub])])
                # Gather using this index; first we need to calculate address
                block.add_bundle(
                    alu=[
                        (
                            "+",
                            tmp_addr3[i_sub],
                            self.scratch["forest_values_p"],
                            vidx[i_sub],
                        )
                    ]
                )
                # Then overlap loads with address calc
                for i_off in range(1, VLEN):
                    block.add_bundle(
                        load=[("load", vtreeval[i_sub] + i_off - 1, tmp_addr3[i_sub])],
                        alu=[
                            (
                                "+",
                                tmp_addr3[i_sub],
                                self.scratch["forest_values_p"],
                                vidx[i_sub] + i_off,  # i_off'th element in vector
                            )
                        ],
                    )
                block.add_bundle(
                    load=[("load", vtreeval[i_sub] + VLEN - 1, tmp_addr3[i_sub])]
                )
            return block

        for i_base in range(0, batch_size, 2 * VLEN):
            blocks = [batch_instrs(i_base, i_sub) for i_sub in range(2)]
            block.append_block(blocks[0].interleave(blocks[1]))

        self.instrs.extend(block.instrs)
        # Required to match with the yield in reference_kernel2
        self.instrs.append({"flow": [("pause",)]})

    def build_const_mapping(self):
        const_mapping = {}
        # How does this work?
        # We have scalar constants, and then we vectorize them.
        # Scalar constants:
        # - all the provided variables. rounds, n_nodes, batch_size, forest_height, forest_values_p, inp_indices_p, inp_values_p
        # -
        # - numerical: 0, 1, 2, 3
        # -

        # Function to allocate scratch space for vectorized constants, broadcast them, and register them in const_mapping.
        # Takes in list of names to associate with these constants, and the originating scalar registers that hold the values.
        def vectorize_consts(names: list[str], regs: list[int]):
            assert len(names) == len(regs), (
                "Names and registers must have the same length"
            )
            broadcast_slots = []
            for name, val_reg in zip(names, regs):
                std_name = f"const_v{name}"
                vbase_reg = self.alloc_scratch(std_name, VLEN)
                const_mapping[std_name] = vbase_reg
                broadcast_slots.append(
                    (
                        "vbroadcast",
                        vbase_reg,
                        val_reg,
                    )
                )
            return broadcast_slots

        # ---- Global constants ----
        init_vars = [
            "const_rounds",
            "const_n_nodes",
            "const_batch_size",
            "const_forest_height",
            "const_forest_values_p",
            "const_inp_indices_p",
            "const_inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        for i, v in enumerate(init_vars):
            const_mapping[v] = self.scratch[v]
        _ = self.alloc_scratch()  # Alignment
        # There are 8 of these, so we can just do a vectorized load once we have a 0 to work with
        num_consts = [self.scratch_const(i) for i in range(4)]  # TODO fix
        for i, reg in enumerate(num_consts):
            # const_0, const_1, const_2, const_3 -> regs for 0, 1, 2, 3
            const_mapping[f"const_{i}"] = reg
        # This loads our variables from constant mem into scratch space
        self.add("load", ("vload", 0, num_consts[0]))  # load into the zero'th slot
        # Constants for vlen, vlen*2, vlen*3
        vlens = [self.scratch_const((i + 1) * VLEN) for i in range(3)]
        for i, reg in enumerate(vlens):
            const_mapping[f"const_vlen{i}"] = reg

        # Next is all the vector constants.

        # numerical: 0, 1, 2, 3
        # makes const_v0, const_v1, const_v2, const_v3 etc
        slots = vectorize_consts([str(i) for i in range(4)], num_consts)
        # hash: makes const_hash_add_i, const_hash_multiply_i, etc for 0 to 5
        hash_add_consts, hash_multiply_consts = {}, {}
        for i in range(len(HASH_STAGES)):
            _, const, _, _, shift = HASH_STAGES[i]
            hash_add_consts[i] = self.scratch_const(const)
            hash_multiply_consts[i] = self.scratch_const(
                1 + 2**shift if i in (0, 2, 4) else shift
            )
        slots += vectorize_consts(
            [f"hash_add_{i}" for i in range(len(HASH_STAGES))],
            [hash_add_consts[i] for i in range(len(HASH_STAGES))],
        )
        slots += vectorize_consts(
            [f"hash_multiply_{i}" for i in range(len(HASH_STAGES))],
            [hash_multiply_consts[i] for i in range(len(HASH_STAGES))],
        )
        # first 8 forest values
        vforest_vals = self.alloc_scratch("vforest_vals", VLEN)
        self.add(
            "load", ("vload", vforest_vals, const_mapping["const_forest_values_p"])
        )

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        # TODO: bundle scratch_const
        # bundle a lot of stuff tbh
        const_mapping = self.build_const_mapping()

        # Constant diffs for caching + linear interpolation
        const_vdiff21 = self.alloc_scratch("const_vdiff21", VLEN)
        const_vdiff43 = self.alloc_scratch("const_vdiff43", VLEN)
        const_vdiff65 = self.alloc_scratch("const_vdiff65", VLEN)
        # Preload the first 8 forest values
        const_vforest_vals = [
            self.alloc_scratch(f"vforest_vals{i}", VLEN) for i in range(VLEN)
        ]
        self.add(
            "load", ("vload", vforest_vals_const[0], self.scratch["forest_values_p"])
        )
        self.add_bundle(
            valu=[
                ("vbroadcast", vforest_vals_const[i], vforest_vals_const[0] + i)
                for i in range(VLEN)
                if i >= VLEN // 2
            ]
        )
        self.add_bundle(
            valu=[
                ("vbroadcast", vforest_vals_const[i], vforest_vals_const[0] + i)
                for i in range(VLEN)
                if i < VLEN // 2
            ]
        )
        # Precompute differences between tree values
        self.add_bundle(
            valu=[
                ("-", vdiff21_const, vforest_vals_const[2], vforest_vals_const[1]),
                ("-", vdiff43_const, vforest_vals_const[4], vforest_vals_const[3]),
                ("-", vdiff65_const, vforest_vals_const[6], vforest_vals_const[5]),
            ]
        )

        # ---- Scratch registers ----
        # One per batch
        N_BATCH = 2

        # Addresses
        tmp_addr1, tmp_addr2, tmp_addr3 = (
            [self.alloc_scratch(f"tmp_addr{i + 1}_batch{b}") for b in range(N_BATCH)]
            for i in range(3)
        )
        # Vector
        vidx = [self.alloc_scratch(f"vidx_batch{b}", VLEN) for b in range(N_BATCH)]
        vtmpidx = [
            self.alloc_scratch(f"vtmpidx_batch{b}", VLEN) for b in range(N_BATCH)
        ]
        vval = [self.alloc_scratch(f"vval_batch{b}", VLEN) for b in range(N_BATCH)]
        vtreeval = [
            self.alloc_scratch(f"vtreeval_batch{b}", VLEN) for b in range(N_BATCH)
        ]
        vparity = [
            self.alloc_scratch(f"vparity_batch{b}", VLEN) for b in range(N_BATCH)
        ]
        vtmp1, vtmp2, vtmp3 = (
            [self.alloc_scratch(f"vtmp{i + 1}_batch{b}", VLEN) for b in range(N_BATCH)]
            for i in range(3)
        )
        # Scratch registers for diffs
        vdiff1, vdiff2 = (
            [self.alloc_scratch(f"vdiff{i}_batch{b}", VLEN) for b in range(N_BATCH)]
            for i in range(2)
        )
        vddiff = [self.alloc_scratch(f"vddiff_batch{b}", VLEN) for b in range(N_BATCH)]

        # Pause instructions are matched up with yield statements in the reference
        # kernel to let you debug at intermediate steps. The testing harness in this
        # file requires these match up to the reference kernel's yields, but the
        # submission harness ignores them.
        self.add("flow", ("pause",))
        # Any debug engine instruction is ignored by the submission simulator
        self.add("debug", ("comment", "Starting loop"))

        # ---- Main program ----

        # TODO: now, let's try two batches at once!

        # Load initial values for first batch.
        # tmp_addr1 contains current address; used for storing
        # tmp_addr2 contains next address; used for loading
        self.add_bundle(
            alu=[
                (
                    "+",
                    tmp_addr1[0],
                    self.scratch["inp_values_p"],
                    zero_const,
                ),
                (
                    "+",
                    tmp_addr2[0],
                    self.scratch["inp_values_p"],
                    vlen2_const,
                ),
                ("+", tmp_addr1[1], self.scratch["inp_values_p"], vlen_const),
                ("+", tmp_addr2[1], self.scratch["inp_values_p"], vlen3_const),
            ],
            load=[
                ("vload", vval[0], self.scratch["inp_values_p"]),
            ],
        )
        self.add_bundle(load=[("vload", vval[1], tmp_addr1[1])])

        block = KernelBlock()  # block of instructions

        def batch_instrs(i_base, i_sub):
            block = KernelBlock()
            for round in range(rounds):
                # Round 0 is special case
                if round % (forest_height + 1) == 0:
                    # At the top, we use tree0 for vtreeval
                    block.add_bundle(
                        valu=[("^", vval[i_sub], vval[i_sub], vforest_vals_const[0])]
                    )
                    block.append_block(
                        self.vbuild_hash(
                            vval[i_sub],
                            vtmp1[i_sub],
                            vhash_add_consts,
                            vhash_shift_consts,
                        )
                    )
                    block.add_bundle(valu=[("%", vparity[i_sub], vval[i_sub], vconst2)])
                    block.add_bundle(
                        valu=[
                            ("+", vidx[i_sub], vconst1, vparity[i_sub]),
                            (
                                "multiply_add",
                                vtreeval[i_sub],
                                vparity[i_sub],
                                vdiff21_const,
                                vforest_vals_const[1],
                            ),
                        ]
                    )
                    continue
                # Round 1: most complicated
                if round % (forest_height + 1) == 1:
                    block.add_bundle(
                        valu=[("^", vval[i_sub], vval[i_sub], vtreeval[i_sub])]
                    )
                    hash_block = self.vbuild_hash(
                        vval[i_sub],
                        vtmp1[i_sub],
                        vhash_add_consts,
                        vhash_shift_consts,
                    )
                    hash_block.hijack_first(
                        valu=[
                            ("multiply_add", vidx[i_sub], vconst2, vidx[i_sub], vconst1)
                        ]
                    )
                    block.append_block(hash_block)
                    # update parity and tmpidx
                    block.add_bundle(
                        valu=[
                            ("%", vparity[i_sub], vval[i_sub], vconst2),
                            ("-", vtmpidx[i_sub], vidx[i_sub], vconst3),
                        ]
                    )
                    # compute diffs
                    block.add_bundle(
                        valu=[
                            (
                                "multiply_add",
                                vdiff1[i_sub],
                                vparity[i_sub],
                                vdiff43_const,
                                vforest_vals_const[3],
                            ),
                            (
                                "multiply_add",
                                vdiff2[i_sub],
                                vparity[i_sub],
                                vdiff65_const,
                                vforest_vals_const[5],
                            ),
                            ("+", vidx[i_sub], vidx[i_sub], vparity[i_sub]),
                        ]
                    )
                    # compute diff of diffs and get higher bit of tmpidx
                    block.add_bundle(
                        valu=[
                            ("-", vddiff[i_sub], vdiff2[i_sub], vdiff1[i_sub]),
                            (">>", vtmpidx[i_sub], vtmpidx[i_sub], vconst1),
                        ]
                    )
                    block.add_bundle(
                        valu=[
                            (
                                "multiply_add",
                                vtreeval[i_sub],
                                vtmpidx[i_sub],
                                vddiff[i_sub],
                                vdiff1[i_sub],
                            )
                        ]
                    )
                    continue
                # Wraparound round
                if (round + 1) % (forest_height + 1) == 0:
                    block.add_bundle(
                        valu=[("^", vval[i_sub], vval[i_sub], vtreeval[i_sub])]
                    )
                    block.append_block(
                        self.vbuild_hash(
                            vval[i_sub],
                            vtmp1[i_sub],
                            vhash_add_consts,
                            vhash_shift_consts,
                        )
                    )
                    continue
                # Last round
                if round == rounds - 1:
                    # First do the hashing. But then we don't need to load treeval or change idx or anything.
                    block.add_bundle(
                        valu=[("^", vval[i_sub], vval[i_sub], vtreeval[i_sub])]
                    )
                    block.append_block(
                        self.vbuild_hash(
                            vval[i_sub],
                            vtmp1[i_sub],
                            vhash_add_consts,
                            vhash_shift_consts,
                        )
                    )
                    # tmp_addr1 has current address, used for storing; tmp_addr2 has next address, used for loading; update tmp_addr1 and tmp_addr2
                    # only issue load if there is more in batch
                    load_insts = (
                        [("vload", vval[i_sub], tmp_addr2[i_sub])]
                        if i_base + 2 * VLEN < batch_size
                        else []
                    )
                    block.add_bundle(
                        store=[("vstore", tmp_addr1[i_sub], vval[i_sub])],
                        alu=[
                            ("+", tmp_addr1[i_sub], tmp_addr1[i_sub], vlen2_const),
                            ("+", tmp_addr2[i_sub], tmp_addr2[i_sub], vlen2_const),
                        ],
                        load=load_insts,
                    )
                    continue

                # Round 2+: steady-state
                block.add_bundle(
                    valu=[("^", vval[i_sub], vval[i_sub], vtreeval[i_sub])]
                )
                hash_block = self.vbuild_hash(
                    vval[i_sub],
                    vtmp1[i_sub],
                    vhash_add_consts,
                    vhash_shift_consts,
                )
                hash_block.hijack_first(
                    valu=[("multiply_add", vidx[i_sub], vconst2, vidx[i_sub], vconst1)]
                )
                block.append_block(hash_block)
                block.add_bundle(
                    valu=[
                        ("%", vparity[i_sub], vval[i_sub], vconst2),
                    ]
                )
                block.add_bundle(valu=[("+", vidx[i_sub], vidx[i_sub], vparity[i_sub])])
                # Gather using this index; first we need to calculate address
                block.add_bundle(
                    alu=[
                        (
                            "+",
                            tmp_addr3[i_sub],
                            self.scratch["forest_values_p"],
                            vidx[i_sub],
                        )
                    ]
                )
                # Then overlap loads with address calc
                for i_off in range(1, VLEN):
                    block.add_bundle(
                        load=[("load", vtreeval[i_sub] + i_off - 1, tmp_addr3[i_sub])],
                        alu=[
                            (
                                "+",
                                tmp_addr3[i_sub],
                                self.scratch["forest_values_p"],
                                vidx[i_sub] + i_off,  # i_off'th element in vector
                            )
                        ],
                    )
                block.add_bundle(
                    load=[("load", vtreeval[i_sub] + VLEN - 1, tmp_addr3[i_sub])]
                )
            return block

        for i_base in range(0, batch_size, 2 * VLEN):
            blocks = [batch_instrs(i_base, i_sub) for i_sub in range(2)]
            block.append_block(blocks[0].interleave(blocks[1]))

        self.instrs.extend(block.instrs)
        # Required to match with the yield in reference_kernel2
        self.instrs.append({"flow": [("pause",)]})

    # For the two-stage hashing operation, put one intermediate in vval and one in vtmp.
    # if vout is specified, then the final result is stored in vout.
    def vbuild_hash(
        self, vval, vtmp, vhash_add_consts, vhash_shift_consts, vout=None
    ) -> KernelBlock:
        block = KernelBlock()
        if vout is None:
            vout = vval
        # (vval op1 val1) op2 (vval op3 val3)
        # val1 and val3 are precomputed/prebroadcasted
        for i, (op1, _, op2, op3, _) in enumerate(HASH_STAGES):
            if i in (0, 2, 4):
                block.add_bundle(
                    valu=[
                        (
                            "multiply_add",
                            vout,
                            vval,
                            vhash_shift_consts[i],
                            vhash_add_consts[i],
                        )
                    ]
                )
                continue
            # Otherwise, do two ops in parallel, then combine
            block.add_bundle(
                valu=[
                    (op1, vval, vval, vhash_add_consts[i]),
                    (op3, vtmp, vval, vhash_shift_consts[i]),
                ]
            )
            dst = vout if i == 5 else vval
            block.add_bundle(valu=[(op2, dst, vval, vtmp)])
        return block


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
