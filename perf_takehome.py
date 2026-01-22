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
import random
from typing import Union
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


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    # def make_bundle(self, insts):
    #     vliw_inst = defaultdict(list)
    #     for engine, slot in insts:
    #         vliw_inst[engine].append(slot)
    #     return vliw_inst

    # def build(
    #     self,
    #     slots: list[Union[tuple[Engine, tuple], list[tuple[Engine, tuple]]]],
    #     vliw: bool = False,
    # ):
    #     # If any of the individual slots are lists, converts to VLIW bundle
    #     # Simple slot packing that just uses one slot per instruction bundle
    #     instrs = []
    #     for v in slots:
    #         if isinstance(v, list):
    #             instrs.append(self.make_bundle(v))
    #             continue
    #         engine, slot = v
    #         instrs.append({engine: [slot]})
    #     return instrs

    # def add_bundle(self, insts):
    #     self.instrs.append(self.make_bundle(insts))

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

    def build_kernel(
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
        vlen_const = self.scratch_const(VLEN)
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

        # Diffs for caching + linear interpolation
        vdiff21 = self.alloc_scratch("vdiff21", VLEN)
        vdiff43 = self.alloc_scratch("vdiff43", VLEN)
        vdiff65 = self.alloc_scratch("vdiff65", VLEN)
        # Scratch registers for diffs
        vdiff1, vdiff2 = (self.alloc_scratch(f"vdiff{i}", VLEN) for i in range(2))
        vddiff = self.alloc_scratch("vddiff", VLEN)

        # ---- Scratch registers ----

        # Addresses
        tmp_addr1, tmp_addr2, tmp_addr3 = (
            self.alloc_scratch(f"tmp_addr{i + 1}") for i in range(3)
        )
        # Vector
        vidx = self.alloc_scratch("vidx", VLEN)
        vtmpidx = self.alloc_scratch("vtmpidx", VLEN)
        vval = self.alloc_scratch("vval", VLEN)
        vtreeval = self.alloc_scratch("vtreeval", VLEN)
        vparity = self.alloc_scratch("vparity", VLEN)
        vtmp1, vtmp2, vtmp3 = (
            self.alloc_scratch(f"vtmp{i + 1}", VLEN) for i in range(3)
        )

        # ---- Cache population ----

        # Preload the first 8 forest values
        vforest_vals = [
            self.alloc_scratch(f"vforest_vals{i}", VLEN) for i in range(VLEN)
        ]
        self.add("load", ("vload", vforest_vals[0], self.scratch["forest_values_p"]))
        self.add_bundle(
            valu=[
                ("vbroadcast", vforest_vals[i], vforest_vals[0] + i)
                for i in range(VLEN)
                if i >= VLEN // 2
            ]
        )
        self.add_bundle(
            valu=[
                ("vbroadcast", vforest_vals[i], vforest_vals[0] + i)
                for i in range(VLEN)
                if i < VLEN // 2
            ]
        )

        # Precompute differences between tree values
        self.add_bundle(
            valu=[
                ("-", vdiff21, vforest_vals[2], vforest_vals[1]),
                ("-", vdiff43, vforest_vals[4], vforest_vals[3]),
                ("-", vdiff65, vforest_vals[6], vforest_vals[5]),
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

        # Load initial values for first batch.
        # tmp_addr1 contains current address; used for storing
        # tmp_addr2 contains next address; used for loading
        self.add_bundle(
            alu=[
                (
                    "+",
                    tmp_addr1,
                    self.scratch["inp_values_p"],
                    zero_const,
                ),
                (
                    "+",
                    tmp_addr2,
                    self.scratch["inp_values_p"],
                    vlen_const,
                ),
            ],
            load=[("vload", vval, self.scratch["inp_values_p"])],
        )

        block = KernelBlock()  # block of instructions

        for i_base in range(0, batch_size, VLEN):
            # In this loop, assume vtreeval is loaded
            for round in range(rounds):
                # Round 0 is special case
                if round % (forest_height + 1) == 0:
                    # At the top, we use tree0 for vtreeval
                    block.add_bundle(valu=[("^", vval, vval, vforest_vals[0])])
                    block.append_block(
                        self.vbuild_hash(
                            vval,
                            vtmp1,
                            vhash_add_consts,
                            vhash_shift_consts,
                        )
                    )
                    block.add_bundle(valu=[("%", vparity, vval, vconst2)])
                    block.add_bundle(
                        valu=[
                            ("+", vidx, vconst1, vparity),
                            (
                                "multiply_add",
                                vtreeval,
                                vparity,
                                vdiff21,
                                vforest_vals[1],
                            ),
                        ]
                    )
                    continue
                # Round 1: most complicated
                if round % (forest_height + 1) == 1:
                    block.add_bundle(valu=[("^", vval, vval, vtreeval)])
                    hash_block = self.vbuild_hash(
                        vval,
                        vtmp1,
                        vhash_add_consts,
                        vhash_shift_consts,
                    )
                    hash_block.hijack_first(
                        valu=[("multiply_add", vidx, vconst2, vidx, vconst1)]
                    )
                    block.append_block(hash_block)
                    # update parity and tmpidx
                    block.add_bundle(
                        valu=[
                            ("%", vparity, vval, vconst2),
                            ("-", vtmpidx, vidx, vconst3),
                        ]
                    )
                    # compute diffs
                    block.add_bundle(
                        valu=[
                            (
                                "multiply_add",
                                vdiff1,
                                vparity,
                                vdiff43,
                                vforest_vals[3],
                            ),
                            (
                                "multiply_add",
                                vdiff2,
                                vparity,
                                vdiff65,
                                vforest_vals[5],
                            ),
                            ("+", vidx, vidx, vparity),
                        ]
                    )
                    # compute diff of diffs and get higher bit of tmpidx
                    block.add_bundle(
                        valu=[
                            ("-", vddiff, vdiff2, vdiff1),
                            (">>", vtmpidx, vtmpidx, vconst1),
                        ]
                    )
                    block.add_bundle(
                        valu=[("multiply_add", vtreeval, vtmpidx, vddiff, vdiff1)]
                    )
                    continue
                # Wraparound round
                if (round + 1) % (forest_height + 1) == 0:
                    block.add_bundle(valu=[("^", vval, vval, vtreeval)])
                    block.append_block(
                        self.vbuild_hash(
                            vval,
                            vtmp1,
                            vhash_add_consts,
                            vhash_shift_consts,
                        )
                    )
                    continue
                # Last round
                if round == rounds - 1:
                    # First do the hashing. But then we don't need to load treeval or change idx or anything.
                    block.add_bundle(valu=[("^", vval, vval, vtreeval)])
                    block.append_block(
                        self.vbuild_hash(
                            vval, vtmp1, vhash_add_consts, vhash_shift_consts
                        )
                    )
                    # tmp_addr1 has current address, used for storing; tmp_addr2 has next address, used for loading; update tmp_addr1 and tmp_addr2
                    # only issue load if there is more in batch
                    load_insts = (
                        [("vload", vval, tmp_addr2)]
                        if i_base + VLEN < batch_size
                        else []
                    )
                    block.add_bundle(
                        store=[("vstore", tmp_addr1, vval)],
                        alu=[
                            ("+", tmp_addr1, tmp_addr1, vlen_const),
                            ("+", tmp_addr2, tmp_addr2, vlen_const),
                        ],
                        load=load_insts,
                    )
                    continue

                # Round 2+: steady-state
                block.add_bundle(valu=[("^", vval, vval, vtreeval)])
                hash_block = self.vbuild_hash(
                    vval,
                    vtmp1,
                    vhash_add_consts,
                    vhash_shift_consts,
                )
                hash_block.hijack_first(
                    valu=[("multiply_add", vidx, vconst2, vidx, vconst1)]
                )
                block.append_block(hash_block)
                block.add_bundle(
                    valu=[
                        ("%", vparity, vval, vconst2),
                    ]
                )
                block.add_bundle(valu=[("+", vidx, vidx, vparity)])
                # Gather using this index; first we need to calculate address
                block.add_bundle(
                    alu=[("+", tmp_addr3, self.scratch["forest_values_p"], vidx)]
                )
                # Then overlap loads with address calc
                for i_off in range(1, VLEN):
                    block.add_bundle(
                        load=[("load", vtreeval + i_off - 1, tmp_addr3)],
                        alu=[
                            (
                                "+",
                                tmp_addr3,
                                self.scratch["forest_values_p"],
                                vidx + i_off,  # i_off'th element in vector
                            )
                        ],
                    )
                block.add_bundle(load=[("load", vtreeval + VLEN - 1, tmp_addr3)])

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
