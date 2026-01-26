"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

Validate your results using `python tests/submission_tests.py` without modifying
anything in the tests/ folder.

We recommend you look through problem.py next.
"""

from collections import defaultdict
from dataclasses import dataclass
import random
from typing import Literal, Optional, Union
import unittest
import networkx as nx

from problem import (
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

Engine = Literal["alu", "valu", "load", "store", "flow", "debug", "NO-OP"]


class FullComputationGraph:
    def __init__(
        self,
        graph: nx.DiGraph,
        scalar_freelist: set[int],
        vector_freelist: set[int],
        consts: dict[str, int],
    ):
        self.graph = graph
        self.using = {}
        self.scalar_freelist = scalar_freelist
        self.vector_freelist = vector_freelist
        self.consts = consts
        self.refcounts = defaultdict(int)
        self.distance_to_notable = {}
        self.longest_dep_chain = {}
        self.__post_init__()

    def __post_init__(self):
        assert all(
            self.graph.in_degree(node) + self.graph.out_degree(node) > 0
            for node in self.graph.nodes
        )
        self.prune_noops()
        self.populate_heuristics()

    def populate_heuristics(self):
        self.distance_to_notable = {}
        self.longest_dep_chain = {}
        for node in reversed(list(nx.topological_sort(self.graph))):
            self.longest_dep_chain[node] = (
                max(
                    (
                        self.longest_dep_chain[succ]
                        for succ in self.graph.successors(node)
                    ),
                    default=0,
                )
                + 1
            )
            if node.is_notable:
                self.distance_to_notable[node] = 0
            else:
                self.distance_to_notable[node] = (
                    min(
                        (
                            self.distance_to_notable[succ]
                            for succ in self.graph.successors(node)
                        ),
                        default=100,
                    )
                    + 1
                )

    def get_leaves(self) -> list[SymbolicInstructionSlot]:
        return [node for node in self.graph.nodes if self.graph.in_degree(node) == 0]

    def get_stores(self) -> list[SymbolicInstructionSlot]:
        return [node for node in self.graph.nodes if node.engine == "store"]

    def prune_noops(self):
        leaves = self.get_leaves()
        for leaf in leaves:
            if leaf.engine == "NO-OP":
                assert leaf.op == "vmerge"
                # Do the same reftracking stuff -- which slots depend on this one?
                n_forward_deps = self.graph.out_degree(leaf)
                assert n_forward_deps > 0
                assert leaf.dest not in self.refcounts
                self.refcounts[leaf.dest] = n_forward_deps
                self.graph.remove_node(leaf)

    @property
    def has_more_work(self) -> bool:
        return self.graph.number_of_nodes() > 0

    # Allocate output reg. If partial, allocate only once.
    def alloc_outreg(self, name: str) -> int:
        if is_partial(name):
            name = get_partial_vname(name)
            if name in self.using:
                return self.using[name]
            reg = self.vector_freelist.pop()
            self.using[name] = reg
            return reg
        if is_vector(name):
            if len(self.vector_freelist) == 0:
                print("using:", self.using)
            reg = self.vector_freelist.pop()
            self.using[name] = reg
            return reg
        assert is_scalar(name)
        reg = self.scalar_freelist.pop()
        self.using[name] = reg
        return reg

    # Frees the register associated with a given variable name.
    def free_outreg(self, name: str):
        # We can't free partial registers; this is fine because the vmerge noop pruning will free the register for us.
        # The underlying vector register gets allocated when the first vload_scalar is scheduled.
        # Then vmerge will define the same variable as the partial vars, and everything works!
        if is_partial(name):
            assert get_partial_vname(name) in self.using
            return
        assert name in self.using
        if is_vector(name):
            self.vector_freelist.add(self.using.pop(name))
            return
        assert is_scalar(name)
        self.scalar_freelist.add(self.using.pop(name))

    def schedule(self) -> Instruction:
        self.prune_noops()
        assert self.has_more_work
        bundle = defaultdict(list)
        seen_dests = set()

        def can_schedule(leaf: SymbolicInstructionSlot) -> bool:
            if leaf.engine == "alu":
                return len(bundle["alu"]) < SLOT_LIMITS["alu"]
            if leaf.engine == "valu":
                return len(bundle["valu"]) < SLOT_LIMITS["valu"]
            if leaf.engine == "load":
                return len(bundle["load"]) < SLOT_LIMITS["load"]
            if leaf.engine == "store":
                return len(bundle["store"]) < SLOT_LIMITS["store"]
            if leaf.engine == "flow":
                return len(bundle["flow"]) < SLOT_LIMITS["flow"]
            if leaf.engine == "debug":
                return len(bundle["debug"]) < SLOT_LIMITS["debug"]
            return False

        leaves = self.get_leaves()
        # Sort leaves by longest dependency chain, so we can schedule the ones that are closest to a notable node first. This will hopefully improve load utilization.
        # Also, lower distance to notable is better. So subtract it.
        leaves.sort(
            key=lambda x: -self.distance_to_notable[x],
            reverse=True,
        )
        # TODO: prioritize valu leaves first. Then we'll naturally have like fill valus -> offload full -> offload half -> schedule ALUs?????
        one_full_offload = False
        one_half_offload = False
        for leaf in leaves:
            alu_offload = False
            alu_half_offload = False
            if not can_schedule(leaf):
                if (
                    leaf.engine == "valu"
                    and len(bundle["valu"]) == SLOT_LIMITS["valu"]
                    and leaf.op not in ("multiply_add", "vbroadcast")
                ):
                    if len(bundle["alu"]) < SLOT_LIMITS["alu"] - 8:
                        # we can offload to alu!
                        alu_offload = True
                        one_full_offload = True
                    else:
                        # If we've already done a full offload, we could still offload half of this one.
                        # But we should only do this once per scheduling round.
                        if one_full_offload:
                            if not one_half_offload:
                                alu_half_offload = True
                                one_half_offload = True
                            else:
                                continue
                        else:
                            continue
                else:
                    continue

            if alu_half_offload:
                # If half offload, we need to turn this slot into 8 slots + 1 vmerge noop.
                # We'll just modify the graph in place, and move on. Next round will handle it.
                op = leaf.op
                a1, a2 = leaf.arg_names
                partials = [make_partial(leaf.dest, i) for i in range(8)]
                partial_leaves = [
                    make_slot("valu_scalar", [op, a1, a2, i], dest=dest)
                    for i, dest in enumerate(partials)
                ]
                vmerge_noop = make_slot("vmerge", partials, dest=leaf.dest)
                # first, add vmerge to the graph. give it the same forward deps as the original leaf, and the same heuristic scores
                self.graph.add_node(vmerge_noop)
                for succ in self.graph.successors(leaf):
                    self.graph.add_edge(vmerge_noop, succ, var=leaf.dest)
                self.distance_to_notable[vmerge_noop] = self.distance_to_notable[leaf]
                self.longest_dep_chain[vmerge_noop] = self.longest_dep_chain[leaf]
                # we can safely remove original node now
                self.graph.remove_node(leaf)
                # then, add partials to the graph
                for partial_leaf in partial_leaves:
                    self.graph.add_node(partial_leaf)
                    self.graph.add_edge(
                        partial_leaf, vmerge_noop, var=partial_leaf.dest
                    )
                    # the only forward dep is vmerge noop, so update scores accordingly
                    self.distance_to_notable[partial_leaf] = (
                        self.distance_to_notable[vmerge_noop] + 1
                    )
                    self.longest_dep_chain[partial_leaf] = (
                        self.longest_dep_chain[vmerge_noop] + 1
                    )
                # Also dependency tracking.
                for arg in leaf.dependencies:
                    # the dependency has gone from 1 to 8
                    # so we need to update the dependency tracking
                    self.refcounts[arg] += 7
                continue

            # Allocate output register and concrete-ize the slot
            out_reg = None
            # If we have multiple things defining the same variable, something is wrong
            if leaf.dest is not None:
                assert leaf.dest not in seen_dests
                seen_dests.add(leaf.dest)
                out_reg = self.alloc_outreg(leaf.dest)

            # Main logic
            if alu_offload:
                insts = leaf.to_concrete_alu_offload(
                    self.using, self.consts, out_reg=out_reg
                )
                bundle["alu"].extend(insts)
            # elif alu_half_offload:
            #     # If half offload, we need to turn this slot into 8 slots + 1 vmerge noop.

            #     insts = leaf.to_concrete_alu_offload(
            #         self.using, self.consts, out_reg=out_reg
            #     )
            #     bundle["alu"].extend(insts)
            else:
                engine, inst = leaf.to_concrete(
                    self.using, self.consts, out_reg=out_reg
                )
                assert engine != "NO-OP"
                bundle[engine].append(inst)

            # Reftracking: what slots depend on this one?
            n_forward_deps = 0
            for _, _, var_name in self.graph.out_edges(leaf, data=True):
                if var_name == "FAKE":
                    continue
                n_forward_deps += 1

            if leaf.dest is not None and n_forward_deps > 0:
                assert leaf.dest not in self.refcounts, (
                    f"leaf.dest {leaf.dest} already in refcounts, from leaf {leaf}"
                )
                self.refcounts[leaf.dest] = n_forward_deps

            # Remove this leaf, expose more work/leaves
            self.graph.remove_node(leaf)

            # # We can also free registers from our arguments, if refcount reaches 0
            for arg in leaf.dependencies:
                self.refcounts[arg] -= 1
                if self.refcounts[arg] == 0:
                    reg = self.using.pop(arg)
                    if is_vector(arg):
                        self.vector_freelist.add(reg)
                    else:
                        self.scalar_freelist.add(reg)

        return bundle

    def generate_code(self):
        code = []
        orig_sfreelist_len = len(self.scalar_freelist)
        orig_vfreelist_len = len(self.vector_freelist)
        while self.has_more_work:
            bundle = self.schedule()
            code.append(bundle)
        assert len(self.scalar_freelist) == orig_sfreelist_len
        assert len(self.vector_freelist) == orig_vfreelist_len
        return code


def make_hash_graph(
    batch: int, round: int, in_name: str, out_name: str
) -> ComputationGraph:
    def vlocalize(name: str, hash_round: int) -> str:
        return vector(f"{localize(name, batch, round)}_hashround{hash_round}")

    graph = ComputationGraph()

    curr_in_name = in_name
    for i, (op1, _, op2, op3, _) in enumerate(HASH_STAGES):
        # Convert to valu operations
        op1, op2, op3 = (f"valu{op}" for op in (op1, op2, op3))
        # Make this round's output variable
        curr_out_name = vlocalize("out", i)

        if i in (0, 2, 4):
            args = [curr_in_name, vconst(f"hash_mult{i}"), vconst(f"hash_add{i}")]
            graph.add_new_slot(
                "multiply_add",
                args,
                dest=curr_out_name,
            )
        else:
            last_stage = i == len(HASH_STAGES) - 1
            tmp1 = vlocalize("tmp1", i)
            tmp2 = vlocalize("tmp2", i)
            graph.add_new_slot(op1, [curr_in_name, vconst(f"hash_add{i}")], dest=tmp1)
            graph.add_new_slot(op3, [curr_in_name, vconst(f"hash_mult{i}")], dest=tmp2)
            graph.add_new_slot(
                op2,
                [tmp1, tmp2],
                dest=out_name if last_stage else curr_out_name,
                exports=last_stage,
            )
        curr_in_name = curr_out_name

    assert len(graph.ext_deps) == 1 and in_name in graph.ext_deps
    assert len(graph.exports) == 1 and out_name in graph.exports
    return graph


def make_round0_graph(
    batch: int, round, val_in: str, val_out: str, idx_out: str, treeval_out: str
) -> ComputationGraph:
    def vlocalize_r0(name: str) -> str:
        return vlocalize(
            name,
            batch,
            round,
        )

    graph = ComputationGraph()
    hash_in = vlocalize_r0("hash_in")
    parity = vlocalize_r0("parity")
    graph.add_new_slot("valu^", [val_in, vconst("treeval0")], dest=hash_in)
    graph.merge_below(make_hash_graph(batch, round, hash_in, val_out))
    graph.exports[val_out] = graph.inner_defs[val_out]
    graph.add_new_slot("valu%", [val_out, vconstn(2)], dest=parity)
    graph.add_new_slot("valu+", [parity, vconstn(1)], dest=idx_out, exports=True)
    graph.add_new_slot(
        "multiply_add",
        [parity, vconst("diff21"), vconst("treeval1")],
        dest=treeval_out,
        exports=True,
    )

    assert len(graph.ext_deps) == 1 and val_in in graph.ext_deps
    assert len(graph.exports) == 3 and all(
        name in graph.exports for name in (val_out, idx_out, treeval_out)
    ), f"exports {graph.exports} do not match (val_out, idx_out, treeval_out)"
    return graph


def make_round1_graph_alu(
    batch: int,
    round: int,
    val_in: str,
    val_out: str,
    idx_in: str,
    idx_out: str,
    treeval_in: str,
    treeval_out: str,
):
    def vlocalize_r1(name: str) -> str:
        return vlocalize(
            name,
            batch,
            round,
        )

    hash_in = vlocalize_r1("hash_in")
    idx_tmp = vlocalize_r1("idx_tmp")
    parity = vlocalize_r1("parity")
    norm_idx = vlocalize_r1("norm_idx")
    norm_idx_down1 = vlocalize_r1("norm_idx_down1")
    bit0 = vlocalize_r1("bit0")
    bit1 = vlocalize_r1("bit1")
    lerp43 = vlocalize_r1("lerp43")
    lerp65 = vlocalize_r1("lerp65")
    ddiff6543 = vlocalize_r1("ddiff6543")

    graph = ComputationGraph()
    graph.add_new_slot("valu^", [val_in, treeval_in], dest=hash_in)
    graph.merge_below(make_hash_graph(batch, round, hash_in, val_out))
    graph.exports[val_out] = graph.inner_defs[val_out]
    graph.add_new_slot("multiply_add", [vconstn(2), idx_in, vconstn(1)], dest=idx_tmp)
    graph.add_new_slot("valu%", [val_out, vconstn(2)], dest=parity)
    graph.add_new_slot("valu+", [idx_tmp, parity], dest=idx_out, exports=True)
    graph.add_new_slot("valu-", [idx_out, vconstn(3)], dest=norm_idx)
    graph.add_new_slot("valu&", [norm_idx, vconstn(1)], dest=bit0)
    graph.add_new_slot("valu>>", [norm_idx, vconstn(1)], dest=norm_idx_down1)
    graph.add_new_slot("valu&", [norm_idx_down1, vconstn(1)], dest=bit1)
    graph.add_new_slot(
        "multiply_add", [bit0, vconst("diff43"), vconst("treeval3")], dest=lerp43
    )
    graph.add_new_slot(
        "multiply_add", [bit0, vconst("diff65"), vconst("treeval5")], dest=lerp65
    )
    graph.add_new_slot("valu-", [lerp65, lerp43], dest=ddiff6543)
    graph.add_new_slot(
        "multiply_add", [bit1, ddiff6543, lerp43], dest=treeval_out, exports=True
    )

    assert len(graph.ext_deps) == 3 and all(
        name in graph.ext_deps for name in (val_in, idx_in, treeval_in)
    )
    assert len(graph.exports) == 3 and all(
        name in graph.exports for name in (val_out, idx_out, treeval_out)
    )
    return graph


def make_round1_graph_flow(
    batch: int,
    round: int,
    val_in: str,
    val_out: str,
    idx_in: str,
    idx_out: str,
    treeval_in: str,
    treeval_out: str,
):
    def vlocalize_r1(name: str) -> str:
        return vlocalize(
            name,
            batch,
            round,
        )

    hash_in = vlocalize_r1("hash_in")
    idx_tmp = vlocalize_r1("idx_tmp")
    parity = vlocalize_r1("parity")
    norm_idx = vlocalize_r1("norm_idx")
    norm_idx_down1 = vlocalize_r1("norm_idx_down1")
    bit0 = vlocalize_r1("bit0")
    bit1 = vlocalize_r1("bit1")
    sel43 = vlocalize_r1("sel43")
    sel65 = vlocalize_r1("sel65")

    graph = ComputationGraph()
    graph.add_new_slot("valu^", [val_in, treeval_in], dest=hash_in)
    graph.merge_below(make_hash_graph(batch, round, hash_in, val_out))
    graph.exports[val_out] = graph.inner_defs[val_out]
    graph.add_new_slot("multiply_add", [vconstn(2), idx_in, vconstn(1)], dest=idx_tmp)
    graph.add_new_slot("valu%", [val_out, vconstn(2)], dest=parity)
    graph.add_new_slot("valu+", [idx_tmp, parity], dest=idx_out, exports=True)
    graph.add_new_slot("valu-", [idx_out, vconstn(3)], dest=norm_idx)
    graph.add_new_slot("valu&", [norm_idx, vconstn(1)], dest=bit0)
    graph.add_new_slot("valu>>", [norm_idx, vconstn(1)], dest=norm_idx_down1)
    graph.add_new_slot("valu&", [norm_idx_down1, vconstn(1)], dest=bit1)
    graph.add_new_slot(
        "vselect", [bit0, vconst("treeval4"), vconst("treeval3")], dest=sel43
    )
    graph.add_new_slot(
        "vselect", [bit0, vconst("treeval6"), vconst("treeval5")], dest=sel65
    )
    graph.add_new_slot("vselect", [bit1, sel65, sel43], dest=treeval_out, exports=True)

    assert len(graph.ext_deps) == 3 and all(
        name in graph.ext_deps for name in (val_in, idx_in, treeval_in)
    )
    assert len(graph.exports) == 3 and all(
        name in graph.exports for name in (val_out, idx_out, treeval_out)
    )
    return graph


def make_round1_graph(
    batch: int,
    round: int,
    val_in: str,
    val_out: str,
    idx_in: str,
    idx_out: str,
    treeval_in: str,
    treeval_out: str,
    use_flow=False,
):
    if use_flow:
        return make_round1_graph_flow(
            batch, round, val_in, val_out, idx_in, idx_out, treeval_in, treeval_out
        )
    return make_round1_graph_alu(
        batch, round, val_in, val_out, idx_in, idx_out, treeval_in, treeval_out
    )


def make_round2_graph_alu(
    batch: int,
    round: int,
    val_in: str,
    val_out: str,
    idx_in: str,
    idx_out: str,
    treeval_in: str,
    treeval_out: str,
):
    def vlocalize_r2(name: str) -> str:
        return vlocalize(
            name,
            batch,
            round,
        )

    hash_in = vlocalize_r2("hash_in")
    idx_tmp = vlocalize_r2("idx_tmp")
    parity = vlocalize_r2("parity")
    norm_idx = vlocalize_r2("norm_idx")
    norm_idx_down1 = vlocalize_r2("norm_idx_down1")
    norm_idx_down2 = vlocalize_r2("norm_idx_down2")
    bit0 = vlocalize_r2("bit0")
    bit1 = vlocalize_r2("bit1")
    bit2 = vlocalize_r2("bit2")
    lerp87 = vlocalize_r2("lerp87")
    lerp109 = vlocalize_r2("lerp109")
    lerp1211 = vlocalize_r2("lerp1211")
    lerp1413 = vlocalize_r2("lerp1413")
    ddiff10987 = vlocalize_r2("ddiff10987")
    ddiff14131211 = vlocalize_r2("ddiff14131211")
    lerp10987 = vlocalize_r2("lerp10987")
    lerp14131211 = vlocalize_r2("lerp14131211")
    dddiff147 = vlocalize_r2("dddiff147")

    graph = ComputationGraph()
    graph.add_new_slot("valu^", [val_in, treeval_in], dest=hash_in)

    graph.merge_below(make_hash_graph(batch, round, hash_in, val_out))
    graph.exports[val_out] = graph.inner_defs[val_out]

    graph.add_new_slot("multiply_add", [vconstn(2), idx_in, vconstn(1)], dest=idx_tmp)
    graph.add_new_slot("valu%", [val_out, vconstn(2)], dest=parity)
    graph.add_new_slot("valu+", [idx_tmp, parity], dest=idx_out, exports=True)
    graph.add_new_slot("valu-", [idx_out, vconstn(7)], dest=norm_idx)
    graph.add_new_slot("valu>>", [norm_idx, vconstn(1)], dest=norm_idx_down1)
    graph.add_new_slot("valu>>", [norm_idx, vconstn(2)], dest=norm_idx_down2)
    graph.add_new_slot("valu&", [norm_idx, vconstn(1)], dest=bit0)
    graph.add_new_slot("valu&", [norm_idx_down1, vconstn(1)], dest=bit1)
    graph.add_new_slot("valu&", [norm_idx_down2, vconstn(1)], dest=bit2)

    graph.add_new_slot(
        "multiply_add", [bit0, vconst("diff87"), vconst("treeval7")], dest=lerp87
    )
    graph.add_new_slot(
        "multiply_add", [bit0, vconst("diff109"), vconst("treeval9")], dest=lerp109
    )
    graph.add_new_slot(
        "multiply_add", [bit0, vconst("diff1211"), vconst("treeval11")], dest=lerp1211
    )
    graph.add_new_slot(
        "multiply_add", [bit0, vconst("diff1413"), vconst("treeval13")], dest=lerp1413
    )

    graph.add_new_slot("valu-", [lerp109, lerp87], dest=ddiff10987)
    graph.add_new_slot("valu-", [lerp1413, lerp1211], dest=ddiff14131211)

    graph.add_new_slot("multiply_add", [bit1, ddiff10987, lerp87], dest=lerp10987)
    graph.add_new_slot(
        "multiply_add", [bit1, ddiff14131211, lerp1211], dest=lerp14131211
    )
    graph.add_new_slot("valu-", [lerp14131211, lerp10987], dest=dddiff147)

    graph.add_new_slot(
        "multiply_add", [bit2, dddiff147, lerp10987], dest=treeval_out, exports=True
    )

    assert len(graph.ext_deps) == 3 and all(
        name in graph.ext_deps for name in (val_in, idx_in, treeval_in)
    )
    assert len(graph.exports) == 3 and all(
        name in graph.exports for name in (val_out, idx_out, treeval_out)
    )
    return graph


def make_round2_graph_flow(
    batch: int,
    round: int,
    val_in: str,
    val_out: str,
    idx_in: str,
    idx_out: str,
    treeval_in: str,
    treeval_out: str,
):
    def vlocalize_r2(name: str) -> str:
        return vlocalize(
            name,
            batch,
            round,
        )

    hash_in = vlocalize_r2("hash_in")
    idx_tmp = vlocalize_r2("idx_tmp")
    parity = vlocalize_r2("parity")
    norm_idx = vlocalize_r2("norm_idx")
    norm_idx_down1 = vlocalize_r2("norm_idx_down1")
    norm_idx_down2 = vlocalize_r2("norm_idx_down2")
    bit0 = vlocalize_r2("bit0")
    bit1 = vlocalize_r2("bit1")
    bit2 = vlocalize_r2("bit2")
    sel87 = vlocalize_r2("sel87")
    sel109 = vlocalize_r2("sel109")
    sel1211 = vlocalize_r2("sel1211")
    sel1413 = vlocalize_r2("sel1413")
    sel10987 = vlocalize_r2("sel10987")
    sel14131211 = vlocalize_r2("sel14131211")

    graph = ComputationGraph()
    graph.add_new_slot("valu^", [val_in, treeval_in], dest=hash_in)

    graph.merge_below(make_hash_graph(batch, round, hash_in, val_out))
    graph.exports[val_out] = graph.inner_defs[val_out]

    graph.add_new_slot("multiply_add", [vconstn(2), idx_in, vconstn(1)], dest=idx_tmp)
    graph.add_new_slot("valu%", [val_out, vconstn(2)], dest=parity)
    graph.add_new_slot("valu+", [idx_tmp, parity], dest=idx_out, exports=True)
    graph.add_new_slot("valu-", [idx_out, vconstn(7)], dest=norm_idx)
    graph.add_new_slot("valu>>", [norm_idx, vconstn(1)], dest=norm_idx_down1)
    graph.add_new_slot("valu>>", [norm_idx, vconstn(2)], dest=norm_idx_down2)
    graph.add_new_slot("valu&", [norm_idx, vconstn(1)], dest=bit0)
    graph.add_new_slot("valu&", [norm_idx_down1, vconstn(1)], dest=bit1)
    graph.add_new_slot("valu&", [norm_idx_down2, vconstn(1)], dest=bit2)

    sel87_slot = make_slot(
        "vselect", [bit0, vconst("treeval8"), vconst("treeval7")], dest=sel87
    )
    sel109_slot = make_slot(
        "vselect", [bit0, vconst("treeval10"), vconst("treeval9")], dest=sel109
    )
    sel1211_slot = make_slot(
        "vselect", [bit0, vconst("treeval12"), vconst("treeval11")], dest=sel1211
    )
    sel1413_slot = make_slot(
        "vselect", [bit0, vconst("treeval14"), vconst("treeval13")], dest=sel1413
    )
    sel10987_slot = make_slot("vselect", [bit1, sel109, sel87], dest=sel10987)
    sel14131211_slot = make_slot("vselect", [bit1, sel1413, sel1211], dest=sel14131211)
    treeval_out_slot = make_slot(
        "vselect", [bit2, sel14131211, sel10987], dest=treeval_out
    )

    graph.add_slot(sel87_slot)
    graph.add_slot(sel109_slot)
    graph.add_slot(sel1211_slot)
    graph.add_slot(sel1413_slot)
    graph.add_slot(sel10987_slot)
    graph.add_slot(sel14131211_slot)
    graph.add_slot(treeval_out_slot, exports=True)

    # for src in [sel87_slot, sel109_slot, sel1211_slot, sel1413_slot]:
    #     for target in [sel10987_slot, sel14131211_slot]:
    #         graph.add_edge(src, target, var="FAKE")

    # graph.add_new_slot(
    #     "vselect", [bit0, vconst("treeval8"), vconst("treeval7")], dest=sel87
    # )
    # graph.add_new_slot(
    #     "vselect", [bit0, vconst("treeval10"), vconst("treeval9")], dest=sel109
    # )
    # graph.add_new_slot(
    #     "vselect", [bit0, vconst("treeval12"), vconst("treeval11")], dest=sel1211
    # )
    # graph.add_new_slot(
    #     "vselect", [bit0, vconst("treeval14"), vconst("treeval13")], dest=sel1413
    # )

    # graph.add_new_slot("vselect", [bit1, sel109, sel87], dest=sel10987)
    # graph.add_new_slot("vselect", [bit1, sel1413, sel1211], dest=sel14131211)
    # graph.add_new_slot(
    #     "vselect", [bit2, sel14131211, sel10987], dest=treeval_out, exports=True
    # )

    assert len(graph.ext_deps) == 3 and all(
        name in graph.ext_deps for name in (val_in, idx_in, treeval_in)
    )
    assert len(graph.exports) == 3 and all(
        name in graph.exports for name in (val_out, idx_out, treeval_out)
    )
    return graph


def make_round2_graph(
    batch: int,
    round: int,
    val_in: str,
    val_out: str,
    idx_in: str,
    idx_out: str,
    treeval_in: str,
    treeval_out: str,
    use_flow=False,
):
    if use_flow:
        return make_round2_graph_flow(
            batch, round, val_in, val_out, idx_in, idx_out, treeval_in, treeval_out
        )
    return make_round2_graph_alu(
        batch, round, val_in, val_out, idx_in, idx_out, treeval_in, treeval_out
    )


def make_round3_graph(
    batch: int,
    round: int,
    val_in: str,
    val_out: str,
    idx_in: str,
    idx_out: str,
    treeval_in: str,
    treeval_out: str,
):
    def vlocalize_r3(name: str) -> str:
        return vlocalize(
            name,
            batch,
            round,
        )

    hash_in = vlocalize_r3("hash_in")
    idx_tmp = vlocalize_r3("idx_tmp")
    parity = vlocalize_r3("parity")
    norm_idx = vlocalize_r3("norm_idx")
    norm_idx_down1 = vlocalize_r3("norm_idx_down1")
    norm_idx_down2 = vlocalize_r3("norm_idx_down2")
    norm_idx_down3 = vlocalize_r3("norm_idx_down3")
    bit0 = vlocalize_r3("bit0")
    bit1 = vlocalize_r3("bit1")
    bit2 = vlocalize_r3("bit2")
    bit3 = vlocalize_r3("bit3")
    sel1615 = vlocalize_r3("sel1615")
    sel1817 = vlocalize_r3("sel1817")
    sel2019 = vlocalize_r3("sel2019")
    sel2221 = vlocalize_r3("sel2221")
    sel2423 = vlocalize_r3("sel2423")
    sel2625 = vlocalize_r3("sel2625")
    sel2827 = vlocalize_r3("sel2827")
    sel3029 = vlocalize_r3("sel3029")
    sel1815 = vlocalize_r3("sel1815")
    sel2219 = vlocalize_r3("sel2219")
    sel2623 = vlocalize_r3("sel2623")
    sel3027 = vlocalize_r3("sel3027")
    sel2215 = vlocalize_r3("sel2215")
    sel3023 = vlocalize_r3("sel3023")

    graph = ComputationGraph()
    graph.add_new_slot("valu^", [val_in, treeval_in], dest=hash_in)

    graph.merge_below(make_hash_graph(batch, round, hash_in, val_out))
    graph.exports[val_out] = graph.inner_defs[val_out]

    graph.add_new_slot("multiply_add", [vconstn(2), idx_in, vconstn(1)], dest=idx_tmp)
    graph.add_new_slot("valu%", [val_out, vconstn(2)], dest=parity)
    graph.add_new_slot("valu+", [idx_tmp, parity], dest=idx_out, exports=True)
    graph.add_new_slot("valu-", [idx_out, vconstn(15)], dest=norm_idx)
    graph.add_new_slot("valu>>", [norm_idx, vconstn(1)], dest=norm_idx_down1)
    graph.add_new_slot("valu>>", [norm_idx, vconstn(2)], dest=norm_idx_down2)
    graph.add_new_slot("valu>>", [norm_idx, vconstn(3)], dest=norm_idx_down3)
    graph.add_new_slot("valu&", [norm_idx, vconstn(1)], dest=bit0)
    graph.add_new_slot("valu&", [norm_idx_down1, vconstn(1)], dest=bit1)
    graph.add_new_slot("valu&", [norm_idx_down2, vconstn(1)], dest=bit2)
    graph.add_new_slot("valu&", [norm_idx_down3, vconstn(1)], dest=bit3)

    graph.add_new_slot(
        "vselect", [bit0, vconst("treeval16"), vconst("treeval15")], dest=sel1615
    )
    graph.add_new_slot(
        "vselect", [bit0, vconst("treeval18"), vconst("treeval17")], dest=sel1817
    )
    graph.add_new_slot(
        "vselect", [bit0, vconst("treeval20"), vconst("treeval19")], dest=sel2019
    )
    graph.add_new_slot(
        "vselect", [bit0, vconst("treeval22"), vconst("treeval21")], dest=sel2221
    )
    graph.add_new_slot(
        "vselect", [bit0, vconst("treeval24"), vconst("treeval23")], dest=sel2423
    )
    graph.add_new_slot(
        "vselect", [bit0, vconst("treeval26"), vconst("treeval25")], dest=sel2625
    )
    graph.add_new_slot(
        "vselect", [bit0, vconst("treeval28"), vconst("treeval27")], dest=sel2827
    )
    graph.add_new_slot(
        "vselect", [bit0, vconst("treeval30"), vconst("treeval29")], dest=sel3029
    )
    graph.add_new_slot("vselect", [bit1, sel1817, sel1615], dest=sel1815)
    graph.add_new_slot("vselect", [bit1, sel2221, sel2019], dest=sel2219)
    graph.add_new_slot("vselect", [bit1, sel2625, sel2423], dest=sel2623)
    graph.add_new_slot("vselect", [bit1, sel3029, sel2827], dest=sel3027)
    graph.add_new_slot("vselect", [bit2, sel2219, sel1815], dest=sel2215)
    graph.add_new_slot("vselect", [bit2, sel3027, sel2623], dest=sel3023)
    graph.add_new_slot(
        "vselect", [bit3, sel3023, sel2215], dest=treeval_out, exports=True
    )

    assert len(graph.ext_deps) == 3 and all(
        name in graph.ext_deps for name in (val_in, idx_in, treeval_in)
    )
    assert len(graph.exports) == 3 and all(
        name in graph.exports for name in (val_out, idx_out, treeval_out)
    )
    return graph


# with this new system, idx_in and treeval_in for wraparound round should automatically be freed naturally? like they have less edges, so detected automatically
def make_wraparound_graph(
    batch: int,
    round: int,
    val_in: str,
    treeval_in: str,
    val_out: str,
):
    def vlocalize_wrap(name: str) -> str:
        return vlocalize(name, batch, round)

    graph = ComputationGraph()
    hash_in = vlocalize_wrap("hash_in")

    graph.add_new_slot("valu^", [val_in, treeval_in], dest=hash_in)
    graph.merge_below(make_hash_graph(batch, "wrap", hash_in, val_out))
    graph.exports[val_out] = graph.inner_defs[val_out]

    assert len(graph.ext_deps) == 2 and all(
        name in graph.ext_deps for name in (val_in, treeval_in)
    )
    assert len(graph.exports) == 1 and val_out in graph.exports
    return graph


def make_last_round_graph(
    batch: int,
    round: int,
    val_in: str,
    treeval_in: str,
    curr_addr_in: str,
):
    def vlocalize_last(name: str) -> str:
        return vlocalize(name, batch, round)

    graph = ComputationGraph()
    hash_in = vlocalize_last("hash_in")
    val_final = vlocalize_last("val_final")

    graph.add_new_slot("valu^", [val_in, treeval_in], dest=hash_in)
    # no export this time
    graph.merge_below(make_hash_graph(batch, round, hash_in, val_final))
    graph.add_new_slot("vstore", [curr_addr_in, val_final])

    assert len(graph.ext_deps) == 3 and all(
        name in graph.ext_deps for name in (val_in, treeval_in, curr_addr_in)
    )
    assert len(graph.exports) == 0
    return graph


def make_mid_round_graph(
    batch: int,
    round: int,
    val_in: str,
    val_out: str,
    idx_in: str,
    idx_out: str,
    treeval_in: str,
    treeval_out: str,
):
    def vlocalize_rmid(name: str) -> str:
        return vlocalize(name, batch, round)

    def partial(vname, lane_numbers: str) -> str:
        return make_partial(vname, lane_numbers)

    hash_in = vlocalize_rmid("hash_in")
    parity = vlocalize_rmid("parity")
    idx_tmp = vlocalize_rmid("idx_tmp")
    val_addrs = vlocalize_rmid("val_addrs")
    # partials
    partials = [partial(treeval_out, i) for i in range(8)]

    graph = ComputationGraph()
    graph.add_new_slot("valu^", [val_in, treeval_in], dest=hash_in)

    graph.merge_below(make_hash_graph(batch, round, hash_in, val_out))
    graph.exports[val_out] = graph.inner_defs[val_out]

    graph.add_new_slot("multiply_add", [vconstn(2), idx_in, vconstn(1)], dest=idx_tmp)
    graph.add_new_slot("valu%", [val_out, vconstn(2)], dest=parity)
    graph.add_new_slot("valu+", [idx_tmp, parity], dest=idx_out, exports=True)
    graph.add_new_slot("valu+", [vconst("forest_values_p"), idx_out], dest=val_addrs)

    for i, partial_var in enumerate(partials):
        graph.add_new_slot("vload_scalar", [val_addrs, i], dest=partial_var)

    graph.add_new_slot(
        "vmerge",
        partials,
        dest=treeval_out,
        exports=True,
    )

    assert len(graph.ext_deps) == 3 and all(
        name in graph.ext_deps for name in (val_in, idx_in, treeval_in)
    )
    assert len(graph.exports) == 3 and all(
        name in graph.exports for name in (val_out, idx_out, treeval_out)
    )
    return graph


# shifts needed to make a number in [0, 32)
def calculate_shifts(n: int) -> list[int]:
    shift = 0
    shifts = []
    while n > 0:
        if n % 2 == 1:
            shifts.append(shift)
        n //= 2
        shift += 1
    return shifts


def make_init_load_graph(
    batch: int, curr_addr_out: str, val_init_out: str
) -> ComputationGraph:
    graph = ComputationGraph()
    assert batch % 8 == 0
    batch_prime = batch // 8
    assert batch_prime < 32

    tmp_offset = [batch_localize(f"tmp_offset{i}", batch) for i in range(5)]
    tmp_offset_sum = [batch_localize(f"tmp_offset_sum{i}", batch) for i in range(4)]
    tmp_offset_final = batch_localize("tmp_offset_final", batch)

    if batch_prime == 0:
        graph.add_new_slot(
            "+", [const("inp_values_p"), constn(0)], dest=curr_addr_out, exports=True
        )
        graph.add_new_slot(
            "vload", [const("inp_values_p")], dest=val_init_out, exports=True
        )
    elif batch_prime == 1:
        graph.add_new_slot(
            "+",
            [const("inp_values_p"), const("s_vlen")],
            dest=curr_addr_out,
            exports=True,
        )
        graph.add_new_slot("vload", [curr_addr_out], dest=val_init_out, exports=True)
    else:
        shifts = calculate_shifts(batch // VLEN)
        if len(shifts) == 1:
            # power of 2 shift
            shift = shifts[0]
            graph.add_new_slot("<<", [constn(1), constn(shift)], dest=tmp_offset[0])
            graph.add_new_slot(
                "*", [const("s_vlen"), tmp_offset[0]], dest=tmp_offset_final
            )
            graph.add_new_slot(
                "+",
                [const("inp_values_p"), tmp_offset_final],
                dest=curr_addr_out,
                exports=True,
            )
            graph.add_new_slot(
                "vload", [curr_addr_out], dest=val_init_out, exports=True
            )
        else:
            # shifts
            for i, shift in enumerate(shifts):
                graph.add_new_slot("<<", [constn(1), constn(shift)], dest=tmp_offset[i])
            # sums -- this part is variable
            if len(shifts) == 2:
                graph.add_new_slot(
                    "+", [tmp_offset[0], tmp_offset[1]], dest=tmp_offset_sum[0]
                )
                graph.add_new_slot(
                    "*", [const("s_vlen"), tmp_offset_sum[0]], dest=tmp_offset_final
                )
            if len(shifts) == 3:
                graph.add_new_slot(
                    "+", [tmp_offset[0], tmp_offset[1]], dest=tmp_offset_sum[0]
                )
                graph.add_new_slot(
                    "+", [tmp_offset_sum[0], tmp_offset[2]], dest=tmp_offset_sum[1]
                )
                graph.add_new_slot(
                    "*", [const("s_vlen"), tmp_offset_sum[1]], dest=tmp_offset_final
                )
            elif len(shifts) == 4:
                graph.add_new_slot(
                    "+", [tmp_offset[0], tmp_offset[1]], dest=tmp_offset_sum[0]
                )
                graph.add_new_slot(
                    "+", [tmp_offset[2], tmp_offset[3]], dest=tmp_offset_sum[1]
                )
                graph.add_new_slot(
                    "+", [tmp_offset_sum[0], tmp_offset_sum[1]], dest=tmp_offset_sum[2]
                )
                graph.add_new_slot(
                    "*", [const("s_vlen"), tmp_offset_sum[2]], dest=tmp_offset_final
                )
            elif len(shifts) == 5:
                graph.add_new_slot(
                    "+", [tmp_offset[0], tmp_offset[1]], dest=tmp_offset_sum[0]
                )
                graph.add_new_slot(
                    "+", [tmp_offset[2], tmp_offset[3]], dest=tmp_offset_sum[1]
                )
                graph.add_new_slot(
                    "+", [tmp_offset_sum[0], tmp_offset_sum[1]], dest=tmp_offset_sum[2]
                )
                graph.add_new_slot(
                    "+", [tmp_offset_sum[2], tmp_offset[4]], dest=tmp_offset_sum[3]
                )
                graph.add_new_slot(
                    "*", [const("s_vlen"), tmp_offset_sum[3]], dest=tmp_offset_final
                )

            # final addr calc and load
            graph.add_new_slot(
                "+",
                [const("inp_values_p"), tmp_offset_final],
                dest=curr_addr_out,
                exports=True,
            )
            graph.add_new_slot(
                "vload", [curr_addr_out], dest=val_init_out, exports=True
            )

    assert len(graph.ext_deps) == 0, f"ext deps {graph.ext_deps} do not match"
    assert len(graph.exports) == 2 and all(
        name in graph.exports for name in (curr_addr_out, val_init_out)
    )
    return graph


def make_batch_graph(
    forest_height: int,
    rounds: int,
    batch: int,
) -> ComputationGraph:
    curr_addr = batch_localize("curr_addr", batch)
    # val, idx, treeval get diff names based on round
    cgraph = ComputationGraph()
    curr_addr = batch_localize("curr_addr", batch)
    val_init = vbatch_localize("val_init", batch)
    cgraph.merge_below(make_init_load_graph(batch, curr_addr, val_init))

    # Only 4 batches can be in round 2 at any given moment? slot for treeval_in, slot for treeval_out for all of the round 2s
    batch1_inout_slots = {}
    batch2_inout_slots = {}
    batch3_inout_slots = {}

    for round in range(rounds):
        # define output variables for this round; will be used as input for next round
        val_out = vlocalize("val", batch, round)
        idx_out = vlocalize("idx", batch, round)
        treeval_out = vlocalize("treeval", batch, round)
        # and the corresponding input vars:
        val_in = vlocalize("val", batch, round - 1)
        idx_in = vlocalize("idx", batch, round - 1)
        treeval_in = vlocalize("treeval", batch, round - 1)

        # round0 graphs
        if round % (forest_height + 1) == 0:
            val_in = val_init if round == 0 else val_in
            cgraph.merge_below(
                make_round0_graph(batch, round, val_in, val_out, idx_out, treeval_out)
            )
        # round1 graphs
        elif round % (forest_height + 1) == 1:
            cgraph.merge_below(
                make_round1_graph(
                    batch,
                    round,
                    val_in,
                    val_out,
                    idx_in,
                    idx_out,
                    treeval_in,
                    treeval_out,
                    use_flow=True,
                )
            )
            batch1_inout_slots[round] = (
                treeval_in,
                treeval_out,
                val_in,
                val_out,
                idx_in,
                idx_out,
            )
        # round2 graphs
        elif round % (forest_height + 1) == 2:
            cgraph.merge_below(
                make_round2_graph(
                    batch,
                    round,
                    val_in,
                    val_out,
                    idx_in,
                    idx_out,
                    treeval_in,
                    treeval_out,
                    use_flow=True,
                )
            )
            batch2_inout_slots[round] = (
                treeval_in,
                treeval_out,
                val_in,
                val_out,
                idx_in,
                idx_out,
            )
        # round3 graphs
        elif round == 3:
            cgraph.merge_below(
                make_round3_graph(
                    batch,
                    round,
                    val_in,
                    val_out,
                    idx_in,
                    idx_out,
                    treeval_in,
                    treeval_out,
                )
            )
            batch3_inout_slots[round] = (
                treeval_in,
                treeval_out,
                val_in,
                val_out,
                idx_in,
                idx_out,
            )
        # last round
        elif round == rounds - 1:
            cgraph.merge_below(
                make_last_round_graph(batch, round, val_in, treeval_in, curr_addr)
            )
        # wraparound graph
        elif (round + 1) % (forest_height + 1) == 0:
            cgraph.merge_below(
                make_wraparound_graph(batch, round, val_in, treeval_in, val_out)
            )
        else:
            cgraph.merge_below(
                make_mid_round_graph(
                    batch,
                    round,
                    val_in,
                    val_out,
                    idx_in,
                    idx_out,
                    treeval_in,
                    treeval_out,
                )
            )

    assert len(cgraph.ext_deps) == 0
    assert len(cgraph.exports) == 0
    # find the corresponding slots
    slots = {}
    for node in cgraph.graph.nodes:
        for round in (2, forest_height + 1 + 2):
            treeval_in, treeval_out, val_in, val_out, idx_in, idx_out = (
                batch2_inout_slots[round]
            )
            if node.dest == treeval_in:
                slots[(round, "tin")] = node
            elif node.dest == treeval_out:
                slots[(round, "tout")] = node
            elif node.dest == val_in:
                slots[(round, "vin")] = node
            elif node.dest == val_out:
                slots[(round, "vout")] = node
            elif node.dest == idx_in:
                slots[(round, "iin")] = node
            elif node.dest == idx_out:
                slots[(round, "iout")] = node
        for round in (1, forest_height + 1 + 1):
            treeval_in, treeval_out, val_in, val_out, idx_in, idx_out = (
                batch1_inout_slots[round]
            )
            if node.dest == treeval_in:
                slots[(round, "tin")] = node
            elif node.dest == treeval_out:
                slots[(round, "tout")] = node
            elif node.dest == val_in:
                slots[(round, "vin")] = node
            elif node.dest == val_out:
                slots[(round, "vout")] = node
            elif node.dest == idx_in:
                slots[(round, "iin")] = node
            elif node.dest == idx_out:
                slots[(round, "iout")] = node
        round = 3
        treeval_in, treeval_out, val_in, val_out, idx_in, idx_out = batch3_inout_slots[
            round
        ]
        if node.dest == treeval_in:
            slots[(round, "tin")] = node
        elif node.dest == treeval_out:
            slots[(round, "tout")] = node
        elif node.dest == val_in:
            slots[(round, "vin")] = node
        elif node.dest == val_out:
            slots[(round, "vout")] = node
        elif node.dest == idx_in:
            slots[(round, "iin")] = node
        elif node.dest == idx_out:
            slots[(round, "iout")] = node

    return cgraph, slots


def make_all_batches_graph(
    forest_height: int,
    batch_size: int,
    rounds: int,
) -> nx.DiGraph:
    comp_graphs = []
    for batch in range(0, batch_size, VLEN):
        comp_graphs.append(make_batch_graph(forest_height, rounds, batch))
    nx_graphs = [g.graph for g, _ in comp_graphs]
    graph = nx.union_all(nx_graphs)
    batch_slots = [slots for _, slots in comp_graphs]
    all_batch_slots = {}
    for i, slots in enumerate(batch_slots):
        batch = i * VLEN
        for round, slot_type in slots:
            all_batch_slots[(batch, round, slot_type)] = slots[(round, slot_type)]

    # Now we need to add cross-batch edges between the graphs. This will allow us to limit the number of concurrent batches in each round.
    def add_fake_dep(
        from_node: SymbolicInstructionSlot,
        to_node: SymbolicInstructionSlot,
    ):
        assert from_node in graph, f"originating node {from_node} must be in graph"
        assert to_node in graph, f"destination node {to_node} must be in graph"
        graph.add_edge(from_node, to_node, data="FAKE")

    # serialize batch0 123, batch8 123, batch16 123, etc
    for i in range(batch_size, VLEN):
        if i - VLEN < 0:
            continue
        # dep. between previous vout and current vin
        add_fake_dep(
            all_batch_slots[(i, 3, "vout")], all_batch_slots[(i - VLEN, 1, "vin")]
        )

    # N_CONCURRENT_BATCHES = 4
    # batches 0*VLEN, 1*VLEN, ..., (N_CONCURRENT_BATCHES-1)*VLEN can run at the same time
    # then batch 4*VLEN depends on batch 0*VLEN, etc
    # for i in range(batch_size // VLEN):
    #     if i - N_CONCURRENT_BATCHES < 0:
    #         continue
    #     add_fake_dep(
    #         all_batch_slots[((i - N_CONCURRENT_BATCHES) * VLEN, 2, "tout")],
    #         all_batch_slots[(i * VLEN, 2, "tin")],
    #     )
    #     add_fake_dep(
    #         all_batch_slots[((i - N_CONCURRENT_BATCHES) * VLEN, 2, "vout")],
    #         all_batch_slots[(i * VLEN, 2, "vin")],
    #     )
    #     add_fake_dep(
    #         all_batch_slots[((i - N_CONCURRENT_BATCHES) * VLEN, 2, "iout")],
    #         all_batch_slots[(i * VLEN, 2, "iin")],
    #     )
    # for i in range(batch_size // VLEN):
    #     if i - N_CONCURRENT_BATCHES < 0:
    #         continue
    #     add_fake_dep(
    #         all_batch_slots[((i - N_CONCURRENT_BATCHES) * VLEN, 1, "tout")],
    #         all_batch_slots[(i * VLEN, 1, "tin")],
    #     )
    #     add_fake_dep(
    #         all_batch_slots[((i - N_CONCURRENT_BATCHES) * VLEN, 1, "vout")],
    #         all_batch_slots[(i * VLEN, 1, "vin")],
    #     )
    #     add_fake_dep(
    #         all_batch_slots[((i - N_CONCURRENT_BATCHES) * VLEN, 1, "iout")],
    #         all_batch_slots[(i * VLEN, 1, "iin")],
    #     )

    return graph


# This "compiles" the program
def make_kernel_graph(
    forest_height: int,
    batch_size: int,
    rounds: int,
    scalar_freelist: set[int],
    vector_freelist: set[int],
    consts: dict[str, int],
) -> FullComputationGraph:
    # make all the batches, and merge_below into const
    return FullComputationGraph(
        make_all_batches_graph(forest_height, batch_size, rounds),
        scalar_freelist,
        vector_freelist,
        consts,
    )


# BELOW THIS LINE IS WHERE NEW CODE IS
# TODO:
# - get rid of vload_scalar, valu_scalar; replace with (vreg @ i) syntax
# ----------------------------------------

CONST_PREFIX = "const "
VECTOR_PREFIX = "v_"

# generally, we can have these things in arguments:
# - strings, for variable names. these can be constant vectors, constant scalars, vectors, or scalars.
#   - note that constant takes precedence over vector -- the const prefix comes first
# - integers, for args to `const` instruction ONLY
# - 2-tuples of (str, int) -- this must be a vector register and an integer index in [0, 8).
Arg = Union[str, int, tuple[str, int]]


# Helpers to effectively rename things.
def const(name: str) -> str:
    return f"{CONST_PREFIX}{name}"


def constn(n: int) -> str:
    return f"{CONST_PREFIX}{n}"


def vector(name: str) -> str:
    return f"{VECTOR_PREFIX}{name}"


def vconst(name: str) -> str:
    return const(vector(name))


def vconstn(n: int) -> str:
    return const(vector(str(n)))


# Detection functions; conversions.
# This uses Arg type.
def is_const(arg: Arg) -> bool:
    assert not isinstance(arg, int)
    if isinstance(arg, tuple):
        vname, offset = arg
        assert isinstance(vname, str) and isinstance(offset, int)
        return is_const(vname)
    return arg.startswith(CONST_PREFIX)


def is_vector(arg: Arg) -> bool:
    assert not isinstance(arg, int)
    if isinstance(arg, tuple):
        return False
    if is_const(arg):
        return is_vector(deconst(arg))
    return arg.startswith(VECTOR_PREFIX)


def deconst(name: str) -> str:
    assert is_const(name)
    return name[len(CONST_PREFIX) :]


def is_scalar(arg: Arg) -> bool:
    assert not isinstance(arg, int)
    if isinstance(arg, tuple):
        vname, offset = arg
        assert isinstance(vname, str) and isinstance(offset, int)
        # vname must be a vector register
        return is_vector(vname)
    return not is_vector(arg)


def batch_localize(name: str, batch: int) -> str:
    # Don't apply this twice.
    tokens = name.split("_")
    assert not (len(tokens) >= 1 and tokens[-1].startswith("batch"))
    return f"{name}_batch{batch}"


# Localization, for batch/round-specific variable names.
def localize(name: str, batch: int, round: int) -> str:
    # Don't apply this twice either.
    tokens = name.split("_")
    assert not (len(tokens) >= 1 and tokens[-1].startswith("round"))
    return f"{batch_localize(name, batch)}_round{round}"


def vlocalize(name: str, batch: int, round: int) -> str:
    return vector(localize(name, batch, round))


def vbatch_localize(name: str, batch: int) -> str:
    return vector(batch_localize(name, batch))


# The only new op we define is vmerge; this becomes a no-op.
def infer_engine(op: str) -> (Engine, str):
    engine = None
    if op == "vmerge":
        engine = "NO-OP"
    elif "load" in op or "const" in op:
        engine = "load"
    elif "store" in op:
        engine = "store"
    elif op in ("+", "-", "*", "//", "cdiv", "^", "&", "|", "<<", ">>", "%", "<", "=="):
        engine = "alu"
    elif op in ("multiply_add", "vbroadcast"):
        engine = "valu"
    elif "valu" in op:
        engine = "valu"
        op = op[len("valu") :]
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

    assert engine is not None, f"invalid engine: {op}"

    return engine, op


ALU_OPS = {
    "+",
    "-",
    "*",
    "//",
    "cdiv",
    "^",
    "&",
    "|",
    "<<",
    ">>",
    "%",
    "<",
    "==",
}


# Must be hashable to be added to DiGraph
@dataclass(frozen=True)
class SymbolicInstructionSlot:
    engine: Engine
    op: str
    args: tuple[Arg, ...]  # the only time we don't get Arg type is for `const`
    dest: Optional[Arg] = None
    # Useful for if we want to impose concurrency limits later.
    batch: int
    round: int
    # Debugging purposes.
    comment: str = ""

    # Check that all variable names follow conventions/are valid for the given engine + op combo.
    # Also that the proper number of arguments are passed, and that the destination is properly defined.
    # Avoids syntax errors.
    def __post_init__(self):
        match self.engine:
            case "alu":
                # should be like alu, op: dest <- arg1, arg2
                assert self.dest is not None
                assert len(self.args) == 2
                assert all(not isinstance(arg, int) for arg in self.args)
                assert all(is_scalar(arg) for arg in self.args)
                assert is_scalar(self.dest)
            case "valu":
                # should be like valu, op: dest <- varg1, varg2
                # except for vbroadcast: 1 scalar arg
                # and multiply_add: vargmul1, vargmul2, vargadd
                assert self.dest is not None
                assert is_vector(self.dest)
                if self.op == "vbroadcast":
                    assert len(self.args) == 1
                    assert is_scalar(self.args[0])
                elif self.op == "multiply_add":
                    assert len(self.args) == 3
                    assert all(is_vector(arg) for arg in self.args)
                else:
                    assert self.op in ALU_OPS
                    assert len(self.args) == 2
                    assert all(is_vector(arg) for arg in self.args)
            case "load":
                # either vload or load; both load from scalar address
                assert self.dest is not None
                assert len(self.args) == 1
                assert is_scalar(self.args[0])
                if self.op == "vload":
                    assert is_vector(self.dest)
                elif self.op == "const":
                    assert len(self.args) == 1
                    assert isinstance(self.args[0], int)
                else:
                    assert self.op == "load"
                    assert is_scalar(self.dest)
            case "store":
                # either vstore or store; both store to scalar address
                assert self.dest is None
                assert len(self.args) == 2
                assert is_scalar(self.args[0])
                if self.op == "vstore":
                    assert is_vector(self.args[1])
                else:
                    assert self.op == "store"
                    assert is_scalar(self.args[1])
            case "flow":
                if self.op == "vselect":
                    assert self.dest is not None
                    assert len(self.args) == 3
                    assert all(is_vector(arg) for arg in self.args)
                else:
                    assert self.op == "pause"
            case "NO-OP":
                # vmerge looks like vmerge vreg, vreg @ 0, vreg @ 1,...
                assert self.op == "vmerge"
                assert self.dest is not None
                assert is_vector(self.dest)
                assert len(self.args) == 8
                assert all(is_scalar(arg) for arg in self.args)
                assert all(isinstance(arg, tuple) for arg in self.args)
                assert all(len(tup) == 2 for tup in self.args)
                # should all have the same name as dest
                names = [arg[0] for arg in self.args]
                assert len(set(names)) == 1
                assert names[0] == self.dest
                # and all offsets 0 to 7
                offsets = [arg[1] for arg in self.args]
                assert set(offsets) == set(range(8))
            case "debug":
                # anything works lmfao
                return

    # Get all variables that this slot depends on.
    # This time, includes constants!
    @property
    def dependencies(self) -> set[str]:
        def is_var(arg: Arg) -> bool:
            if isinstance(arg, int):
                return False
            # everything can be a dependency now, even constants!
            return True

        return {arg for arg in self.args if is_var(arg)}

    @property
    def has_def(self) -> bool:
        return self.dest is not None

    # TODO: find a better heuristic?
    @property
    def is_notable(self) -> bool:
        return self.engine == "load" or self.engine == "store"

    def to_concrete(
        self,
        using: dict[str, int],
        out_reg: Optional[int] = None,
    ) -> (Engine, tuple):
        if self.dest is None:
            assert out_reg is None
        else:
            assert out_reg is not None
        # Resolution/translation happens here.
        assert self.engine != "NO-OP"

        def to_reg(arg: Arg) -> int:
            if isinstance(arg, int):
                return arg
            # This is an offset
            if isinstance(arg, tuple):
                vname, offset = arg
                assert vname in using
                return using[vname] + offset
            # normal var
            assert arg in using
            return using[arg]

        # Translate all input arguments; add dest if needed
        args = [to_reg(arg) for arg in self.args]
        if self.dest is not None:
            args = [out_reg] + args
        return self.engine, (self.op, *args)


def make_slot(
    op: str,
    args: list[Arg],
    dest: Optional[str] = None,
    batch: Optional[int] = None,
    round: Optional[int] = None,
    comment: Optional[str] = "",
) -> SymbolicInstructionSlot:
    engine, op = infer_engine(op)
    return SymbolicInstructionSlot(
        engine=engine,
        op=op,
        args=tuple(args),
        dest=dest,
        batch=batch,
        round=round,
        comment=comment,
    )


# Computation graph of VLIW slots that can be merged with other graphs.
# TODO finish editing
class ComputationGraph:
    def __init__(self):
        self.graph = nx.DiGraph()

        # Temporary staging while we build the graph.
        self.inner_defs: dict[str, SymbolicInstructionSlot] = {}

        # Incoming edges (external dependencies). Which slots depend on a given outside variable?
        self.ext_deps: dict[str, list[SymbolicInstructionSlot]] = defaultdict(list)
        # Outgoing edges (exported variables).
        self.exports: dict[str, SymbolicInstructionSlot] = {}

    def add_edge(
        self,
        from_node: SymbolicInstructionSlot,
        to_node: SymbolicInstructionSlot,
        var: str,  # can be either the dest or FAKE
    ):
        assert from_node in self.graph, f"originating node {from_node} must be in graph"
        assert to_node in self.graph, f"destination node {to_node} must be in graph"
        assert var == "FAKE" or var == from_node.dest, (
            f"var {var} is not the dest or FAKE"
        )
        self.graph.add_edge(from_node, to_node, data=var)

    # Must add slots in dependency order.
    # The last slot added to the graph must be marked. This defines the export.
    def add_slot(self, slot: SymbolicInstructionSlot, exports=False):
        self.graph.add_node(slot)

        # Check that dependencies satisfied, and draw incoming edges
        for arg in slot.dependencies:
            # Inner defs take precedence over upper defs
            if arg in self.inner_defs:
                # Draw edge from inner def's node to this one
                # We don't need to mark with var name, because it's always the same as the originating node's `dest`
                # TODO not true anymore! fake deps possible
                self.add_edge(self.inner_defs[arg], slot, var=arg)
                continue
            # Otherwise, we need this variable from outside our graph. Add to external dependencies; we will draw the edge later.
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

    def add_new_slot(
        self, op: str, arg_names: list[str], dest: Optional[str] = None, exports=False
    ):
        self.add_slot(make_slot(op, arg_names, dest=dest), exports=exports)

    # Add another computation graph "below" this one.
    # `other` must have external dependencies that are either our external dependencies or things that we export.
    # `other`'s exports become part of our inner defs.
    def merge_below(self, other: "ComputationGraph"):
        # Merge, then add edges connecting them.
        self.graph = nx.union(self.graph, other.graph)
        # Link up our outgoing edges to `other`'s incoming edges.
        for var_name, slots in other.ext_deps.items():
            # If this graph has slots depending on something we produce, then we add edges connecting them.
            if var_name in self.inner_defs:
                for slot in slots:
                    self.graph.add_edge(self.inner_defs[var_name], slot, var=var_name)
                continue
            # Otherwise, we add it as another external dependency.
            self.ext_deps[var_name].extend(slots)

        # `other`'s outgoing edges become part of our graph, but as inner defs
        for var_name, slot in other.exports.items():
            # Make sure we're not already exporting this variable name
            assert var_name not in self.exports
            assert var_name not in self.inner_defs
            self.inner_defs[var_name] = slot


def make_const_graph() -> ComputationGraph:
    graph = ComputationGraph()
    # Scalar constants: numerical, VLEN, and hash
    for i in (0, 1, 2, 3, 4, 5, 6, 7):
        graph.add_new_slot("const", [i], dest=constn(i))
    s_vlen = const("s_vlen")
    graph.add_new_slot("const", [VLEN], dest=s_vlen)
    for i, (_, add_imm, _, _, shift_imm) in enumerate(HASH_STAGES):
        if i in (0, 2, 4):
            shift_imm = 1 + 2**shift_imm
        graph.add_new_slot("const", [add_imm], dest=const(f"hash_add{i}"))
        graph.add_new_slot("const", [shift_imm], dest=const(f"hash_mult{i}"))
    # vload the initial variables
    init_vars = vconst("init_vars")
    graph.add_new_slot("vload", [constn(0)], dest=init_vars)

    # Address calculations
    treevals_addr8, treevals_addr16, treevals_addr24 = (
        vector(f"treevals_addr{i}") for i in (8, 16, 24)
    )
    graph.add_new_slot("+", [(init_vars, 3), s_vlen], dest=treevals_addr8)
    graph.add_new_slot("+", [treevals_addr8, s_vlen], dest=treevals_addr16)
    graph.add_new_slot("+", [treevals_addr16, s_vlen], dest=treevals_addr24)

    # vloads
    treevals_starting0, treevals_starting8, treevals_starting16, treevals_starting24 = (
        vconst(f"treevals_starting{i}") for i in (0, 8, 16, 24)
    )
    graph.add_new_slot("vload", [(init_vars, 3)], dest=treevals_starting0)
    graph.add_new_slot("vload", [treevals_addr8], dest=treevals_starting8)
    graph.add_new_slot("vload", [treevals_addr16], dest=treevals_starting16)
    graph.add_new_slot("vload", [treevals_addr24], dest=treevals_starting24)

    # vbroadcast tree values
    for i in range(8):
        graph.add_new_slot(
            "vbroadcast", [(treevals_starting0, i)], dest=vconst(f"treeval{i}")
        )
    for i in range(8):
        graph.add_new_slot(
            "vbroadcast", [treevals_starting8, i], dest=vconst(f"treeval{8 + i}")
        )
    for i in range(8):
        graph.add_new_slot(
            "vbroadcast", [treevals_starting16, i], dest=vconst(f"treeval{16 + i}")
        )
    for i in range(8):
        graph.add_new_slot(
            "vbroadcast", [treevals_starting24, i], dest=vconst(f"treeval{24 + i}")
        )
    # vbroadcast hash values
    for i in range(len(HASH_STAGES)):
        graph.add_new_slot(
            "vbroadcast", [const(f"hash_add{i}")], dest=vconst(f"hash_add{i}")
        )
        graph.add_new_slot(
            "vbroadcast", [const(f"hash_mult{i}")], dest=vconst(f"hash_mult{i}")
        )
    # vbroadcast forest_values_p pointer
    graph.add_new_slot("vbroadcast", [(init_vars, 3)], dest=vconst("forest_values_p"))
    # vbroadcast i values
    for i in (1, 2, 3, 7, 15):
        graph.add_new_slot("vbroadcast", [constn(i)], dest=vconstn(i))


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    # Make scratch registers. Modifies scratch allocator but does not issue instructions.
    def make_freelists(self, n_scalar_reg: int):
        assert self.scratch_ptr == 0
        # Just do this statically.
        scalar_freelist = set(self.alloc_scratch() for _ in range(n_scalar_reg))
        N_VECTOR_REG = (SCRATCH_SIZE - n_scalar_reg) // VLEN
        vector_freelist = set(
            self.alloc_scratch(length=VLEN) for _ in range(N_VECTOR_REG)
        )
        assert all(reg < 1536 for reg in scalar_freelist), (
            "scalar register is out of bounds"
        )
        assert all(reg < 1536 for reg in vector_freelist), (
            "vector register is out of bounds"
        )
        print(
            f"{len(scalar_freelist)} scalar registers, {len(vector_freelist)} vector registers"
        )
        return scalar_freelist, vector_freelist

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        N_SCALAR_REG = 64
        scalar_freelist, vector_freelist = self.make_freelists(N_SCALAR_REG)

        # required to match with reference
        self.add("flow", ("pause",))
        # make_kernel_graph should contain the const code within itself.
        self.instrs.extend(
            make_kernel_graph(
                forest_height,
                batch_size,
                rounds,
                scalar_freelist,
                vector_freelist,
            ).generate_code()
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
        do_kernel_test(10, 16, 256)


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
