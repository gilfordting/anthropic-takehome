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
from typing import Literal, Optional
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

# Variable naming stuff.
# Const has precedence over vector. Vector constants are like const vX.
CONST_PREFIX = "const "
VECTOR_PREFIX = "v_"


def const(name: str) -> str:
    return f"{CONST_PREFIX}{name}"


def constn(n: int) -> str:
    return f"{CONST_PREFIX}{n}"


def is_const(name: str) -> bool:
    return name.startswith(CONST_PREFIX)


def vector(name: str) -> str:
    return f"{VECTOR_PREFIX}{name}"


def is_vector(name: str) -> bool:
    if is_const(name):
        return is_vector(name[len(CONST_PREFIX) :])
    return name.startswith(VECTOR_PREFIX)


# for partial vector vars, where 1 lane of a vector register is used
# partial variables cannot be constants or vectors. they are used in two places:
# - output of vload_scalar custom instruction
# - input to vmerge custom instruction
# partial vars must have a corresponding vector var `v` if they're used as args; will be created only once if used as the dest.
# will be in the format partial_<v's name>_<lane number>
PARTIAL_PREFIX = "partial_"


def make_partial(vname: str, lane_number: int) -> str:
    return f"{PARTIAL_PREFIX}{vname}_{lane_number}"


def get_partial_vname(name: str) -> str:
    assert is_partial(name)
    # get rid of prefix and lane number
    return "_".join(name.split("_")[1:-1])


def is_partial(name: str) -> bool:
    return name.startswith(PARTIAL_PREFIX)


def is_scalar(name: str) -> bool:
    if is_const(name):
        return is_scalar(name[len(CONST_PREFIX) :])
    return not is_vector(name) and not is_partial(name)


def vconst(name: str) -> str:
    return const(vector(name))


def vconstn(n: int) -> str:
    return vconst(str(n))


# localize a name to a batch and round
def localize(name: str, batch: int, round: int) -> str:
    return f"{name}_batch{batch}_round{round}"


def vlocalize(name: str, batch: int, round: int) -> str:
    return vector(localize(name, batch, round))


def batch_localize(name: str, batch: int) -> str:
    return f"{name}_batch{batch}"


def vbatch_localize(name: str, batch: int) -> str:
    return vector(batch_localize(name, batch))


# Define a new op, "vload_scalar": we will create our own custom backend for it.
# Same for vmerge.
def infer_engine(op: str) -> (Engine, str):
    engine = None
    if op == "vmerge":
        engine = "NO-OP"
    elif op == "vload_scalar":
        engine = "load"
    elif "load" in op or "const" in op:
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

    assert engine is not None, f"invalid engine: {op}"

    return engine, op


# Must be hashable to be added to DiGraph
@dataclass(frozen=True)
class SymbolicInstructionSlot:
    engine: Engine
    op: str
    # dependencies are all here; args can also be constants
    arg_names: tuple[str, ...]
    # produced variable name
    dest: Optional[str] = None
    # batch: int
    # round: int
    # comment: str = None

    # Check that all variable names follow conventions/are valid for the given engine + op combo.
    # Also that the proper number of arguments are passed, and that the destination is properly defined.
    def __post_init__(self):
        match self.engine:
            case "alu":
                assert all(is_scalar(arg) for arg in self.arg_names)
                assert len(self.arg_names) == 2
                assert self.dest is not None and is_scalar(self.dest)
            case "valu":
                assert self.dest is not None and is_vector(self.dest), (
                    f"dest {self.dest} is not a vector"
                )
                if self.op == "vbroadcast":
                    assert len(self.arg_names) == 1
                    assert is_scalar(self.arg_names[0])
                    return
                assert all(is_vector(arg) for arg in self.arg_names)
                n_args = 3 if self.op == "multiply_add" else 2
                assert len(self.arg_names) == n_args
            case "load":
                assert self.dest is not None
                if self.op == "vload_scalar":
                    assert is_partial(self.dest)
                    assert len(self.arg_names) == 2
                    assert is_vector(self.arg_names[0])
                    assert isinstance(self.arg_names[1], int)
                elif self.op == "vload":
                    assert is_vector(self.dest), f"dest {self.dest} is not a vector"
                # else:
                #     assert is_scalar(self.dest)
                # if self.op == "load_offset":
                #     assert len(self.arg_names) == 2
                # else:
                #     assert len(self.arg_names) == 1
                # TODO: mix of concrete value and actual variable names?
                # TODO: note that our custom ops must be validated here
            case "store":
                assert self.dest is None
                assert len(self.arg_names) == 2
                assert is_scalar(self.arg_names[0])
                if self.op == "vstore":
                    assert is_vector(self.arg_names[1])
                else:
                    assert is_scalar(self.arg_names[1])
            case "flow":
                assert self.op == "pause"
            case "NO-OP":
                assert len(self.arg_names) == 8
                assert all(is_partial(arg) for arg in self.arg_names)
                names = [get_partial_vname(arg) for arg in self.arg_names]
                names_set = set(names)
                assert len(names_set) == 1  # should all have the same vname
                assert names_set.pop() == self.dest
            case "debug":
                # anything works
                return

    @property
    def dependencies(self) -> set[str]:
        return {
            arg
            for arg in self.arg_names
            if not isinstance(arg, int) and not is_const(arg)
        }

    @property
    def has_def(self) -> bool:
        return self.dest is not None


def make_slot(
    op: str, arg_names: list[str], dest: Optional[str] = None
) -> SymbolicInstructionSlot:
    engine, op = infer_engine(op)
    return SymbolicInstructionSlot(
        engine=engine,
        op=op,
        arg_names=tuple(arg_names),
        dest=dest,
    )


# also TODO: offload valu to alu slots. Keep in mind this is not possible for vbroadcast or multiply_add


# TODO: custom vload_scalar instruction that we need to implement
# Will be load vload_scalar dst, addrs, offset
# - dst is a partial variable, a 2-slice of a vector register.
# - addrs is a vector variable!
# - offset is a concrete numerical value, one of 0, 2, 4, 6.
# We'll also have a vmerge instruction, which will take in partial vars and resolve them
# Semantically, vdst is a vector pair
# Also need a vmerge instruction -- this is a no-op.
# So in scheduling, we'll first need to remove all vmerges with resolved dependencies. Then we look at the leaves.
# The execution engine will need to handle partial variables, and make sure the physical mapping is done correctly.
# But for the purposes of the computation graph/dependency tracking, these instructions allow us to schedule things properly.


# Computation graph of VLIW slots that can be merged with other graphs.
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
        self, from_node: SymbolicInstructionSlot, to_node: SymbolicInstructionSlot
    ):
        assert from_node in self.graph, f"originating node {from_node} must be in graph"
        assert to_node in self.graph, f"destination node {to_node} must be in graph"
        self.graph.add_edge(from_node, to_node)

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
                    self.graph.add_edge(self.inner_defs[var_name], slot)
                continue
            # Otherwise, we add it as another external dependency.
            self.ext_deps[var_name].extend(slots)

        # `other`'s outgoing edges become part of our graph, but as inner defs
        for var_name, slot in other.exports.items():
            # Make sure we're not already exporting this variable name
            assert var_name not in self.exports
            assert var_name not in self.inner_defs
            self.inner_defs[var_name] = slot


class FullComputationGraph:
    def __init__(self, graphs: list[ComputationGraph]):
        self.graph = nx.DiGraph()
        for g in graphs:
            self.graph = nx.union(self.graph, g.graph)


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


def make_round1_graph(
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


def make_round2_graph(
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


# with this new system, idx_in and treeval_in for wraparound round should automatically be freed naturally? like they have less edges, so detected automatically
# TODO check this; should mean idx_in is freed
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
        return vlocalize(name, batch, 2)

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

    for i, partial in enumerate(partials):
        graph.add_new_slot("vload_scalar", [val_addrs, i], dest=partial)

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
        shifts = calculate_shifts(batch)
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
    graph = ComputationGraph()
    curr_addr = batch_localize("curr_addr", batch)
    val_init = vbatch_localize("val_init", batch)
    graph.merge_below(make_init_load_graph(batch, curr_addr, val_init))

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
            graph.merge_below(
                make_round0_graph(batch, round, val_in, val_out, idx_out, treeval_out)
            )
        # round1 graphs
        elif round % (forest_height + 1) == 1:
            graph.merge_below(
                make_round1_graph(
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
        # round2 graphs
        elif round % (forest_height + 1) == 2:
            graph.merge_below(
                make_round2_graph(
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
        # last round
        elif round == rounds - 1:
            graph.merge_below(
                make_last_round_graph(batch, round, val_in, treeval_in, curr_addr)
            )
        # wraparound graph
        elif (round + 1) % (forest_height + 1) == 0:
            graph.merge_below(
                make_wraparound_graph(batch, round, val_in, treeval_in, val_out)
            )
        else:
            graph.merge_below(
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

    return graph


# This "compiles" the program
def make_kernel_graph(
    forest_height: int, batch_size: int, rounds: int
) -> FullComputationGraph:
    return FullComputationGraph(
        [
            make_batch_graph(forest_height, rounds, batch)
            for batch in range(0, batch_size, VLEN)
        ]
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

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        # Simple slot packing that just uses one slot per instruction bundle
        instrs = []
        for engine, slot in slots:
            instrs.append({engine: [slot]})
        return instrs

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

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Like reference_kernel2 but building actual instructions.
        Scalar implementation using only scalar ALU and load/store.
        """
        make_kernel_graph(forest_height, batch_size, rounds)
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
