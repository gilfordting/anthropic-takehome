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


# shifts needed to make a number in [0, 32)
def calculate_shifts(n: int) -> list[int]:
    assert 0 <= n < 32
    shift = 0
    shifts = []
    while n > 0:
        if n % 2 == 1:
            shifts.append(shift)
        n //= 2
        shift += 1
    return shifts


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

Var = Union[str, tuple[str, int]]


# Must be hashable to be added to DiGraph
@dataclass(frozen=True)
class SymbolicInstructionSlot:
    engine: Engine
    op: str
    args: tuple[Arg, ...]  # the only time we don't get Arg type is for `const`
    # Useful for if we want to impose concurrency limits later.
    batch: int
    round: int
    # Debugging purposes.
    comment: str = ""
    dest: Optional[Arg] = None

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
                if self.op == "vload":
                    assert is_scalar(self.args[0])
                    assert is_vector(self.dest)
                elif self.op == "const":
                    assert isinstance(self.args[0], int)
                    assert is_const(self.dest)
                else:
                    assert self.op == "load"
                    assert is_scalar(self.args[0])
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
                elif self.op == "add_imm":
                    assert self.dest is not None
                    assert is_scalar(self.dest)
                    assert len(self.args) == 2
                    assert is_scalar(self.args[0])
                    assert isinstance(self.args[1], int)
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
    # Also includes offset vregs.
    @property
    def dependencies(self) -> set[Var]:
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


class ComputationGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        # Temporary staging while we build the graph.
        self.inner_defs: dict[Var, SymbolicInstructionSlot] = {}
        # If we draw a box around this graph, the ext_deps allow us to make connections between boxes via incoming edges, and exports are outgoing edges.
        # The slots in this graph that depend on externally defined variables.
        # Maps var name, slots that depend on that variable.
        self.ext_deps: dict[Var, list[SymbolicInstructionSlot]] = defaultdict(list)
        # Outgoing edges (exported variables).
        self.exports: dict[Var, SymbolicInstructionSlot] = {}

    # The associated var name must be the from_node's dest.
    # This edge means that to_node has a dependency on from_node's output.
    # TODO fake deps
    def add_edge(
        self,
        from_node: SymbolicInstructionSlot,
        to_node: SymbolicInstructionSlot,
        var: Var,
    ):
        assert from_node in self.graph, f"originating node {from_node} must be in graph"
        assert to_node in self.graph, f"destination node {to_node} must be in graph"
        assert var != "FAKE"  # TODO add support
        # assert var == "FAKE" or var == from_node.dest, (
        #     f"var {var} is not the dest or FAKE"
        # )
        self.graph.add_edge(from_node, to_node, var=var)

    # Must add slots in dependency order.
    # exports means this variable should live beyond the internal computations.
    def add_slot(self, slot: SymbolicInstructionSlot, exports=False):
        self.graph.add_node(slot)
        # Check that dependencies satisfied, and draw incoming edges
        for var in slot.dependencies:
            if isinstance(var, tuple):
                # For vmerge only, we use the tuple. Otherwise, we use the name.
                if slot.op != "vmerge":
                    var_name, offset = var
                    var = var_name
            # Inner defs take precedence over upper defs
            if var in self.inner_defs:
                # Draw edge from inner def's node to this one
                # Mark the edge with the name of the relevant arg
                self.add_edge(self.inner_defs[var], slot, var=var)
                continue
            # Otherwise, we need this variable from outside our graph. Add to external dependencies; we will draw the edge later, because it requires a source not visible to us.
            self.ext_deps[var].append(slot)

        # if this slot defines a new variable, add to inner_defs
        # inner variables must be unique!
        # this naturally comes out of vector offsets being paired
        if slot.has_def:
            # If we write to a offset vector, we add it to inner defs and it must be joined with a vmerge
            assert slot.dest not in self.inner_defs, f"{slot.dest = }"
            self.inner_defs[slot.dest] = slot

        if exports:
            self.exports[slot.dest] = slot

    def add_new_slot(
        self,
        op: str,
        args: list[str],
        dest: Optional[str] = None,
        batch: Optional[int] = None,
        round: Optional[int] = None,
        comment: Optional[str] = "",
        exports=False,
    ):
        self.add_slot(
            make_slot(op, args, dest=dest, batch=batch, round=round, comment=comment),
            exports=exports,
        )

    # Add another computation graph "below" this one.
    # `other` must have external dependencies that are either our external dependencies or things that we export.
    # `other`'s exports become part of our inner defs.
    def merge_below(self, other: "ComputationGraph"):
        # Merge, then add edges connecting them.
        self.graph = nx.union(self.graph, other.graph)
        # ext_deps is the lower graph's incoming edges that we need to draw
        for var, slots in other.ext_deps.items():
            # If this graph has slots depending on something we produce, then we add edges connecting them.
            if var in self.inner_defs:
                for slot in slots:
                    self.graph.add_edge(self.inner_defs[var], slot, var=var)
                continue
            # Otherwise, we add it as another external dependency.
            self.ext_deps[var].extend(slots)
        # `other`'s outgoing edges become part of our graph, but as inner defs
        for var, slot in other.exports.items():
            # Make sure we're not already exporting this variable name
            assert var not in self.exports
            assert var not in self.inner_defs
            self.inner_defs[var] = slot

    @staticmethod
    def make_const_graph() -> "ComputationGraph":
        cgraph = ComputationGraph()
        for i in (0, 1, 2, 3, 4, 5, 8):
            # Only export the ones that aren't 0. 0 is used for loading init_vars.
            cgraph.add_new_slot("const", [i], dest=constn(i))
        for i, (_, add_imm, _, _, shift_imm) in enumerate(HASH_STAGES):
            cgraph.add_new_slot(
                "const",
                [add_imm],
                dest=const(f"hash_add{i}"),
            )
            cgraph.add_new_slot(
                "const",
                [1 + 2**shift_imm if i in (0, 2, 4) else shift_imm],
                dest=const(f"hash_mult{i}"),
            )
        # load init vars
        init_vars = vconst("init_vars")
        cgraph.add_new_slot(
            "vload",
            [constn(0)],
            dest=init_vars,
        )
        forest_values_p = (init_vars, 4)
        treevals_addr8 = const("treevals_addr8")
        cgraph.add_new_slot("add_imm", [forest_values_p, 8], dest=treevals_addr8)
        treevals_starting0, treevals_starting8 = (
            vconst(f"treevals_starting{i}") for i in (0, 8)
        )
        cgraph.add_new_slot(
            "vload",
            [forest_values_p],
            dest=treevals_starting0,
        )
        cgraph.add_new_slot(
            "vload",
            [treevals_addr8],
            dest=treevals_starting8,
        )
        for i in range(len(HASH_STAGES)):
            hash_addi = f"hash_add{i}"
            cgraph.add_new_slot(
                "vbroadcast",
                [const(hash_addi)],
                dest=vconst(hash_addi),
            )
            hash_multi = f"hash_mult{i}"
            cgraph.add_new_slot(
                "vbroadcast",
                [const(hash_multi)],
                dest=vconst(hash_multi),
            )
        cgraph.add_new_slot(
            "vbroadcast",
            [forest_values_p],
            dest=vconst("forest_values_p"),
        )
        for i in (1, 2, 3, 5):
            cgraph.add_new_slot(
                "vbroadcast",
                [constn(i)],
                dest=vconstn(i),
            )

        assert len(cgraph.ext_deps) == 0
        numerical = (1, 2, 3, 5)
        assert all(constn(i) in cgraph.inner_defs for i in numerical)
        assert all(vconstn(i) in cgraph.inner_defs for i in numerical)
        assert all(
            const(f"hash_add{i}") in cgraph.inner_defs for i in range(len(HASH_STAGES))
        )
        assert all(
            const(f"hash_mult{i}") in cgraph.inner_defs for i in range(len(HASH_STAGES))
        )
        assert all(
            vconst(f"hash_add{i}") in cgraph.inner_defs for i in range(len(HASH_STAGES))
        )
        assert all(
            vconst(f"hash_mult{i}") in cgraph.inner_defs
            for i in range(len(HASH_STAGES))
        )
        assert init_vars in cgraph.inner_defs
        assert treevals_starting0 in cgraph.inner_defs
        assert treevals_starting8 in cgraph.inner_defs
        assert vconst("forest_values_p") in cgraph.inner_defs
        assert len(cgraph.inner_defs) >= 2 * len(numerical) + 4 * len(HASH_STAGES) + 4
        return cgraph

    @staticmethod
    def make_hash_graph(
        batch: int,
        round: int,
        in_name: str,
        out_name: str,
    ) -> "ComputationGraph":
        def vlocalize(name: str, hash_round: int) -> str:
            return vector(f"{localize(name, batch, round)}_hashround{hash_round}")

        cgraph = ComputationGraph()

        def add_new_slot(op: str, args: list[str], dest: str, exports: bool = False):
            cgraph.add_new_slot(
                op, args, dest=dest, exports=exports, batch=batch, round=round
            )

        curr_in_name = in_name
        for i, (op1, _, op2, op3, _) in enumerate(HASH_STAGES):
            # Convert to valu operations
            op1, op3 = (f"valu{op}" for op in (op1, op3))
            # Make this round's output variable
            curr_out_name = vlocalize("out", i)

            if i in (0, 2, 4):
                args = [curr_in_name, vconst(f"hash_mult{i}"), vconst(f"hash_add{i}")]
                add_new_slot(
                    "multiply_add",
                    args,
                    dest=curr_out_name,
                )
                curr_in_name = curr_out_name
                continue
            last_stage = i == len(HASH_STAGES) - 1
            tmp1 = vlocalize("tmp1", i)
            tmp2 = vlocalize("tmp2", i)
            add_new_slot(op1, [curr_in_name, vconst(f"hash_add{i}")], dest=tmp1)
            add_new_slot(op3, [curr_in_name, vconst(f"hash_mult{i}")], dest=tmp2)
            target_name = out_name if last_stage else curr_out_name
            for i in range(8):
                add_new_slot(op2, [(tmp1, i), (tmp2, i)], dest=(target_name, i))
            add_new_slot(
                "vmerge",
                [(target_name, i) for i in range(8)],
                dest=target_name,
                exports=last_stage,
            )
            curr_in_name = curr_out_name

        rest_len = ComputationGraph.check_hash_deps(cgraph.ext_deps)
        assert in_name in cgraph.ext_deps
        assert rest_len == 1

        assert len(cgraph.exports) == 1 and out_name in cgraph.exports
        return cgraph

    @staticmethod
    def check_hash_deps(ext_deps: dict[str, SymbolicInstructionSlot]) -> int:
        hash_adds = [vconst(f"hash_add{i}") for i in range(len(HASH_STAGES))]
        hash_mults = [vconst(f"hash_mult{i}") for i in range(len(HASH_STAGES))]
        assert all(var_name in ext_deps for var_name in hash_adds)
        assert all(var_name in ext_deps for var_name in hash_mults)
        diff = len(ext_deps) - (len(hash_adds) + len(hash_mults))
        assert diff >= 0
        return diff

    @staticmethod
    def make_top_layer_graph(
        batch: int,
        round: int,
        forest_height: int,
        val_in: str,
        val_out: str,
        treeval_out: str,
        parity_out: str,
    ) -> "ComputationGraph":
        assert round % (forest_height + 1) == 0

        def vlocalize_top(name: str) -> str:
            return vlocalize(name, batch, round)

        cgraph = ComputationGraph()

        def add_new_slot(op: str, args: list[str], dest: str, exports: bool = False):
            cgraph.add_new_slot(
                op, args, dest=dest, exports=exports, batch=batch, round=round
            )

        treeval0 = vconst(f"treeval0_round{round}")
        treeval1 = vconst(f"treeval1_round{round}")
        treeval2 = vconst(f"treeval2_round{round}")

        hash_in = vlocalize_top("hash_in")
        add_new_slot("valu^", [val_in, treeval0], dest=hash_in)
        cgraph.merge_below(
            ComputationGraph.make_hash_graph(batch, round, hash_in, val_out)
        )
        cgraph.exports[val_out] = cgraph.inner_defs[val_out]
        add_new_slot("valu%", [val_out, vconstn(2)], dest=parity_out, exports=True)
        add_new_slot(
            "vselect",
            [
                parity_out,
                treeval2,
                treeval1,
            ],
            dest=treeval_out,
            exports=True,
        )

        assert val_in in cgraph.ext_deps
        assert all(
            treeval in cgraph.ext_deps for treeval in (treeval0, treeval1, treeval2)
        )
        assert vconstn(2) in cgraph.ext_deps
        rest_len = ComputationGraph.check_hash_deps(cgraph.ext_deps)
        assert rest_len == 5

        assert len(cgraph.exports) == 3 and all(
            name in cgraph.exports for name in (val_out, treeval_out, parity_out)
        )
        return cgraph

    @staticmethod
    def make_second_layer_graph(
        batch: int,
        round: int,
        forest_height: int,
        val_in: str,
        parity_in: str,
        treeval_in: str,
        val_out: str,
        idx_out: str,
        treeval_out: str,
    ) -> "ComputationGraph":
        assert round % (forest_height + 1) == 1

        def vlocalize_second(name: str) -> str:
            return vlocalize(name, batch, round)

        cgraph = ComputationGraph()

        def add_new_slot(op: str, args: list[str], dest: str, exports: bool = False):
            cgraph.add_new_slot(
                op, args, dest=dest, exports=exports, batch=batch, round=round
            )

        hash_in = vlocalize_second("hash_in")
        parity_curr = vlocalize_second("parity_curr")
        idx_base = vlocalize_second("idx_base")
        treeval3, treeval4, treeval5, treeval6 = (
            vconst(f"treeval{i}_round{round}") for i in (3, 4, 5, 6)
        )
        sel_43 = vlocalize_second("sel_43")
        sel_65 = vlocalize_second("sel_65")

        add_new_slot("valu^", [val_in, treeval_in], dest=hash_in)
        cgraph.merge_below(
            ComputationGraph.make_hash_graph(batch, round, hash_in, val_out)
        )
        cgraph.exports[val_out] = cgraph.inner_defs[val_out]
        add_new_slot("valu%", [val_out, vconstn(2)], dest=parity_curr)
        add_new_slot("multiply_add", [parity_in, vconstn(2), vconstn(3)], dest=idx_base)
        add_new_slot("valu+", [idx_base, parity_curr], dest=idx_out, exports=True)
        add_new_slot("vselect", [parity_curr, treeval4, treeval3], dest=sel_43)
        add_new_slot("vselect", [parity_curr, treeval6, treeval5], dest=sel_65)
        add_new_slot(
            "vselect", [parity_in, sel_65, sel_43], dest=treeval_out, exports=True
        )

        assert all(name in cgraph.ext_deps for name in (val_in, parity_in, treeval_in))
        assert all(
            name in cgraph.ext_deps for name in (treeval3, treeval4, treeval5, treeval6)
        )
        assert all(vconstn(i) in cgraph.ext_deps for i in (2, 3))
        rest_len = ComputationGraph.check_hash_deps(cgraph.ext_deps)
        assert rest_len == 3 + 4 + 2

        assert all(name in cgraph.exports for name in (val_out, idx_out, treeval_out))
        assert len(cgraph.exports) == 3
        return cgraph

    @staticmethod
    def make_mid_layer_graph(
        batch: int,
        round: int,
        rounds: int,
        forest_height: int,
        val_in: str,
        idx_in: str,
        treeval_in: str,
        val_out: str,
        idx_out: str,
        treeval_out: str,
    ) -> "ComputationGraph":
        assert 2 <= round % (forest_height + 1) <= forest_height - 1
        assert round != rounds - 1

        def vlocalize_mid(name: str) -> str:
            return vlocalize(name, batch, round)

        cgraph = ComputationGraph()

        def add_new_slot(op: str, args: list[str], dest: str, exports: bool = False):
            cgraph.add_new_slot(
                op, args, dest=dest, exports=exports, batch=batch, round=round
            )

        hash_in = vlocalize_mid("hash_in")
        idx_base = vlocalize_mid("idx_base")
        parity = vlocalize_mid("parity")
        val_addrs = vlocalize_mid("val_addrs")
        add_new_slot("valu^", [val_in, treeval_in], dest=hash_in)
        cgraph.merge_below(
            ComputationGraph.make_hash_graph(batch, round, hash_in, val_out)
        )
        cgraph.exports[val_out] = cgraph.inner_defs[val_out]
        add_new_slot("multiply_add", [vconstn(2), idx_in, vconstn(1)], dest=idx_base)
        add_new_slot("valu%", [val_out, vconstn(2)], dest=parity)
        add_new_slot("valu+", [idx_base, parity], dest=idx_out, exports=True)
        add_new_slot("valu+", [vconst("forest_values_p"), idx_out], dest=val_addrs)
        # 8 scalar loads
        for i in range(8):
            add_new_slot("load", [(val_addrs, i)], dest=(treeval_out, i))
        add_new_slot(
            "vmerge",
            [(treeval_out, i) for i in range(8)],
            dest=treeval_out,
            exports=True,
        )

        assert all(name in cgraph.ext_deps for name in (val_in, idx_in, treeval_in))
        assert all(vconstn(i) in cgraph.ext_deps for i in (1, 2))
        assert vconst("forest_values_p") in cgraph.ext_deps
        rest_len = ComputationGraph.check_hash_deps(cgraph.ext_deps)
        assert rest_len == 3 + 2 + 1, f"{cgraph.ext_deps.keys()}"

        assert all(name in cgraph.exports for name in (val_out, idx_out, treeval_out))
        assert len(cgraph.exports) == 3
        return cgraph

    @staticmethod
    def make_wraparound_layer_graph(
        batch: int,
        round: int,
        forest_height: int,
        val_in: str,
        treeval_in: str,
        val_out: str,
    ) -> "ComputationGraph":
        assert (round + 1) % (forest_height + 1) == 0

        def vlocalize_wraparound(name: str) -> str:
            return vlocalize(name, batch, round)

        cgraph = ComputationGraph()

        def add_new_slot(op: str, args: list[str], dest: str, exports: bool = False):
            cgraph.add_new_slot(
                op, args, dest=dest, exports=exports, batch=batch, round=round
            )

        hash_in = vlocalize_wraparound("hash_in")
        add_new_slot("valu^", [val_in, treeval_in], dest=hash_in)
        cgraph.merge_below(
            ComputationGraph.make_hash_graph(batch, round, hash_in, val_out)
        )
        cgraph.exports[val_out] = cgraph.inner_defs[val_out]

        assert all(name in cgraph.ext_deps for name in (val_in, treeval_in))
        rest_len = ComputationGraph.check_hash_deps(cgraph.ext_deps)
        assert rest_len == 2

        assert val_out in cgraph.exports
        assert len(cgraph.exports) == 1
        return cgraph

    @staticmethod
    def make_last_layer_graph(
        batch: int,
        round: int,
        rounds: int,
        val_in: str,
        treeval_in: str,
    ) -> "ComputationGraph":
        assert round == rounds - 1

        def localize_last(name: str) -> str:
            return localize(name, batch, round)

        def vlocalize_last(name: str) -> str:
            return vlocalize(name, batch, round)

        cgraph = ComputationGraph()

        def add_new_slot(
            op: str, args: list[str], dest: str = None, exports: bool = False
        ):
            cgraph.add_new_slot(
                op, args, dest=dest, exports=exports, batch=batch, round=round
            )

        hash_in = vlocalize_last("hash_in")
        val_final = vlocalize_last("val_final")
        addr_end = localize_last("addr_end")
        add_new_slot("valu^", [val_in, treeval_in], dest=hash_in)
        cgraph.merge_below(
            ComputationGraph.make_hash_graph(batch, round, hash_in, val_final)
        )
        add_new_slot("add_imm", [(vconst("init_vars"), 6), batch], dest=addr_end)
        add_new_slot("vstore", [addr_end, val_final])

        assert all(name in cgraph.ext_deps for name in (val_in, treeval_in))
        assert vconst("init_vars") in cgraph.ext_deps
        rest_len = ComputationGraph.check_hash_deps(cgraph.ext_deps)
        assert rest_len == 3

        assert len(cgraph.exports) == 0
        return cgraph

    @staticmethod
    def make_init_load_graph(batch: int, val_init_out: str) -> "ComputationGraph":
        cgraph = ComputationGraph()

        def add_new_slot(op: str, args: list[str], dest: str, exports: bool = False):
            cgraph.add_new_slot(op, args, dest=dest, exports=exports, batch=batch)

        tmp_offset = [batch_localize(f"tmp_offset{i}", batch) for i in range(5)]
        tmp_offset_sum = [batch_localize(f"tmp_offset_sum{i}", batch) for i in range(4)]
        tmp_offset_final = batch_localize("tmp_offset_final", batch)
        curr_addr = batch_localize("addr_start", batch)

        batch_prime = batch // 8
        inp_values_p = (vconst("init_vars"), 6)

        if batch_prime == 0:
            add_new_slot("vload", [inp_values_p], dest=val_init_out, exports=True)
            return cgraph
        if batch_prime == 1:
            add_new_slot("+", [inp_values_p, constn(8)], dest=curr_addr)
            add_new_slot("vload", [curr_addr], dest=val_init_out, exports=True)
            return cgraph

        shifts = calculate_shifts(batch_prime)
        if len(shifts) == 1:
            # power of 2
            shift = shifts[0]
            add_new_slot("<<", [constn(1), constn(shift)], dest=tmp_offset[0])
            add_new_slot("*", [constn(8), tmp_offset[0]], dest=tmp_offset_final)
            add_new_slot(
                "+",
                [inp_values_p, tmp_offset_final],
                dest=curr_addr,
                exports=True,
            )
            add_new_slot("vload", [curr_addr], dest=val_init_out, exports=True)
            return cgraph
        # shifts
        for i, shift in enumerate(shifts):
            add_new_slot("<<", [constn(1), constn(shift)], dest=tmp_offset[i])
        # sums -- this part is variable
        if len(shifts) == 2:
            add_new_slot("+", [tmp_offset[0], tmp_offset[1]], dest=tmp_offset_sum[0])
            add_new_slot("*", [constn(8), tmp_offset_sum[0]], dest=tmp_offset_final)
        if len(shifts) == 3:
            add_new_slot("+", [tmp_offset[0], tmp_offset[1]], dest=tmp_offset_sum[0])
            add_new_slot(
                "+", [tmp_offset_sum[0], tmp_offset[2]], dest=tmp_offset_sum[1]
            )
            add_new_slot("*", [constn(8), tmp_offset_sum[1]], dest=tmp_offset_final)
        elif len(shifts) == 4:
            add_new_slot("+", [tmp_offset[0], tmp_offset[1]], dest=tmp_offset_sum[0])
            add_new_slot("+", [tmp_offset[2], tmp_offset[3]], dest=tmp_offset_sum[1])
            add_new_slot(
                "+",
                [tmp_offset_sum[0], tmp_offset_sum[1]],
                dest=tmp_offset_sum[2],
            )
            add_new_slot("*", [constn(8), tmp_offset_sum[2]], dest=tmp_offset_final)
        elif len(shifts) == 5:
            add_new_slot("+", [tmp_offset[0], tmp_offset[1]], dest=tmp_offset_sum[0])
            add_new_slot("+", [tmp_offset[2], tmp_offset[3]], dest=tmp_offset_sum[1])
            add_new_slot(
                "+",
                [tmp_offset_sum[0], tmp_offset_sum[1]],
                dest=tmp_offset_sum[2],
            )
            add_new_slot(
                "+", [tmp_offset_sum[2], tmp_offset[4]], dest=tmp_offset_sum[3]
            )
            add_new_slot("*", [constn(8), tmp_offset_sum[3]], dest=tmp_offset_final)

        # final addr calc and load
        add_new_slot(
            "+",
            [inp_values_p, tmp_offset_final],
            dest=curr_addr,
            exports=True,
        )
        add_new_slot("vload", [curr_addr], dest=val_init_out, exports=True)

        return cgraph

    @staticmethod
    def make_batch_graph(
        batch: int,
        rounds: int,
        forest_height: int,
    ) -> "ComputationGraph":
        cgraph = ComputationGraph()

        def add_new_slot(op: str, args: list[str], dest: str, exports: bool = False):
            cgraph.add_new_slot(op, args, dest=dest, exports=exports, batch=batch)

        val_init = vbatch_localize("val_init", batch)
        cgraph.merge_below(ComputationGraph.make_init_load_graph(batch, val_init))
        for round in range(rounds):
            # Output vars for this round
            val_out = vlocalize("val", batch, round)
            idx_out = vlocalize("idx", batch, round)
            treeval_out = vlocalize("treeval", batch, round)
            parity_out = vlocalize("parity", batch, round)
            # And the corresponding input vars
            val_in = vlocalize("val", batch, round - 1)
            idx_in = vlocalize("idx", batch, round - 1)
            treeval_in = vlocalize("treeval", batch, round - 1)
            parity_in = vlocalize("parity", batch, round - 1)

            if round % (forest_height + 1) == 0:
                val_in = val_init if round == 0 else val_in
                cgraph.merge_below(
                    ComputationGraph.make_top_layer_graph(
                        batch,
                        round,
                        forest_height,
                        val_in,
                        val_out,
                        treeval_out,
                        parity_out,
                    )
                )
            elif round % (forest_height + 1) == 1:
                cgraph.merge_below(
                    ComputationGraph.make_second_layer_graph(
                        batch,
                        round,
                        forest_height,
                        val_in,
                        parity_in,
                        treeval_in,
                        val_out,
                        idx_out,
                        treeval_out,
                    )
                )
            elif (round + 1) % (forest_height + 1) == 0:
                cgraph.merge_below(
                    ComputationGraph.make_wraparound_layer_graph(
                        batch, round, forest_height, val_in, treeval_in, val_out
                    )
                )
            elif round == rounds - 1:
                cgraph.merge_below(
                    ComputationGraph.make_last_layer_graph(
                        batch, round, rounds, val_in, treeval_in
                    )
                )
            else:
                cgraph.merge_below(
                    ComputationGraph.make_mid_layer_graph(
                        batch,
                        round,
                        rounds,
                        forest_height,
                        val_in,
                        idx_in,
                        treeval_in,
                        val_out,
                        idx_out,
                        treeval_out,
                    )
                )

        # treeval0,1,2 for round 0/11
        # treeval3, 4, 5, 6 for round 1/12
        # vconstn 1, 2, 3, 5
        # vconst(forest_values_p)
        top_layer_treevals = [
            vconst(f"treeval{i}_round{round}")
            for i in (0, 1, 2)
            for round in range(0, rounds, forest_height + 1)
        ]
        second_layer_treevals = [
            vconst(f"treeval{i}_round{round}")
            for i in (3, 4, 5, 6)
            for round in range(1, rounds, forest_height + 1)
        ]
        vconstns = [vconstn(i) for i in (1, 2, 3)]
        assert all(treeval in cgraph.ext_deps for treeval in top_layer_treevals)
        assert all(treeval in cgraph.ext_deps for treeval in second_layer_treevals)
        assert all(vconstn in cgraph.ext_deps for vconstn in vconstns)
        assert vconst("forest_values_p") in cgraph.ext_deps
        assert vconst("init_vars") in cgraph.ext_deps
        ComputationGraph.check_hash_deps(cgraph.ext_deps)
        # assert (
        #     rest_len
        #     == len(top_layer_treevals) + len(second_layer_treevals) + len(vconstns) + 2
        # ), f"{cgraph.ext_deps.keys()}"

        assert len(cgraph.exports) == 0
        return cgraph

    @staticmethod
    def make_kernel_graph(
        forest_height: int,
        batch_size: int,
        rounds: int,
    ) -> "ComputationGraph":
        cgraph = ComputationGraph.make_const_graph()
        # vbroadcast for round 0/11
        for round in range(0, rounds, forest_height + 1):
            cgraph.add_new_slot(
                "vbroadcast",
                [(vconst("treevals_starting0"), 0)],
                dest=vconst(f"treeval0_round{round}"),
            )
            for i in (1, 2):
                cgraph.add_new_slot(
                    "vbroadcast",
                    [(vconst("treevals_starting0"), i)],
                    dest=vconst(f"treeval{i}_round{round}"),
                )
        for round in range(1, rounds, forest_height + 1):
            for i in (3, 4, 5, 6):
                cgraph.add_new_slot(
                    "vbroadcast",
                    [(vconst("treevals_starting0"), i)],
                    dest=vconst(f"treeval{i}_round{round}"),
                )

        for batch in range(0, batch_size, VLEN):
            cgraph.merge_below(
                ComputationGraph.make_batch_graph(batch, rounds, forest_height)
            )

        assert len(cgraph.ext_deps) == 0, f"{cgraph.ext_deps.keys()}"
        assert len(cgraph.exports) == 0, f"{cgraph.exports.keys()}"
        return cgraph


class KernelGraph:
    def __init__(
        self,
        cgraph: ComputationGraph,
        scalar_freelist: set[int],
        vector_freelist: set[int],
    ):
        self.graph = cgraph.graph
        self.using = {}
        self.scalar_freelist = scalar_freelist
        self.vector_freelist = vector_freelist
        # Whenever we remove a node from the graph, we've scheduled it and its dest var is active.
        # So we need to keep track of number of forward dependencies.
        # When the last edge is removed, we can free the register.
        self.refcounts: dict[Var, int] = {}
        self.distance_to_notable = {}
        self.longest_dep_chain = {}
        self.min_vector_freelist_len = float("inf")
        self.__post_init__()

    def __post_init__(self):
        # Check that all variables have unique names in the graph
        vars = set()
        for node in self.graph.nodes:
            if node.dest is not None:
                assert node.dest not in vars
                vars.add(node.dest)
        # Get rid of dead code (terminals that are not stores)
        self.elim_dead_code()
        assert all(
            self.graph.out_degree(node) > 0
            for node in self.graph.nodes
            if node.engine != "store"
        )
        self.prune_noops()  # TODO fix
        self.populate_heuristics()

    def elim_dead_code(self):
        while True:
            terminals = [
                node
                for node in self.graph.nodes
                if self.graph.out_degree(node) == 0 and node.engine != "store"
            ]
            if len(terminals) == 0:
                break
            for terminal in terminals:
                self.graph.remove_node(terminal)

    def prune_noops(self):
        leaves = self.leaves
        for leaf in leaves:
            if leaf.engine != "NO-OP":
                continue
            assert leaf.op == "vmerge"
            # The underlying register should have already been allocated
            assert leaf.dest in self.using
            # Which slots depend on this one?
            n_forward_deps = self.graph.out_degree(leaf)
            assert n_forward_deps > 0
            assert leaf.dest not in self.refcounts
            self.refcounts[leaf.dest] = n_forward_deps
            self.graph.remove_node(leaf)

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
                        default=float("inf"),
                    )
                    + 1
                )

    @property
    def leaves(self) -> list[SymbolicInstructionSlot]:
        return [node for node in self.graph.nodes if self.graph.in_degree(node) == 0]

    @property
    def has_more_work(self) -> bool:
        return self.graph.number_of_nodes() > 0

    # Allocate output reg. If a vector offset, allocate it only once.
    def alloc_outreg(self, var: Var) -> int:
        if isinstance(var, tuple):
            vname, offset = var
            if vname in self.using:
                return self.using[vname] + offset
            if len(self.vector_freelist) == 0:
                print(self.using)
            reg = self.vector_freelist.pop()
            if len(self.vector_freelist) < self.min_vector_freelist_len:
                self.min_vector_freelist_len = len(self.vector_freelist)
                print(f"New min vector freelist len: {self.min_vector_freelist_len}")
            self.using[vname] = reg
            return reg + offset
        if is_vector(var):
            if len(self.vector_freelist) == 0:
                print(self.using)
            reg = self.vector_freelist.pop()
            if len(self.vector_freelist) < self.min_vector_freelist_len:
                self.min_vector_freelist_len = len(self.vector_freelist)
                print(f"New min vector freelist len: {self.min_vector_freelist_len}")
            self.using[var] = reg
            return reg
        assert is_scalar(var)
        reg = self.scalar_freelist.pop()
        self.using[var] = reg
        return reg

    # Frees the variable associated with a given variable name.
    def free_outreg(self, var: Var):
        if isinstance(var, tuple):
            # If this is a vector offset variable, it should have been allocated. vmerge will take care of this; we shouldn't actually free yet.
            vname, offset = var
            assert vname in self.using
            return
        assert var in self.using
        if is_vector(var):
            self.vector_freelist.add(self.using.pop(var))
            return
        assert is_scalar(var)
        self.scalar_freelist.add(self.using.pop(var))

    def schedule_round(self, round_i: int) -> Instruction:
        self.prune_noops()
        assert self.has_more_work
        bundle = defaultdict(list)

        def can_schedule(leaf: SymbolicInstructionSlot) -> bool:
            return len(bundle[leaf.engine]) < SLOT_LIMITS[leaf.engine]

        leaves = self.leaves
        # leaves.sort(key=lambda x: self.longest_dep_chain[x], reverse=True)
        # leaves.sort(key=lambda x: self.distance_to_notable[x])
        leaves.sort(
            key=lambda x: self.longest_dep_chain[x] - self.distance_to_notable[x],
            reverse=True,
        )
        for leaf in leaves:
            if not can_schedule(leaf):
                continue
            # Heuristic: don't immediately calculate ending addresses
            if leaf.op == "add_imm" and round_i < 50:
                continue

            # Allocate output reg
            out_reg = None
            if leaf.dest is not None:
                out_reg = self.alloc_outreg(leaf.dest)

            # concrete-ize the slot
            engine, inst = leaf.to_concrete(self.using, out_reg=out_reg)
            assert engine != "NO-OP"
            bundle[engine].append(inst)

            # reftracking
            # we are now defining the variable; we need to see how many depend on it
            # TODO: if fake dep, then need to check edge annotations
            n_forward_deps = self.graph.out_degree(leaf)
            if leaf.dest is None:
                assert n_forward_deps == 0
            else:
                assert n_forward_deps > 0
                assert leaf.dest not in self.refcounts
                self.refcounts[leaf.dest] = n_forward_deps

            # this might also be the last time one of our dependencies is used; if that's the case, free the register
            for arg in leaf.dependencies:
                if isinstance(arg, tuple):
                    if leaf.op != "vmerge":
                        var_name, offset = arg
                        arg = var_name
                self.refcounts[arg] -= 1
                if self.refcounts[arg] == 0:
                    # If this is a vector offset register, don't get rid of it yet! The vmerge reuses the same register
                    if isinstance(arg, tuple):
                        del self.refcounts[arg]
                        continue
                    reg = self.using.pop(arg)
                    if is_vector(arg):
                        self.vector_freelist.add(reg)
                    else:
                        self.scalar_freelist.add(reg)

            # finally, remove this leaf; expose more work
            self.graph.remove_node(leaf)

        return bundle

    def generate_code(self):
        code = []
        orig_sfreelist_len = len(self.scalar_freelist)
        orig_vfreelist_len = len(self.vector_freelist)
        round_i = 0
        while self.has_more_work:
            bundle = self.schedule_round(round_i)
            code.append(bundle)
            round_i += 1
        assert len(self.scalar_freelist) == orig_sfreelist_len
        assert len(self.vector_freelist) == orig_vfreelist_len
        return code

    @staticmethod
    def make_graph(
        forest_height: int,
        batch_size: int,
        rounds: int,
        scalar_freelist: set[int],
        vector_freelist: set[int],
    ) -> "KernelGraph":
        cgraph = ComputationGraph.make_kernel_graph(forest_height, batch_size, rounds)
        return KernelGraph(cgraph, scalar_freelist, vector_freelist)

class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

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
        # needed to match with reference
        self.add("flow", ("pause",))
        N_SCALAR_REG = 64
        scalar_freelist, vector_freelist = self.make_freelists(N_SCALAR_REG)
        kernel_graph = KernelGraph.make_graph(
            forest_height, batch_size, rounds, scalar_freelist, vector_freelist
        )
        self.instrs.extend(kernel_graph.generate_code())
        self.add("flow", ("pause",))


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
