import dataclasses
import contextlib
from typing import List
from typing import Set
from typing import Dict
from torch._inductor.codegen import cpp, wrapper, common
from torch._inductor.scheduler import BaseScheduling
from torch._inductor.virtualized import V
from torch._inductor.utils import IndentedBuffer
import sympy

class ExtensionWrapperCodegen(wrapper.WrapperCodeGen):
    def __init__(self):
        super().__init__()

class ExtensionOverrides(common.OpOverrides):
    pass

class ExtensionKernel(common.Kernel):
    overrides = ExtensionOverrides
    newvar_prefix = ""
    suffix = ""

    def __init__(self, args=None):
        super().__init__(args)
        self.call_ranges = None
        self.ranges = None
        self.itervars = None
        self.reduction_depth = None
        self.reduction_prefix = IndentedBuffer()
        self.reduction_suffix = IndentedBuffer()
        self.reduction_vars = {}
        self.reduction_cse = common.CSE(self.newvar_prefix, self.suffix, name_prefix="tmp_acc")

    def load(self, name: str, index: sympy.Expr):
        pass

    def store(self, name: str, index: sympy.Expr, value, *args, **kwargs):
        pass

    def reduction(self, dtype, src_dtype, reduction_type, value):
        pass

    def store_reduction(self, name, index, value):
        pass

    def codegen_loops(self):
        pass

    def codegen_kernel(self, wrapper):
        pass

    def set_ranges(self, lengths, reduction_lengths):
        pass

@dataclasses.dataclass
class LoopLevel:
    var: sympy.Expr
    size: sympy.Expr
    reduction_vars: Dict[str, str] = None

    # Todo. Type change for reduction
    def lines(self):
        pass

@dataclasses.dataclass
class LoopNest:
    loops: List[LoopLevel]

    def __bool__(self):
        return bool(self.loops)

    def mark_reduction(self, reduction_vars):
        for loop in self.loops:
            loop.reduction_vars = reduction_vars

    def mark_parallel(self, par_depth):
        loops = self.loops
        loops[0].parallel = par_depth
        for i in range(1, par_depth):
            loops[i].collapsed = True
        loops[0].simd = loops[par_depth - 1].simd

    def codegen(self, code, stack):
        for loop in self.loops:
            code.writelines(loop.lines())
            stack.enter_context(code.indent())

class ExtensionScheduling(BaseScheduling):
    count = 0
    def __init__(self, scheduler):
        self.scheduler = scheduler
        self._scheduling = cpp.CppScheduling(scheduler)

    def can_fuse_vertical(self, node1, node2):
        return False

    def can_fuse_horizontal(self, node1, node2):
        return False

    def group_fn(self, sizes):
        return tuple(tuple(map(V.graph.sizevars.simplify, s)) for s in sizes)

    def codegen_nodes(self, nodes):
        _, (group, reduction_group) = max(
            nodes, key=lambda x: int(x.is_reduction())
        ).group

        for node in nodes:
            ex_kernel = ExtensionKernel()
            vars, reduction_vars = ex_kernel.set_ranges(group, reduction_group)
            with ex_kernel:
                node.run(vars, reduction_vars)

        wrapper = V.graph.wrapper_code
        ex_kernel.codegen_kernel(wrapper)
        pass

    def codegen_sync(self):
        pass

    def flush(self):
        self._scheduling.flush()