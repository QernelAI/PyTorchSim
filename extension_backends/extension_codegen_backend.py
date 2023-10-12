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

cexpr = cpp.CppPrinter().doprint

class ExtensionWrapperCodegen(wrapper.WrapperCodeGen):
    def __init__(self):
        super().__init__()
        
class ExtensionOverrides(common.OpOverrides):
    pass

class ExtensionKernel(common.Kernel):
    overrides = ExtensionOverrides
    def __init__(self, args=None):
        super().__init__(args)
        self.call_ranges = None
        self.ranges = None
        self.itervars = None
        self.reduction_depth = None
        
    def load(self, name: str, index: sympy.Expr):
        var = self.args.input(name)
        line = f"Extension.load({name}, {index})"
        line = f"{var}[{line}]"
        return self.cse.generate(self.loads, line)

    def store(self, name, index, value, *args, **kwargs):
        var = self.args.output(name)
        line = f"Extension.store({name}, {index}, {value})"
        line = f"{var}[{line}]"
        self.stores.writeline(line)
    
    def reduction(self, dtype, src_dtype, reduction_type, value):
        # Todo. 1. args handling
        line = f"Extension.reduction(dtype={dtype}, src_dtype={src_dtype},\
                reduction_type={reduction_type}, value={value})"
        self.cse.generate(self.compute, line)

    def codegen_loops(self):
        code = common.BracesBuffer()
        # Loop body part
        loops = [LoopLevel(var, size) for var, size in zip(self.itervars, self.ranges)]
        loops = LoopNest(loops[: self.reduction_depth])

        with contextlib.ExitStack() as stack:
            loops.codegen(code, stack)
            with contextlib.ExitStack() as stack_outer:
                with contextlib.ExitStack() as stack:
                    code.splice(self.loads)
                    code.splice(self.compute)
                    code.splice(self.stores)
        return code
        
    def codegen_kernel(self, wrapper):
        arg_defs, call_args, arg_types = self.args.cpp_argdefs()
        arg_defs = ",\n".ljust(25).join(arg_defs)
        arg_types = ",".join(arg_types)
        code = common.BracesBuffer()

        kernel_name = f"Extensin_Kernel"
        kernel_decl_name = kernel_name if V.graph.cpp_wrapper else "kernel"
        code.writeline(f'extern "C" void {kernel_decl_name}({arg_defs})')
        with code.indent():
            for old, new in self.args.aliases():
                code.writeline(f"auto {old} = {new};")
            # Loop body part
            code.splice(self.codegen_loops())

        wrapper.define_kernel(kernel_name, code.getvalue(), cuda=False)
        # generate the code to call this
        wrapper.generate_kernel_call(kernel_name, call_args, cuda=False)
        print(code.getvalue())
        return code.getvalue()

    def set_ranges(self, lengths, reduction_lengths):
        if self.call_ranges:
            assert self.call_ranges == tuple(lengths) + tuple(
                reduction_lengths
            ), f"{self.call_ranges} == {tuple(lengths)} + {tuple(reduction_lengths)}"
            assert self.reduction_depth == len(lengths)
        else:
            self.call_ranges = tuple(lengths) + tuple(reduction_lengths)
            self.ranges = [self.rename_indexing(x) for x in self.call_ranges]
            self.itervars = [sympy.Symbol(f"i{n}") for n in range(len(self.ranges))]
            self.reduction_depth = len(lengths)
        return (
            self.itervars[: self.reduction_depth],
            self.itervars[self.reduction_depth :],
        )

@dataclasses.dataclass
class LoopLevel:
    var: sympy.Expr
    size: sympy.Expr

    INDEX_TYPE = "long"
    def lines(self):
        line = f"for({self.INDEX_TYPE} {self.var}=0; {self.var}<{cexpr(self.size)}; ++{self.var})"
        return [line]

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
        return True

    def can_fuse_horizontal(self, node1, node2):
        return True

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