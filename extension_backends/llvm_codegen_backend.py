import dataclasses
import contextlib
from typing import List
from typing import Set
from typing import Dict
from torch._inductor.codegen import cpp, wrapper, common
from . import llvm_common
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

class ExtensionKernel(llvm_common.LLVM_Kernel):
    overrides = ExtensionOverrides
    newvar_prefix = ""
    # suffix = ';'
    def __init__(self, args=None):
        super().__init__(llvm_common.LLVMKernelArgs())
        self.call_ranges = None
        self.ranges = None
        self.itervars = None
        self.reduction_depth = None
        self.reduction_prefix = IndentedBuffer()
        self.reduction_suffix = IndentedBuffer()
        self.reduction_vars = {}
        self.reduction_cse = common.CSE(self.newvar_prefix, self.suffix, name_prefix="tmp_acc")

    def load(self, name: str, index: sympy.Expr):
        index = self.rename_indexing(index)
        var = self.args.input(name)
        dtype = V.graph.get_dtype(name)
        type_name = cpp.DTYPE_TO_CPP[dtype]
        line = f"getelementptr inbounds {type_name}, ptr %{var}, i64 %{index}" # TODO: index for loop
        var = self.cse.generate(self.loads, line)
        line = f"load {type_name}, ptr {var}, align 4"
        return self.cse.generate(self.loads, line)

    def store(self, name: str, index: sympy.Expr, value, *args, **kwargs):
        index = self.rename_indexing(index)
        var = self.args.output(name)
        dtype = V.graph.get_dtype(name)
        type_name = cpp.DTYPE_TO_CPP[dtype]
        line = f"getelementptr inbounds {type_name}, ptr %{var}, i64 %{index}"
        var = self.cse.generate(self.stores, line)
        line = f"store {type_name} {value}, ptr {var}, align 4"
        self.cse.generate(self.stores, line, assignment = False)

    def reduction(self, dtype, src_dtype, reduction_type, value):
        raise NotImplementedError()
        argmax_or_argmin = reduction_type in {"argmax", "argmin"}
        if argmax_or_argmin:
            raise NotImplementedError() #TODO: argmin, argmax
        else:
            reduction_key = src_dtype, reduction_type, value
            acc = self.reduction_cse.generate(
                self.loads, f"reduction {reduction_key}", write=False
            )
            self.reduction_vars[acc] = reduction_type
            acc_type = cpp.reduction_acc_type(reduction_type, dtype)
            self.reduction_prefix.writeline(f"{acc_type} {acc} = {cpp.reduction_init(reduction_type, dtype)};")
            line = f"{acc} = {cpp.reduction_combine(reduction_type, acc, value)}"
            self.cse.generate(self.stores, line, assignment = False)
            self.reduction_cse.reduction_cache[reduction_key] = acc
        return acc

    def store_reduction(self, name, index, value):
        raise NotImplementedError()
        index = self.rename_indexing(index)
        var = self.args.output(name)
        self.reduction_suffix.writeline(f"{var}[{index}] = {value};")

    def codegen_loops(self):
        code = common.BracesBuffer()
        # Loop body part
        loops = [LoopLevel(var, size) for var, size in zip(self.itervars, self.ranges)]
        loops, reductions = [LoopNest(loops[: self.reduction_depth]),
                             LoopNest(loops[self.reduction_depth :])]
        reductions.mark_reduction(self.reduction_vars)

        with contextlib.ExitStack() as stack:
            loops.codegen(code, stack)
            with contextlib.ExitStack() as stack_outer:
                if self.reduction_prefix:
                    stack_outer.enter_context(code.indent())
                code.splice(self.reduction_prefix)
                with contextlib.ExitStack() as stack:
                    reductions.codegen(code, stack)
                    code.splice(self.loads)
                    code.splice(self.compute)
                    code.splice(self.stores)
                code.splice(self.reduction_suffix)
        return code

    def codegen_kernel(self, wrapper):
        arg_defs, call_args = self.args.llvm_argdefs()
        arg_defs = ",\n".ljust(25).join(arg_defs)
        code = common.BracesBuffer()

        # Todo. kernel name custom
        kernel_name = f"Extensin_Kernel"
        kernel_decl_name = kernel_name if V.graph.cpp_wrapper else "kernel"
        code.writeline(f'define void @{kernel_decl_name}({arg_defs})')
        with code.indent():
            for old, new in self.args.aliases():
                code.writeline(f"auto {old} = {new};")
            # Loop body part
            code.splice(self.codegen_loops())

        codecache_def = IndentedBuffer()
        if not V.graph.cpp_wrapper:
            codecache_def.writeline("async_compile.cpp('''")
        codecache_def.splice(code)
        if not V.graph.cpp_wrapper:
            codecache_def.writeline("''')")

        wrapper.define_kernel(kernel_name, codecache_def.getvalue(), cuda=False)
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
    reduction_vars: Dict[str, str] = None

    # Todo. Type change for reduction
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
        return False

    def can_fuse_horizontal(self, node1, node2):
        return False

    def group_fn(self, sizes):
        return tuple(tuple(map(V.graph.sizevars.simplify, s)) for s in sizes)

    def codegen_nodes(self, nodes):
        _, (group, reduction_group) = max(
            nodes, key=lambda x: int(x.is_reduction())
        ).group
        ex_kernel = ExtensionKernel()
        with ex_kernel as kernel:
            for node in nodes:
                vars, reduction_vars = ex_kernel.set_ranges(group, reduction_group)
                node.run(vars, reduction_vars)

        wrapper = V.graph.wrapper_code
        ex_kernel.codegen_kernel(wrapper)
        pass

    def codegen_sync(self):
        pass

    def flush(self):
        self._scheduling.flush()