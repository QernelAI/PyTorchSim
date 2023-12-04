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
    """Map element-wise ops to LLVM IR"""

    @staticmethod
    def add(self, other):
        return f'fadd float {self}, {other}' # TODO: separate float and integer

    def sub(self, other):
        return f'fsub float {self}, {other}'

    def mul(self, operand1, operand2):
        return f'fmul float {operand1}, {operand2}'

    def div(self, operand1, operand2):
        return f'fdiv float {operand1}, {operand2}'


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
        type_name = llvm_common.DTYPE_TO_LLVM[dtype]
        align = llvm_common.DTYPE_SIZE[dtype]
        line = f"mul nsw i64 %{index}, {align}"
        offset = self.cse.generate(self.loads, line)
        line = f"getelementptr inbounds {type_name}, ptr %{var}, i64 {offset}" # TODO: index for loop
        var = self.cse.generate(self.loads, line)
        line = f"load {type_name}, ptr {var}, align {align}"
        return self.cse.generate(self.loads, line)

    def store(self, name: str, index: sympy.Expr, value, *args, **kwargs):
        index = self.rename_indexing(index)
        var = self.args.output(name)
        dtype = V.graph.get_dtype(name)
        type_name = llvm_common.DTYPE_TO_LLVM[dtype]
        align = llvm_common.DTYPE_SIZE[dtype]
        line = f"mul nsw i64 %{index}, {align}"
        offset = self.cse.generate(self.loads, line)
        line = f"getelementptr inbounds {type_name}, ptr %{var}, i64 {offset}"
        var = self.cse.generate(self.stores, line)
        line = f"store {type_name} {value}, ptr {var}, align {align}"
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
        loops = [LoopLevel(var, size, idx) for idx, (var, size) in enumerate(zip(self.itervars, self.ranges))]
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
        code.writeline(f"ret void")
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
            self.itervars = [sympy.Symbol(f"loop{n}") for n in range(len(self.ranges))]
            self.reduction_depth = len(lengths)
        return (
            self.itervars[: self.reduction_depth],
            self.itervars[self.reduction_depth :],
        )

@dataclasses.dataclass
class LoopLevel:
    var: sympy.Expr
    size: sympy.Expr
    idx: int
    reduction_vars: Dict[str, str] = None

    # Todo. Type change for reduction
    INDEX_TYPE = "i64"
    INDEX_SIZE = 8
    def init_lines(self):
        idx_var = f"%loop_idx{self.idx}"
        return f"{idx_var} = alloca {self.INDEX_TYPE}, align {self.INDEX_SIZE}"

    def lines(self, line, stride=1):
        loop_index = self.idx
        @contextlib.contextmanager
        def ctx():
            idx_var = f"%loop_idx{loop_index}"
            loop_var = f"%loop{loop_index}"
            loop_var2 = f"%loop.inc{loop_index}"
            loop_var3 = f"%inc{loop_index}"
            cmp_var = f"%cmp{loop_index}"
            line.writeline(f"store {self.INDEX_TYPE} 0, ptr {idx_var}, align {self.INDEX_SIZE}")
            line.writeline(f"br label %for{loop_index}")

            line.writeline(f"\nfor{loop_index}:")
            line.writeline(f"{loop_var} = load {self.INDEX_TYPE}, ptr {idx_var}, align {self.INDEX_SIZE}")
            line.writeline(f"{cmp_var} = icmp slt {self.INDEX_TYPE} {loop_var}, {self.size}")
            line.writeline(f"br i1 {cmp_var}, label %for.body{loop_index}, label %for.end{loop_index}")

            line.writeline(f"\nfor.body{loop_index}:")
            yield
            line.writeline(f"br label %for.inc{loop_index}")
            line.writeline(f"\nfor.inc{loop_index}:")
            line.writeline(f"{loop_var2} = load {self.INDEX_TYPE}, ptr {idx_var}, align {self.INDEX_SIZE}")
            line.writeline(f"{loop_var3} = add nsw {self.INDEX_TYPE} {loop_var2}, 1")
            line.writeline(f"store {self.INDEX_TYPE} {loop_var3}, ptr {idx_var}, align {self.INDEX_SIZE}")
            line.writeline(f"br label %for{loop_index}")
            line.writeline(f"\nfor.end{loop_index}:")
        return ctx()


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
        size_list = []
        stride_list = []
        for loop in self.loops:
            stride_list.append(loop.size)
            code.writeline(loop.init_lines())
        stride_list.append(1)

        var = 1
        for sz in size_list[::-1]:
            var = var * sz
            stride_list.append(var)
        stride_list = stride_list[::-1]

        for loop, stride in zip(self.loops, stride_list):
            stack.enter_context(loop.lines(code, stride=stride))

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