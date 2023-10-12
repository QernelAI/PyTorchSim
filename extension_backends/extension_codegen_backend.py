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
    def __init__(self, args=None):
        super().__init__(args)

    def load(self, name: str, index: sympy.Expr):
        var = self.args.input(name)
        line = f"Extension.load({name}, {index})"
        line = f"{var}" + line
        return self.cse.generate(self.loads, line)

    def store(self, name, index, value, *args, **kwargs):
        var = self.args.output(name)
        line = f"Extension.store({name}, {index}, {value})"
        line = f"{var}" + line
        self.stores.writeline(line)
    
    def reduction(self, dtype, src_dtype, reduction_type, value):
        # Todo. 1. args handling
        line = f"Extension.reduction(dtype={dtype}, src_dtype={src_dtype},\
                reduction_type={reduction_type}, value={value})"
        self.cse.generate(self.compute, line)
        
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
            code.splice(self.loads)
            code.splice(self.compute)
            code.splice(self.stores)

        wrapper.define_kernel(kernel_name, code.getvalue(), cuda=False)
        # generate the code to call this
        wrapper.generate_kernel_call(kernel_name, call_args, cuda=False)
        print(code.getvalue())


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
        for node in nodes:
            ex_kernel = ExtensionKernel()
            with ex_kernel:
                node.codegen(node.get_ranges())

        wrapper = V.graph.wrapper_code
        ex_kernel.codegen_kernel(wrapper)
        pass

    def codegen_sync(self):
        pass

    def flush(self):
        self._scheduling.flush()