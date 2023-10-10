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
        line = f"Extension.load({name}, {index})"
        return self.cse.generate(self.loads, line)

    def store(self, name, index, value, *args, **kwargs):
        line = f"Extension.store({name}, {index}, {value})"
        self.stores.writeline(line)
    
    def reduction(self, dtype, src_dtype, reduction_type, value):
        line = f"Extension.reduction(dtype={dtype}, src_dtype={src_dtype},\
                reduction_type={reduction_type}, value={value})"
        self.cse.generate(self.compute, line)
        
    def codegen_kernel(self):
        code = IndentedBuffer()
        code.splice(
            f"""
            # This is dummy code for Extension device kernel
            # Hello Extension Code!
            """
        )
        code.splice(self.loads)
        code.splice(self.compute)
        code.splice(self.stores)
        return code.getvalue()


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
        kernel_name = f"Extensin_node_{self.count}"
        self.count += 1
        wrapper = V.graph.wrapper_code
        wrapper.define_kernel(
            kernel_name, ex_kernel.codegen_kernel(), "\nmeta\n"
        )
        pass

    def codegen_sync(self):
        pass

    def flush(self):
        self._scheduling.flush()