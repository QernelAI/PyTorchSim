import torch
from torch._inductor.select_algorithm import ExternKernelChoice

class MLIRExternKernelChoice(ExternKernelChoice):
    def call_name(self):
        return f"torch.ops.extension_op.{self.name}"

custom_lib = torch.library.Library("extension_op", "DEF")

# FIXME: Custom op is defined in this file for example. Need refactoring
def _sparse_mm(a, b, out):
    print("PYTHON CUSTOM OP EXAMPLE")
    out.copy_(a + b)

custom_lib.define("_sparse_mm(Tensor a, Tensor b, Tensor out) -> Tensor")
custom_lib.impl("_sparse_mm", _sparse_mm, "PrivateUse1")
custom_lib.impl("_sparse_mm", _sparse_mm, "AutogradPrivateUse1")
