import torch
from . import common
from torch._inductor.utils import get_sympy_Expr_dtype, unique
from torch._inductor.virtualized import V
import sympy

DTYPE_TO_LLVM = {
    torch.float32: "float",
    torch.float64: "double",
    torch.float16: "half",
    torch.int64: "i64",
    torch.int32: "i32",
    torch.int16: "i16",
    torch.int8: "i8",
    torch.uint8: "i8",
    torch.bool: "i8",
    torch.bfloat16: "bfloat",
}

class LLVMKernelArgs(common.KernelArgs):
    def llvm_argdefs(self):
        buffer_types = {x.get_name(): x.get_dtype() for x in V.graph.buffers}
        for name, val in V.graph.graph_inputs.items():
            if isinstance(val, sympy.Expr):
                buffer_types[name] = get_sympy_Expr_dtype(val)
            else:
                buffer_types[name] = val.get_dtype()
        buffer_types.update(
            {name: val.dtype for name, val in V.graph.constants.items()}
        )

        call_args = []
        arg_defs = []
        for inplaced in unique(self.inplace_buffers.values()):
            if self._buffer_is_marked_removed(inplaced):
                continue
            outer = inplaced.other_names[-1]
            inner = inplaced.inner_name
            dtype = buffer_types[outer]
            arg_defs.append(f"ptr %{inner}")
            call_args.append(self.wrap_ptr_arg(outer, dtype))
        for outer, inner in self.input_buffers.items():
            if outer in self.inplace_buffers:
                continue
            dtype = buffer_types[outer]
            arg_defs.append(f"ptr readonly %{inner}")
            call_args.append(self.wrap_ptr_arg(outer, dtype))
        for outer, inner in self.output_buffers.items():
            if outer in self.inplace_buffers or self._buffer_is_marked_removed(inner):
                continue
            dtype = buffer_types[outer]
            arg_defs.append(f"ptr %{inner}")
            call_args.append(self.wrap_ptr_arg(outer, dtype))
        for outer, inner in self.sizevars.items():
            arg_defs.append(f"ptr readonly %{inner}")
            call_args.append(self.wrap_size_arg(outer))
        return arg_defs, call_args