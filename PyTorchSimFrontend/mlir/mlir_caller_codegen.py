import os
import subprocess
import shlex

from torch._inductor.utils import IndentedBuffer
from torch._inductor.codegen import cpp
from torch._inductor.codecache import write_atomic

from PyTorchSimFrontend.llvm.llvm_common import LLVMKernelArgs
from PyTorchSimFrontend.llvm.llvm_caller_codegen import LLVMKernelCallerCodeGen

class MLIRKernelCallerCodeGen(LLVMKernelCallerCodeGen):

    def __init__(self, validation, arg_attributes):
        super().__init__(validation, arg_attributes)

    def write_header(self):
        super().write_header()
        self.writeline(f"#include \"global_var.h\"")

    def generate_kernel_declare(self):
        # memref to llvm arguments (memref -> ptr, ptr, i64, <?xi64>, <?xi64>) allocated pointer, aligned pointer, offset, size, stride
        args_type_p = [f'{cpp.DTYPE_TO_CPP[arg_type[1]]}*, {cpp.DTYPE_TO_CPP[arg_type[1]]}*, int64_t, int64_t, int64_t' for arg_type in self.arg_attributes.values()]

        self.writeline(f"void {self.kernel_name}({', '.join(args_type_p)}){self.ending}{self.newline}")

    def generate_args_define(self):
        for arg_name, (_, arg_type, arg_shape) in self.arg_attributes.items():
            self.writeline(f'{cpp.DTYPE_TO_CPP[arg_type]} {arg_name}[atoi(argv[{self.get_argv_idx()}])] __attribute__ ((aligned (4096))){self.ending}')
        self.writeline(self.newline)

    def generate_main(self):
        self.writeline(f'{self.newline}int main(int argc, char *argv[]) {self.open_bracket}{self.newline}')
        with self.code.indent():
            self.generate_args_define()
            if self.validation:
                self.load_arg()
                self.writeline(self.newline)

            func_arguments = [f"{arg_name}, {arg_name}, 0, {arg_shape}, 1" for arg_name, (_, _, arg_shape) in self.arg_attributes.items()]
            self.writeline(f"{self.kernel_name}({', '.join(func_arguments)}){self.ending}{self.newline}")

            if self.validation:
                self.dump_arg()

            self.write_exit()
        self.writeline(self.closed_bracket)