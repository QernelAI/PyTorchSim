import torch
from PyTorchSimFrontend.mlir.mlir_common import MLIRKernelArgs
from PyTorchSimFrontend.llvm.llvm_caller_codegen import LLVMKernelCallerCodeGen
from PyTorchSimFrontend.mlir.mlir_common import DTYPE_TO_C

class MLIRKernelCallerCodeGen(LLVMKernelCallerCodeGen):

    def __init__(self, validation, arg_attributes, cycle_sim=False):
        super().__init__(validation, arg_attributes)
        self.cycle_sim = cycle_sim

    def write_header(self):
        super().write_header()
        global_var_header = "gem5_global_var.h" if self.cycle_sim else "global_var.h"
        self.writeline(f"#include \"{global_var_header}\"")

    def is_in_arg(self, value):
        return MLIRKernelArgs.is_mlir_arg_in(value)

    def is_out_arg(self, value):
        return MLIRKernelArgs.is_mlir_arg_out(value)

    def is_inout_arg(self, value):
        return MLIRKernelArgs.is_mlir_arg_inout(value)

    def load_arg(self):
        for arg_name, arg_attribute in self.arg_attributes:
            if self.is_in_arg(arg_attribute[0]):
                argv_idx = self.get_argv_idx() if arg_name not in self.load_args else self.load_args[arg_name]
                self.load_args[arg_name] = argv_idx
                self.writeline(f'if(load_arg(c_{arg_name}, sizeof(c_{arg_name}), argv[{argv_idx}]) == -1){self.open_bracket}')
                with self.code.indent():
                    self.writeline(f'return -1{self.ending}')
                self.writeline(self.closed_bracket)

    def dump_arg(self):
        for arg_name, arg_attribute in self.arg_attributes:
            if self.is_out_arg(arg_attribute[0]):
                argv_idx = self.get_argv_idx() if not self.is_inout_arg(arg_attribute[0]) else self.load_args[arg_name]
                self.writeline(f'if(dump_arg(c_{arg_name}, sizeof(c_{arg_name}), argv[{argv_idx}]) == -1){self.open_bracket}')
                with self.code.indent():
                    self.writeline(f'return -1{self.ending}')
                self.writeline(self.closed_bracket)

    def generate_kernel_declare(self):
        # memref to llvm arguments (memref -> ptr, ptr, i64, <?xi64>, <?xi64>) allocated pointer, aligned pointer, offset, size, stride
        args_type_p = [f'{DTYPE_TO_C[arg_type[1]]}*, {DTYPE_TO_C[arg_type[1]]}*, int64_t, int64_t, int64_t' for (_, arg_type) in self.arg_attributes]

        self.writeline(f"void wrapper_{self.kernel_name}({', '.join(args_type_p)}){self.ending}{self.newline}")

    def generate_args_define(self):
        name_set = set()
        if self.validation:
            self.writeline(f'int padding[0x100000]{self.ending}') # FIXME. For pooling operation... Some pooling layer use negative offset
        for arg_name, (_, arg_type, arg_size, arg_sizes, arg_stride) in self.arg_attributes:
            if not arg_name in name_set:
                if self.validation:
                    self.writeline(f'{DTYPE_TO_C[arg_type]} c_{arg_name}[{arg_size}ULL]{self.ending}')
                else:
                    if torch.is_floating_point(torch.tensor([], dtype=arg_type)):
                        bits = torch.finfo(arg_type).bits
                    elif arg_type == torch.bool:
                        bits = 8
                    else:
                        bits = torch.iinfo(arg_type).bits
                    self.writeline(f'{DTYPE_TO_C[arg_type]}* c_{arg_name} = malloc({arg_size * bits // 8}ULL){self.ending}')
                name_set.add(arg_name)
        self.writeline(self.newline)

    def generate_main(self):
        if self.validation:
            self.generate_args_define()

        self.writeline(f'{self.newline}int main(int argc, char *argv[]) {self.open_bracket}{self.newline}')
        with self.code.indent():
            if self.validation:
                self.load_arg()
                self.writeline(self.newline)
            else:
                self.generate_args_define()

            func_arguments = [f"c_{arg_name}, c_{arg_name}, 0, {arg_shape}, 1" for arg_name, (_, arg_type, arg_shape, _, _) in self.arg_attributes]
            self.writeline(f"wrapper_{self.kernel_name}({', '.join(func_arguments)}){self.ending}{self.newline}")

            if self.validation:
                self.dump_arg()

            self.write_exit()
        self.writeline(self.closed_bracket)