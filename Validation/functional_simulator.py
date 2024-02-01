import os
import shlex
import subprocess

import torch
import numpy as np

from extension_backends.llvm_common import LLVMKernelArgs

class FunctionalSimulator():
    def __init__(self, path, key, arg_attributes, args):
        self.path = path
        self.key = key
        self.arg_attributes = arg_attributes
        self.args = args

    def load_tensor(self, arg, arg_name, arg_attribute, n_call):
        path = os.path.join(self.path, arg_name, f'{n_call}.raw')

        with open(path, 'rb') as f:
            np_array = np.fromfile(f)
            src_tensor = torch.from_numpy(np_array).view(dtype=arg.dtype)
            src_tensor = src_tensor.reshape(arg.shape)
            arg.copy_(src_tensor)

    def run_spike(self, n_call, target_binary):
        run = f'spike --isa rv64gcv /workspace/riscv-pk/build/pk {target_binary} {n_call}'
        run_cmd = shlex.split(run)
        try:
            subprocess.check_call(run_cmd)
        except subprocess.CalledProcessError as e:
            print("Spike error")
            assert(0)

        for (arg_name, arg_attribute), arg in zip(self.arg_attributes.items(), self.args):
            if LLVMKernelArgs.is_llvm_arg_out(arg_attribute[0]):
                self.load_tensor(arg, arg_name, arg_attribute, n_call)