import functools
import getpass
import tempfile
import os
import re
import shlex
import subprocess

import torch
from torch._inductor.codecache import AsyncCompile, code_hash, get_path, get_lock_dir, get_hash, write, write_atomic

from concurrent.futures import Future

LOCK_TIMEOUT = 600
TORCHSIM_DUMP_PATH = os.environ.get('TORCHSIM_DUMP_PATH',
                        default = f"{tempfile.gettempdir()}/torchinductor_{getpass.getuser()}")

def hash_prefix(hash_value):
    return hash_value[1:5]

def write_arg(arg, path, index, n_call):
    if (isinstance(arg, torch.Tensor)):
        meta_path = os.path.join(path, f'meta_{n_call}.txt')
        data_path = os.path.join(path, f'arg_{n_call}_{index}.raw')
        tensor = arg.cpu()
        try:
            with open(meta_path, "a") as file:
                file.write(f'arg_{index}=({tensor.dtype}, {tensor.shape})\n')
                file.close()
        except Exception as e:
            print(f"Error while writing meta data.")

        t_arr = tensor.numpy().flatten()
        t_arr.tofile(data_path)
    else:
        assert(0)

def dump_args(args, path):
    n_arg = len(args)

    file_pattern = re.compile(r'meta_\d+\.txt')
    matching_files = []

    for _, _, files in os.walk(path):
        for file in files:
            if file_pattern.match(file):
                matching_files.append(int(file[5:-4]))
    matching_files.sort(reverse=True)

    n_call = 0 if len(matching_files) == 0 else matching_files[0]+1

    for i in range(n_arg):
        write_arg(args[i], path, i, n_call)

def llvm_compile_command(input, output):
    opt_output = f"{input[:-3]}_opt.ll"
    return [re.sub(r"[ \n]+", " ",
        f"""
            opt -march=riscv64 -passes=lower-matrix-intrinsics {input} -o {opt_output}
        """,
    ).strip(),
            re.sub(r"[ \n]+", " ",
        f"""
            llc -march=riscv64 -mattr=+m,+f,+d,+a,+c,+v -O2 {opt_output} -o {output}
        """,
    ).strip()]

class LLVMCodeCache:
    cache = dict()
    clear = staticmethod(cache.clear)   # Todo: Cache

    @staticmethod
    def _load_library(path):
        pass

    @classmethod
    def load(cls, source_code):
        global TORCHSIM_DUMP_PATH
        write_path = os.path.join(TORCHSIM_DUMP_PATH, "tmp", hash_prefix(get_hash(source_code)))
        key, input_path = write(source_code, "ll", specified_dir=write_path)
        output_path = input_path[:-2] + "s"

        cmds = llvm_compile_command(input_path, output_path)
        opt_cmd = shlex.split(cmds[0])
        llc_cmd = shlex.split(cmds[1])

        from filelock import FileLock
        lock_dir = get_lock_dir()
        lock = FileLock(os.path.join(lock_dir, key + ".lock"), timeout=LOCK_TIMEOUT)
        with lock:
            if not os.path.exists(output_path):
                try:
                    subprocess.check_call(opt_cmd)
                    subprocess.check_call(llc_cmd)
                except subprocess.CalledProcessError as e:
                    assert(0)   # Todo: make LLVMCompileError
            else:
                pass
        return key

class CustomAsyncCompile(AsyncCompile):
    def __init__(self):
        pass

    def llvm(self, source_code):
        def task():
            self.key = LLVMCodeCache.load(source_code)
            return
        future = self.submit(task)

        def dummy_simulator(*args, **kwargs):
            future.result()
            os.system('echo "Running dummy simulator!"')
            result_path = os.path.join(TORCHSIM_DUMP_PATH, "tmp", hash_prefix(self.key))
            print("OUTPUT PATH > ", result_path)

            dump_args(args, result_path)

            assembly_path = os.path.join(result_path, f'{self.key}.s')
            try:
                with open(assembly_path, 'r') as file:
                    file_contents = file.read()
                    print("Assembly > \n", file_contents)
            except FileNotFoundError:
                print(f'{assembly_path} not found.')
            except Exception as e:
                print(f"Error while reading.")
        return dummy_simulator