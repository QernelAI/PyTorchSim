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
OutputPath = ''

def llvm_compile_command(input, output):
    return re.sub(
        r"[ \n]+",
        " ",
        f"""
            llc -march=riscv64 -mattr=+m,+f,+d,+a,+c,+v -O2 {input} -o {output}
        """,
    ).strip()

class LLVMCodeCache:
    cache = dict()
    clear = staticmethod(cache.clear)   # Todo: Cache

    @staticmethod
    def _load_library(path):
        pass

    @classmethod
    def load(cls, source_code):
        global TORCHSIM_DUMP_PATH
        write_path = os.path.join(TORCHSIM_DUMP_PATH, "tmp")
        key, input_path = write(source_code, "ll", specified_dir=write_path)
        output_path = input_path[:-2] + "s"

        cmd = shlex.split(
            llvm_compile_command(input_path, output_path)
        )

        from filelock import FileLock
        lock_dir = get_lock_dir()
        lock = FileLock(os.path.join(lock_dir, key + ".lock"), timeout=LOCK_TIMEOUT)
        with lock:
            if not os.path.exists(output_path):
                try:
                    subprocess.check_call(cmd)
                except subprocess.CalledProcessError as e:
                    assert(0)   # Todo: make LLVMCompileError
            else:
                pass
        return output_path

class CustomAsyncCompile(AsyncCompile):
    def llvm(self, source_code):
        def task():
            global OutputPath
            OutputPath = LLVMCodeCache.load(source_code)
            return
        future = self.submit(task)

        def dummy_simulator(*args, **kwargs):
            future.result()
            os.system('echo "Running dummy simulator!"')
            global OutputPath
            print("OUTPUT PATH > ", OutputPath)
            try:
                with open(OutputPath, 'r') as file:
                    file_contents = file.read()

                    print("Assembly > ")
                    print(file_contents)
            except FileNotFoundError:
                print(f'{OutputPath} not found.')
            except Exception as e:
                print(f"Error while reading {OutputPath}.")
        return dummy_simulator