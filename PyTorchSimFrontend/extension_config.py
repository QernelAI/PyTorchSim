import os
import sys
import tempfile

# Hardware info config
CONFIG_VECTOR_LANE = int(os.environ.get("TORCHSIM_VECTOR_LANE", default=128))
CONFIG_SPAD_INFO = {
  "spad_vaddr" : 0xD0000000,
  "spad_paddr" : 0x2000000000,
  "spad_size" : 128 << 10
}
CONFIG_PRECISION = 4 # 32bit
CONFIG_NUM_CORES = 1
CONFIG_VLEN = 32 // CONFIG_PRECISION # 256bits / 32bits = 8 [elements]

# Tile size config
CONFIG_TORCHSIM_DIR = os.environ.get('TORCHSIM_DIR', default='/workspace/PyTorchSim')

# DUMP PATH
CONFIG_BACKEND_RESULT_PATH_KEY = os.getenv("BACKEND_RESULT_PATH_KEY")

CONFIG_TORCHSIM_DUMP_PATH = os.environ.get('TORCHSIM_DUMP_PATH',
                        default = f"{tempfile.gettempdir()}/torchinductor")
CONFIG_TORCHSIM_DUMP_FILE = int(os.environ.get('TORCHSIM_DUMP_FILE', default=True))
CONFIG_TORCHSIM_VALIDATION_MODE = int(os.environ.get('TORCHSIM_VALIDATION_MODE', default=True))
CONFIG_CLEANUP_DUMP_ARGS = int(os.environ.get('CLEANUP_DUMP_ARGS', default=False))

# LLVM PATH
CONFIG_TORCHSIM_LLVM_PATH = os.environ.get('TORCHSIM_LLVM_PATH', default="/usr/bin")
CONFIG_TORCHSIM_CUSTOM_PASS_PATH = os.environ.get('TORCHSIM_CUSTOM_PASS_PATH',
                                           default=f"{CONFIG_TORCHSIM_DIR}/GemminiLowerPass/build")
CONFIG_TORCHSIM_DUMP_MLIR_IR = int(os.environ.get("TORCHSIM_DUMP_MLIR_IR", default=False))

# Backendsim config
CONFIG_TORCHSIM_BACKEND_CONFIG = os.environ.get('TORCHSIM_CONFIG',
                                        default=f'{CONFIG_TORCHSIM_DIR}/PyTorchSimBackend/configs/systolic_ws_128x128_c1_simple_noc_tpuv2.json')
CONFIG_BACKENDSIM_SPIKE_ONLY = int(os.environ.get("BACKENDSIM_SPIKE_ONLY", False))
CONFIG_BACKENDSIM_EAGER_MODE = int(os.environ.get("BACKENDSIM_EAGER_MODE", default=False))
CONFIG_BACKENDSIM_DRYRUN = int(os.environ.get('BACKENDSIM_DRYRUN', default=False))
CONFIG_BACKENDSIM_DEBUG_LEVEL = os.environ.get("BACKENDSIM_DEBUG_LEVEL", "")

# GEM5 config
CONFIG_GEM5_PATH = os.environ.get('GEM5_PATH', default="/workspace/gem5/build/RISCV/gem5.opt")
CONFIG_GEM5_SCRIPT_PATH = os.environ.get('GEM5_SCRIPT_PATH',
                                  default=f"{CONFIG_TORCHSIM_DIR}/gem5_script/script_systolic.py")

# For block sparse
CONFIG_BLOCK_SPARSE = int(os.environ.get('BLOCK_SPARSE', default=0))
CONFIG_FORCE_TILE_M = int(os.environ.get("TORCHSIM_FORCE_TIME_M", default=sys.maxsize))
CONFIG_FORCE_TILE_N = int(os.environ.get("TORCHSIM_FORCE_TIME_N", default=sys.maxsize))
CONFIG_FORCE_TILE_K = int(os.environ.get("TORCHSIM_FORCE_TIME_K", default=sys.maxsize))