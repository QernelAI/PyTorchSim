import os
import sys
import tempfile
import importlib

def __getattr__(name):

    # Hardware info config
    if name == "CONFIG_VECTOR_LANE":
        return int(os.environ.get("TORCHSIM_VECTOR_LANE", default=128))
    if name == "CONFIG_VECTOR_LANE_STRIDE":
        return int(os.environ.get("TORCHSIM_VECTOR_LANE_STRIDE", default=2))
    if name == "CONFIG_SPAD_INFO":
        return {
          "spad_vaddr" : 0xD0000000,
          "spad_paddr" : 0x2000000000,
          "spad_size" : int(os.environ.get("TORCHSIM_SPAD_SIZE", default=128)) << 10 # Note: spad size per lane
        }
    if name == "CONFIG_PRECISION":
          return 4 # 32bit
    if name == "CONFIG_NUM_CORES":
          return 1
    if name == "CONFIG_VLEN":
          return 256 # 256bits / 32bits = 8 [elements]

    # Tile size config
    if name == "CONFIG_TORCHSIM_DIR":
          return os.environ.get('TORCHSIM_DIR', default='/workspace/PyTorchSim')

    if name == "CONFIG_TORCHSIM_DUMP_PATH":
          return os.environ.get('TORCHSIM_DUMP_PATH', default = f"{tempfile.gettempdir()}/torchinductor")
    if name == "CONFIG_TORCHSIM_DUMP_FILE":
          return int(os.environ.get('TORCHSIM_DUMP_FILE', default=True))
    if name == "CONFIG_TORCHSIM_FUNCTIONAL_MODE":
          return int(os.environ.get('TORCHSIM_FUNCTIONAL_MODE', default=True))
    if name == "CONFIG_TORCHSIM_TIMING_MODE":
          return int(os.environ.get("TORCHSIM_TIMING_MODE", True))
    if name == "CONFIG_CLEANUP_DUMP_ARGS":
          return int(os.environ.get('CLEANUP_DUMP_ARGS', default=False))

    # LLVM PATH
    if name == "CONFIG_TORCHSIM_LLVM_PATH":
        return os.environ.get('TORCHSIM_LLVM_PATH', default="/usr/bin")
    if name == "CONFIG_TORCHSIM_CUSTOM_PASS_PATH":
        return os.environ.get('TORCHSIM_CUSTOM_PASS_PATH',
                                              default=f"{__getattr__('CONFIG_TORCHSIM_DIR')}/GemminiLowerPass/build")
    if name == "CONFIG_TORCHSIM_DUMP_MLIR_IR":
        return int(os.environ.get("TORCHSIM_DUMP_MLIR_IR", default=False))
    if name == "CONFIG_TORCHSIM_DUMP_LLVM_IR":
        return int(os.environ.get("TORCHSIM_DUMP_LLVM_IR", default=False))

    # TOGSim config
    if name == "CONFIG_TOGSIM_CONFIG":
        return os.environ.get('TORCHSIM_CONFIG',
                default=f"{__getattr__('CONFIG_TORCHSIM_DIR')}/TOGSim/configs/systolic_ws_128x128_c1_simple_noc_tpuv3.json")
    if name == "CONFIG_TOGSIM_EAGER_MODE":
        return int(os.environ.get("TOGSIM_EAGER_MODE", default=False))
    if name == "CONFIG_TOGSIM_DRYRUN":
        return int(os.environ.get('TOGSIM_DRYRUN', default=False))
    if name == "CONFIG_TOGSIM_DEBUG_LEVEL":
        return os.environ.get("TOGSIM_DEBUG_LEVEL", "")

    # GEM5 config
    if name == "CONFIG_GEM5_PATH":
        return os.environ.get('GEM5_PATH', default="/workspace/gem5/build/RISCV/gem5.opt")
    if name == "CONFIG_GEM5_SCRIPT_PATH":
        return os.environ.get('GEM5_SCRIPT_PATH',
                                      default=f"{__getattr__('CONFIG_TORCHSIM_DIR')}/gem5_script/script_systolic.py")

    # AUTOTUNE config
    if name == "CONFIG_AUTOTUNE":
        return int(os.environ.get('AUTOTUNE', default=False))
    if name == "CONFIG_AUTOTUNE_TEMPLATE":
        return int(os.environ.get('AUTOTUNE_TEMPLATE', default=False))
    if name == "CONFIG_MAX_AUTOTUNE_TRY":
        return int(os.environ.get('MAX_AUTOTUNE_TRY', default=10))
    if name == "CONFIG_AUTOTUNE_TEMPLATE_TOPK":
        return int(os.environ.get('AUTOTUNE_TEMPLATE_TOPK', default=4))

    # For block sparse
    if name == "CONFIG_BLOCK_SPARSE":
        return int(os.environ.get('BLOCK_SPARSE', default=0))

    # For GEMM tile size
    if name == "CONFIG_MANUAL_TILE_SIZE":
        return int(os.environ.get('TORCHSIM_MANUAL_TILE_SIZE', default=False))
    if name == "CONFIG_TILE_M":
        return int(os.getenv("TORCHSIM_TILE_M", __getattr__("CONFIG_VECTOR_LANE")))
    if name == "CONFIG_TILE_N":
        return int(os.getenv("TORCHSIM_TILE_N", __getattr__("CONFIG_VECTOR_LANE")))
    if name == "CONFIG_TILE_K":
        return int(os.getenv("TORCHSIM_TILE_K", __getattr__("CONFIG_VECTOR_LANE")))

    if name == "CONFIG_SUBTILE":
        return int(os.environ.get('TORCHSIM_SUBTILE', default=True))
    if name == "CONFIG_MANUAL_SUBTILE_SIZE":
        return int(os.environ.get('TORCHSIM_MANUAL_SUBTILE_SIZE', default=False))
    if name == "CONFIG_SUBTILE_M":
        return int(os.environ.get('TORCHSIM_SUBTILE_M', default=__getattr__("CONFIG_VECTOR_LANE")))
    if name == "CONFIG_SUBTILE_N":
        return int(os.environ.get('TORCHSIM_SUBTILE_N', default=__getattr__("CONFIG_VECTOR_LANE")))
    if name == "CONFIG_SUBTILE_K":
        return int(os.environ.get('TORCHSIM_SUBTILE_K', default=__getattr__("CONFIG_VECTOR_LANE")))

    if name == "CONFIG_GEMM_CHEATSHEET_PATH":
          return os.environ.get('TORCHSIM_GEMM_CHEATSHEET_PATH',
                          default=f"{__getattr__('CONFIG_TORCHSIM_DIR')}/validation/gemm_tpuv3_cheatsheet.json")
    # Compiler Optimization
    if name == "CONFIG_COMPILER_OPTIMIZATION":
          return os.environ.get('TORCHSIM_COMPILER_OPTIMIZATION', default="all")  # options: all, none, custom
    # Advanced fusion options
    if name == "CONFIG_FUSION":
          return True if (__getattr__("CONFIG_COMPILER_OPTIMIZATION") == "all" or "fusion" in __getattr__("CONFIG_COMPILER_OPTIMIZATION")) else False
    if name == "CONFIG_FUSION_REDUCTION_EPILOGUE":
          return True if (__getattr__("CONFIG_COMPILER_OPTIMIZATION") == "all" or "reduction_epliogue" in __getattr__("CONFIG_COMPILER_OPTIMIZATION")) else False
    if name == "CONFIG_FUSION_REDUCTION_REDUCTION":
          return True if (__getattr__("CONFIG_COMPILER_OPTIMIZATION") == "all" or "reduction_reduction" in __getattr__("CONFIG_COMPILER_OPTIMIZATION")) else False
    if name == "CONFIG_FUSION_PROLOGUE":
          return True if ((__getattr__("CONFIG_COMPILER_OPTIMIZATION") == "all") or ("prologue" in __getattr__("CONFIG_COMPILER_OPTIMIZATION"))) else False
    if name == "CONFIG_SINGLE_BATCH_CONV":
          return True if (__getattr__("CONFIG_COMPILER_OPTIMIZATION") == "all" or "single_batch_conv" in __getattr__("CONFIG_COMPILER_OPTIMIZATION")) else False
    if name == "CONFIG_MULTI_TILE_CONV":
          return True if (__getattr__("CONFIG_COMPILER_OPTIMIZATION") == "all" or "multi_tile_conv" in __getattr__("CONFIG_COMPILER_OPTIMIZATION")) else False

# SRAM Buffer allocation plan
def load_plan_from_module(module_path):
    if module_path is None:
      return None

    try:
        spec = importlib.util.spec_from_file_location("plan_module", module_path)
        if spec is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if hasattr(module, 'plan'):
            return module.plan
        return None
    except Exception as e:
        print(f"[Warning] Failed to load SRAM buffer plan from module: {e}")
        return None

CONFIG_SRAM_BUFFER_PLAN_PATH = os.environ.get("SRAM_BUFFER_PLAN_PATH", default=None)
CONFIG_SRAM_BUFFER_PLAN = load_plan_from_module(CONFIG_SRAM_BUFFER_PLAN_PATH)

# For ILS experiment
CONFIG_TLS_MODE = int(os.environ.get('TORCHSIM_TLS_MODE', default=1))

CONFIG_USE_TIMING_POOLING = int(os.environ.get('TORCHSIM_USE_TIMING_POOLING', default=0))

CONFIG_DEBUG_MODE = int(os.environ.get('TORCHSIM_DEBUG_MODE', default=0))