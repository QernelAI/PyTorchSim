import os
import sys
import importlib
import json

CONFIG_TORCHSIM_DIR = os.environ.get('TORCHSIM_DIR', default='/workspace/PyTorchSim')
CONFIG_GEM5_PATH = os.environ.get('GEM5_PATH', default="/workspace/gem5/build/RISCV/gem5.opt")
CONFIG_TORCHSIM_LLVM_PATH = os.environ.get('TORCHSIM_LLVM_PATH', default="/usr/bin")

CONFIG_TORCHSIM_DUMP_MLIR_IR = int(os.environ.get("TORCHSIM_DUMP_MLIR_IR", default=False))
CONFIG_TORCHSIM_DUMP_LLVM_IR = int(os.environ.get("TORCHSIM_DUMP_LLVM_IR", default=False))

def __getattr__(name):
    # TOGSim config
    config_path = os.environ.get('TOGSIM_CONFIG',
                default=f"{CONFIG_TORCHSIM_DIR}/configs/systolic_ws_128x128_c1_simple_noc_tpuv3.json")
    if name == "CONFIG_TOGSIM_CONFIG":
        return config_path
    config_json = json.load(open(config_path, 'r'))

    # Hardware info config
    if name == "vpu_num_lanes":
        return config_json["vpu_num_lanes"]
    if name == "CONFIG_SPAD_INFO":
        return {
          "spad_vaddr" : 0xD0000000,
          "spad_paddr" : 0x2000000000,
          "spad_size" : config_json["vpu_spad_size_kb_per_lane"] << 10 # Note: spad size per lane
        }

    if name == "CONFIG_PRECISION":
        return 4 # 32bit
    if name == "CONFIG_NUM_CORES":
        return config_json["num_cores"]
    if name == "vpu_vector_length_bits":
        return config_json["vpu_vector_length_bits"]

    if name == "pytorchsim_functional_mode":
        return config_json['pytorchsim_functional_mode']
    if name == "pytorchsim_timing_mode":
        return config_json['pytorchsim_timing_mode']

    # Mapping strategy
    if name == "codegen_mapping_strategy":
        codegen_mapping_strategy = config_json["codegen_mapping_strategy"]
        assert(codegen_mapping_strategy in ["heuristic", "autotune", "external-then-heuristic", "external-then-autotune"]), "Invalid mapping strategy!"
        return codegen_mapping_strategy

    if name == "codegen_external_mapping_file":
        return config_json["codegen_external_mapping_file"]

    # Autotune config
    if name == "codegen_autotune_max_retry":
        return config_json["codegen_autotune_max_retry"]
    if name == "codegen_autotune_template_topk":
        return config_json["codegen_autotune_template_topk"]

    # Compiler Optimization
    if name == "codegen_compiler_optimization":
        return config_json["codegen_compiler_optimization"]

    # Advanced fusion options
    if name == "CONFIG_FUSION":
        return True if (__getattr__("codegen_compiler_optimization") == "all" or "fusion" in __getattr__("codegen_compiler_optimization")) else False
    if name == "CONFIG_FUSION_REDUCTION_EPILOGUE":
        return True if (__getattr__("codegen_compiler_optimization") == "all" or "reduction_epliogue" in __getattr__("codegen_compiler_optimization")) else False
    if name == "CONFIG_FUSION_REDUCTION_REDUCTION":
        return True if (__getattr__("codegen_compiler_optimization") == "all" or "reduction_reduction" in __getattr__("codegen_compiler_optimization")) else False
    if name == "CONFIG_FUSION_PROLOGUE":
        return True if ((__getattr__("codegen_compiler_optimization") == "all") or ("prologue" in __getattr__("codegen_compiler_optimization"))) else False
    if name == "CONFIG_SINGLE_BATCH_CONV":
        return True if (__getattr__("codegen_compiler_optimization") == "all" or "single_batch_conv" in __getattr__("codegen_compiler_optimization")) else False
    if name == "CONFIG_MULTI_TILE_CONV":
        return True if (__getattr__("codegen_compiler_optimization") == "all" or "multi_tile_conv" in __getattr__("codegen_compiler_optimization")) else False
    if name == "CONFIG_SUBTILE":
        return True if (__getattr__("codegen_compiler_optimization") == "all" or "subtile" in __getattr__("codegen_compiler_optimization")) else False

    if name == "CONFIG_TOGSIM_DEBUG_LEVEL":
        return os.environ.get("TOGSIM_DEBUG_LEVEL", "")
    if name == "CONFIG_TORCHSIM_DUMP_PATH":
        return os.environ.get('TORCHSIM_DUMP_PATH', default = CONFIG_TORCHSIM_DIR)
    if name == "CONFIG_TORCHSIM_LOG_PATH":
        return os.environ.get('TORCHSIM_DUMP_LOG_PATH', default = os.path.join(CONFIG_TORCHSIM_DIR, "togsim_results"))

    if name == "CONFIG_TOGSIM_EAGER_MODE":
        return int(os.environ.get("TOGSIM_EAGER_MODE", default=False))

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