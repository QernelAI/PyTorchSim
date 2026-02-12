# CLAUDE.md — PyTorchSim: NPU Simulation Framework for Q32 CIM Accelerator

## Project Overview

PyTorchSim is a comprehensive, cycle-accurate NPU simulation framework. It integrates with the PyTorch 2.x compiler stack to simulate DNN inference and training on a configurable RISC-V-based NPU architecture with systolic arrays, vector processing units, and DMA engines.

This simulator is used to model and evaluate the **Q32 CIM (Compute-In-Memory) tensor core accelerator** — a custom AI/ML hardware architecture designed for LLM inference and training.

## Target Hardware: Q32 Architecture

### Board Topology

- **1 board = 10 Q-Tiles**, each Q-Tile contains **48 Q32 tensor cores** + **1 Cadence Tensilica Vision 341 DSP**
- **480 tensor cores per board** total

### Q32 Tensor Cores (CIM)

- **Compute-In-Memory**: weights are loaded into a crossbar/SRAM compute array; MAC streams input activations through the array
- **Max CIM tile size**: 512x512
- **Local DRAM**: 120 MB per core (weight storage + KV-cache + workspace)
- No shared memory between tensor cores
- **WEIGHT_FILL latency**: 960 ns (load 512x512 tile from local DRAM to CIM array)
- **MAC latency**: 40 ns (stream 1x512 input through CIM, produce 1x512 output)
- **Fill:MAC ratio**: 24:1 — weight fill dominates; reuse weights by streaming multiple inputs
- **Input/weight formats**: INT8, INT4, MXFP4, NVFP4
- **Output**: FP32
- **Command FIFO**: 32-deep per tensor core, controlled via MMIO by the DSP

### Q32 Firmware Opcodes

| Opcode | Name | Description |
|--------|------|-------------|
| 0x00 | FW_Q32_WEIGHT_FILL | Load weights into CIM array |
| 0x01 | FW_Q32_MAC | Matrix-multiply (stream input through CIM) |
| 0x02 | FW_Q32_FENCE | Wait for completion (barrier) |
| 0x03 | FW_Q32_RESET | Reset core |
| 0x04 | FW_Q32_NOP | No-op |

### Vision 341 DSP (one per Q-Tile)

- 1024-bit SIMD, ~4 TOPS, 2048-bit memory bus
- Handles: softmax, layernorm/RMSNorm, RoPE, SiLU/GeLU, residual adds, KV-cache quantization/dequantization
- Can directly access all 48 children tensor cores' local DRAM banks (fast intra-Q-Tile reduction)
- NatureDSP library for optimized kernels

### Memory Hierarchy

- **Per-core local DRAM**: 120 MB (weights + KV-cache + workspace)
- **Intra-Q-Tile**: DSP has direct access to all 48 cores' DRAM
- **Cross-Q-Tile**: Network-on-Chip (NoC) for inter-tile communication

### Primary Targets

- LLM decode (autoregressive M=1 GEMVs, memory-bandwidth-bound)
- Initial models: Llama 2 7B, Llama 3 8B
- Supports inference and training

## How PyTorchSim Simulates This

### Two Main Components

1. **Compiler** — Integrated with PyTorch 2.x compiler stack (FX -> MLIR -> LLVM -> RISC-V ISA). Generates NPU machine code and Tile-Operation Graphs (TOG).
2. **TOGSim** — C++ cycle-accurate simulator. Executes TOGs with integrated BookSim2 (NoC) and Ramulator2 (HBM2/DDR) simulators for shared resources.

### Simulation Pipeline

```
PyTorch Model
  -> torch.compile() with custom NPU device
  -> Compiler generates TOG + machine code
  -> Gem5: obtains compute latency for each tile operation
  -> Spike: functional verification (optional, disable with pytorchsim_functional_mode=False)
  -> TOGSim: cycle-accurate NPU architecture simulation
  -> Results: per-core utilization, memory bandwidth, instruction counts, total cycles
```

### TOGSim Instruction Set (4 opcodes)

| Opcode | Description |
|--------|-------------|
| MOVIN | DMA read (DRAM -> local scratchpad) |
| MOVOUT | DMA write (local scratchpad -> DRAM) |
| COMP | Compute operation (GEMM on systolic array, or vector op on VPU) |
| BAR | Synchronization barrier |

### Core Architecture (per core in simulation)

- **Systolic Arrays (SA)**: 1-2 per core (configurable), weight-stationary for GEMM
- **Vector Processing Unit (VPU)**: configurable lanes (default 128), each with scratchpad memory (default 128 KB/lane)
- **DMA Engine**: manages data movement with tag-based synchronization, supports indirect addressing and async operations
- **Interconnect ports**: configurable injection ports per core for NoC traffic

### Supported Core Types

- `WS_MESH` — Weight-stationary systolic array (primary)
- `STONNE` — Sparse tensor operation support

## Directory Structure

```
PyTorchSim/
├── PyTorchSimFrontend/     # PyTorch compiler backend integration
│   ├── extension_config.py # Hardware config (reads from JSON + env vars)
│   ├── extension_op.py     # Custom op definitions
│   ├── extension_device.cpp # Custom NPU device registration
│   └── mlir/               # MLIR codegen templates (GEMM, Conv, BMM, etc.)
├── TOGSim/                 # C++ cycle-accurate simulator
│   ├── include/            # Headers: Core.h, DMA.h, Instruction.h, Tile.h, TileGraph.h, SimulationConfig.h, etc.
│   ├── src/                # Implementation: Simulator.cc, Core.cc, etc.
│   └── extern/             # External simulators (Ramulator, BookSim2)
├── Simulator/
│   └── simulator.py        # Python simulator interface (invokes TOGSim)
├── Scheduler/
│   └── scheduler.py        # Multi-tenancy scheduler, PyTorchSimRunner
├── AsmParser/              # ONNX utility and TOG generator
├── configs/                # JSON config files for different NPU configurations
├── tests/                  # Test workloads (matmul, conv, transformer, ResNet, BERT, GPT-2, etc.)
├── experiments/            # Artifact evaluation scripts
├── docs/                   # Architecture diagrams
├── tpuv4/                  # L2 cache (CMEM) plan examples from TPUv4 profiling
└── scripts/                # Build scripts
```

## Configuration

### TOGSim JSON Config (configs/*.json)

Key parameters:

```json
{
  "num_cores": 1,                        // Number of NPU cores
  "core_freq_mhz": 940,                  // Core frequency (MHz)
  "num_systolic_array_per_core": 2,       // Systolic arrays per core

  "vpu_num_lanes": 128,                  // VPU lanes
  "vpu_spad_size_kb_per_lane": 128,       // Scratchpad per lane (KB)
  "vpu_vector_length_bits": 256,          // VPU vector register width (bits)

  "dram_type": "ramulator2",              // DRAM model: "simple" or "ramulator2"
  "dram_freq_mhz": 940,                  // DRAM frequency
  "dram_channels": 16,                   // HBM2 channels
  "dram_req_size_byte": 32,              // Request granularity (bytes)

  "icnt_type": "simple",                 // Interconnect: "simple" or "booksim2"
  "icnt_latency_cycles": 10,             // Interconnect latency
  "icnt_injection_ports_per_core": 16,    // NoC ports per core

  "l2d_type": "nocache",                 // L2 cache: "nocache" or "datacache"

  "codegen_mapping_strategy": "heuristic", // "heuristic", "autotune", "external-then-heuristic", "external-then-autotune"
  "codegen_compiler_optimization": "all"   // "all", "none", or list of specific opts
}
```

### Available Config Presets

- `systolic_ws_128x128_c1_simple_noc_tpuv3.json` — Single-core, TPUv3-like, simple NoC (default)
- `systolic_ws_128x128_c2_*` — Dual-core variants
- `*_booksim_*` — Full BookSim2 NoC simulation
- `*_tpuv4.json` — TPUv4-like configs with L2 cache support
- `*_chiplet_*` — Chiplet configs with NUMA
- `stonne_*` — Sparse tensor core configs
- `q32_cim_dsp.json` — 2-core Q32 CIM (1 tensor core + 1 DSP), simple NoC
- `q32_4core_dsp.json` — 5-core Q32 CIM (4 tensor cores + 1 DSP), BookSim2 NoC

### Environment Variables

```bash
export TORCHSIM_DIR=/workspace/PyTorchSim          # Home directory
export TOGSIM_CONFIG=path/to/config.json            # TOGSim config file
export TORCHSIM_DUMP_PATH=/tmp/torchinductor        # Output dump path
export TORCHSIM_TLS_MODE=1                          # 1=TLS (Tile-Level Sim), 0=ILS (Instruction-Level Sim)
export pytorchsim_functional_mode=False             # Disable Spike for faster sim
export SRAM_BUFFER_PLAN_PATH=tpuv4/gemm_plan.py     # L2 cache allocation plan
export TORCHSIM_DUMP_MLIR_IR=1                      # Dump MLIR IR (debug)
export TORCHSIM_DUMP_LLVM_IR=1                      # Dump LLVM IR (debug)
```

## Commands

```bash
# Run with Docker (recommended)
docker run -it --ipc=host --name torchsim -w /workspace/PyTorchSim ghcr.io/psal-postech/torchsim-ci:v1.0.1 bash

# Run a basic test
python tests/test_matmul.py

# Run multi-tenancy test
python tests/test_scheduler.py

# Run Q32 multi-core tests (set TOGSIM_CONFIG first)
export TOGSIM_CONFIG=configs/q32_4core_dsp.json
python tests/test_q32_multicore_sweep.py   # Single matmul, size sweep
python tests/test_q32_multi_gemm.py        # 4 concurrent matmuls on 4 cores

# Analyze simulation logs
python scripts/analyze_multicore_log.py togsim_results/*.log

# Build from source (optional)
bash scripts/build_from_source.sh
```

## Compiler Optimizations

- GEMM prologue fusion
- GEMM epilogue fusion
- GEMM reduction fusion
- CONV epilogue fusion
- Single-batch convolution optimization
- Multi-channel (multi-tile) convolution optimization
- Subtile optimization

## Mapping Strategies

- **Heuristic** (default): GEMMINI-inspired, maximizes scratchpad utilization
- **Auto-tune**: searches tile shape and vector lane stride candidates (top-k)
- **External**: user-supplied mapping files (e.g., from Timeloop)

## Supported Models

ResNet-18/50, BERT, GPT-2, ViT, Mistral, Diffusion models. Llama-4 and DeepSeek v1 under development.

## Key Simulation Stats (output log)

- Per-channel HBM2 bandwidth utilization, row hits/misses/conflicts
- Per-core instruction counts by type (MOVIN, MOVOUT, COMP with GEMM/Vector breakdown, BAR)
- Per-core systolic array utilization (active/idle cycles)
- Per-core DMA active/idle cycles and DRAM bandwidth
- Per-core vector unit utilization
- NUMA local vs remote memory access counts
- Total execution cycles and wall-clock simulation time

## Related Qernel Repositories

- `q32-compiler/` — MLIR-based compiler targeting Q32 CIM tensor cores
- `q32-rt/` — Q32 runtime driver (hardware abstraction layer with pluggable backends)
- `qcompiler/` — Advanced MLIR compiler for full LLM compilation to per-core firmware
- `performance_model_q32/` — Analytical performance model (latency, power, energy estimation)
- `aie-rt/` — AMD AIE runtime
- `driver/` — Low-level hardware driver

## Q32 CIM + DSP Simulation (2-Core Model)

### What Was Implemented

A 2-core simulation mapping one Q32 CIM tensor core (core 0) + one Vision 341 DSP (core 1) onto PyTorchSim's TOGSim. The goal is cycle-accurate latency and bandwidth estimation for a single Q32+DSP pair.

**Architecture mapping**:
- Core 0 (Q32): SA pipeline for GEMM (MAC ops). Weight MOVIN from partition 0 (local DRAM).
- Core 1 (DSP): VU pipeline for vector ops (softmax, layernorm, etc.). MOVIN from partition 1 (shared SRAM).
- Partition 0 = local DRAM (high latency, 83 cycles). Partition 1 = shared SRAM (low latency, 10 cycles).

**Files changed/created**:
- `configs/q32_cim_dsp.json` — 2-core config with calibrated DRAM partitions
- `TOGSim/include/SimulationConfig.h` — `dram_latency_per_partition` field
- `TOGSim/include/Dram.h` — `_latency_per_partition` in SimpleDRAM
- `TOGSim/src/Dram.cc` — per-partition latency lookup in `SimpleDRAM::cycle()`
- `TOGSim/src/Common.cc` — JSON parsing for `dram_latency_per_partition`
- `Simulator/simulator.py` — `Q32CycleModel` class (analytical MAC/vector cycles)
- `PyTorchSimFrontend/extension_config.py` — `q32_cim_mode` / `q32_mac_ns` / `q32_cim_tile_dim` accessors
- `PyTorchSimFrontend/extension_codecache.py` — Q32 branch skipping Gem5, using analytical cycles + attribute gen
- `scripts/q32_attribute_gen.py` — subgraph-to-core mapping (GEMM→core 0, vector→core 1)

### How Latencies Are Currently Modeled

**MAC latency (40 ns)**:
- Modeled via `Q32CycleModel.gemm_cycles()` in `Simulator/simulator.py`.
- Each GEMM compute node in the TOG gets `gemm_cycles(1, 512, 512)` = 1 MAC pass = 40 ns ≈ 37 cycles at 940 MHz.
- The TOG loop structure handles tiling repetition (iterating over M/N/K dimensions).

**Weight fill latency (960 ns)**:
- **NOT explicitly modeled as a CIM operation.** Currently approximated by the MOVIN instruction going through SimpleDRAM on partition 0.
- SimpleDRAM config calibrated so a 256 KB transfer (512x512 INT8 tile) takes ~960 ns:
  - 10 total channels, 5 per partition, 64 B/request → 4096 requests → ~820/channel
  - `dram_latency_per_partition[0] = 83` → 820 + 83 - 1 = 902 cycles → 959.6 ns
- This is an approximation: real weight fill is a distinct CIM opcode (`FW_Q32_WEIGHT_FILL`) that loads data from local DRAM into the crossbar array, not a generic memory read.

**SRAM (partition 1, activation transfer between Q32 and DSP)**:
- `dram_latency_per_partition[1] = 10` → ~10.6 ns per request. Reasonable for SRAM.

### Known Limitations / Future Work

1. **Weight fill should be modeled as explicit COMP, not DRAM approximation** (Option B from planning):
   - MOVIN from partition 0 should model the DRAM read to a staging buffer.
   - A separate COMP phase should model the CIM crossbar load (fixed 960 ns = 902 cycles).
   - Then MAC COMP models streaming inputs (40 ns per row).
   - This decouples DRAM bandwidth from CIM fill timing and is more architecturally honest.

2. **MAC cycle assignment per compute node**: `gemm_cycles(1, cim_dim, cim_dim)` assumes each TOG compute node = one MAC pass. This depends on how the MLIR pass tiles the GEMM. Need to verify against actual TOG structure for different workloads.

3. **No command FIFO modeling**: The 32-deep per-core command FIFO is not simulated. In real hardware, FIFO depth limits how far ahead the DSP can queue commands.

4. **No data format modeling**: INT8/INT4/MXFP4/NVFP4 formats are not distinguished. The 256 KB tile size assumes INT8 (512x512x1 byte). Other formats would change the weight fill data volume.

5. **DRAM calibration is approximate**: The SimpleDRAM pipeline model (`requests_per_channel + latency - 1`) doesn't account for interconnect overhead, DMA request generation timing, or cache pass-through delays. The 83-cycle latency was chosen analytically, not validated against actual simulation output.

6. **Multi-core scaling path**: When expanding to full Q-Tile (48 Q32 + 1 DSP), need to increase partitions, switch to `booksim2` NoC, and round-robin GEMM subgraphs across 48 cores.

## Q32 CIM Multi-Core Simulation (4 Q32 + 1 DSP)

### What Was Implemented

Scaled from 2-core to 5-core: 4 Q32 tensor cores (cores 0-3) + 1 Vision 341 DSP (core 4) with cycle-accurate BookSim2 flattened butterfly NoC. Demonstrates concurrent GEMM execution across multiple cores with NoC contention modeling.

**Architecture**:
```
Core 0 (Q32)  Core 1 (Q32)  Core 2 (Q32)  Core 3 (Q32)  Core 4 (DSP)
Partition 0    Partition 1    Partition 2    Partition 3    Partition 4
(local DRAM)   (local DRAM)   (local DRAM)   (local DRAM)   (shared SRAM)
  83 cycles      83 cycles      83 cycles      83 cycles      10 cycles
  4 channels     4 channels     4 channels     4 channels     4 channels
                    BookSim2 Flattened Butterfly NoC (k=40)
```

**Files created/modified**:
- `configs/q32_4core_dsp.json` — 5-core config: 20 DRAM channels, 5 partitions, BookSim2 NoC
- `configs/booksim2_configs/fly_c20_m20.icnt` — Flattened butterfly, k=40 (5 cores × 4 ports + 20 channels)
- `scripts/q32_attribute_gen.py` — Round-robin GEMM subgraphs across N Q32 cores (was: hardcoded core 0)
- `PyTorchSimFrontend/extension_config.py` — `q32_num_cores` accessor (reads from JSON, default 1)
- `PyTorchSimFrontend/extension_codecache.py` — Passes core count to attribute gen
- `TOGSim/src/Simulator.cc` — Fixed NUMA tracking to use `get_partition_id(core_id)` instead of raw `core_id`

### Multi-Core Test Scripts

**`tests/test_q32_multicore_sweep.py`**: Runs matmul at various sizes through the standard `torch.compile()` path with the 4-core config. Single matmul → single subgraph → lands on core 0 only (round-robin with 1 GEMM = core 0).

**`tests/test_q32_multi_gemm.py`**: Exercises all 4 cores concurrently by:
1. Compiling a matmul through standard path to generate the TOG
2. Launching the same TOG to 4 separate partitions via interactive TOGSim (`launch config tog attr time partition_id`)
3. Using an empty attribute JSON (no `subgraph_map` → `core_id=-1` → any core in the partition picks up the subgraph)
4. Comparing single-core baseline vs 4-core concurrent execution

**`scripts/analyze_multicore_log.py`**: Post-hoc analysis of TOGSim logs. Extracts:
- Per-core instruction counts (MOVIN, MOVOUT, COMP with GEMM/Vector breakdown, BAR)
- Per-core DMA stats (active/idle cycles, utilization, DRAM bandwidth, request counts)
- Per-core SA and VPU utilization
- NUMA local vs remote memory access pattern
- BookSim2 NoC stats (packet/network/flit latency, injection/accepted rates, hotspot detection)
- Scheduler timing (per-partition finish cycles, spread)
- ICNT bandwidth intervals

Usage: `python scripts/analyze_multicore_log.py togsim_results/*.log`

### Key Architecture Insights (TOGSim Interactive Mode)

- **Partition-based scheduling**: `schedule_graph(partition_id, tile_graph)` assigns a TOG to a specific partition's scheduler. Each core polls only its own partition.
- **Subgraph allocation**: `TileGraph::allocate_subgraph(core_id)` checks `subgraph.core_id == -1 || subgraph.core_id == core_id`. Empty attribute → `core_id=-1` → any core can pick up.
- **Interactive commands**: `launch config_path onnx_path attr_path arrival_time partition_id` via stdin, responses on stderr. `quit` triggers `cycle()` then `print_core_stat()`.
- **BookSim2 node count formula**: `(num_cores × icnt_injection_ports_per_core) + dram_channels = k` for fly topology.
- **Partition map typo**: `_config.partiton_map` (not "partition") throughout C++ codebase.

### Known Multi-Core Limitations

1. **NUMA asymmetry with identical TOGs**: When launching the same TOG to multiple partitions, all cores use identical virtual addresses → all DRAM requests hash to partition 0's channels. Cores 1-3 show 100% remote accesses. In production, each core would have distinct weight addresses in its own local DRAM.
2. **NoC hotspot at DRAM channels**: Node 20 (first DRAM channel node) receives ~5x average injection rate because all cores target the same address space.
3. **SimpleDRAM limitations**: No per-channel bandwidth stats or row hit/miss/conflict ratios. Switch to `dram_type: "ramulator2"` for detailed DRAM analytics.
4. **ICNT bandwidth reporting**: Periodic stats use integer division that truncates to 0 — actual NoC traffic is confirmed by BookSim2 stats.
5. **Single matmul = single core**: `torch.compile` of one matmul produces one GEMM subgraph, which lands on core 0 (0 % 4 = 0). Multi-core requires multiple independent TOGs launched to different partitions.

## Codebase Conventions

- **Languages**: C++ (TOGSim core), Python (compiler frontend, scheduler, tests)
- **Build**: CMake for TOGSim C++ code, Conan for C++ dependencies
- **External dependencies**: spdlog, fmt, robin_hood (hashing), nlohmann/json, BookSim2, Ramulator2
- **Testing**: Python test scripts in `tests/` directory
- **Config-driven**: all hardware parameters are in JSON configs, not hardcoded
