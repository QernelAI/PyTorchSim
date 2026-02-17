"""Test large GEMM distributed across 4 Q32 cores via round-robin.

Runs a 1x4096 @ 4096x4096 GEMM on a 512x512 int4 tensor core config.
Output tiles along N: 4096 / 512 = 8 subgraphs, round-robin across 4 Q32 cores.

Expected with q32_local_dram_dsp.json config (512-lane, int4):
- 8 subgraphs, 2 per Q32 core
- Reads: local DRAM (902 cycles each) — bypass NoC
- Writes: core-to-core transfers over NoC to DSP (core 4)
- SRAM: 1 KB/lane x 512 lanes = 512 KB total (double-buffers one 128 KB weight tile)

Usage:
    TOGSIM_CONFIG=configs/q32_local_dram_dsp.json python tests/test_q32_4core_gemm.py
"""
import os
import sys
import glob
import time

sys.path.append(os.environ.get('TORCHSIM_DIR', default='/workspace/PyTorchSim'))
os.environ.setdefault('TORCHSIM_DIR', '/workspace/PyTorchSim')

import torch
import torch._dynamo

from Scheduler.scheduler import PyTorchSimRunner
module = PyTorchSimRunner.setup_device()
device = module.custom_device()

LOG_DIR = os.path.join(os.environ.get('TORCHSIM_DIR', '/workspace/PyTorchSim'), 'togsim_results')

config_path = os.environ.get('TOGSIM_CONFIG', '(not set, using default)')
print(f"TOGSIM_CONFIG = {config_path}")

def get_latest_log():
    logs = sorted(glob.glob(os.path.join(LOG_DIR, '*.log')))
    return logs[-1] if logs else None

def parse_log_stats(log_path):
    """Extract key stats from a TOGSim log."""
    stats = {
        'total_cycles': None,
        'core_to_core': 0,
        'core_stats': {},
        'icnt_lines': [],
    }
    with open(log_path) as f:
        for line in f:
            if 'Total execution cycles' in line:
                stats['total_cycles'] = int(line.split(':')[-1].strip())
            elif 'Core<->Core transfers' in line:
                stats['core_to_core'] = int(line.split('transfers:')[-1].strip())
            elif 'local_mem_latency_cycles' in line:
                parts = line.split('Core ')[1]
                core_id = int(parts.split(':')[0])
                latency = int(parts.split('= ')[1].strip())
                stats['core_stats'][core_id] = {'local_mem_latency_cycles': latency}
            elif any(kw in line for kw in ['ICNT', 'NUMA', 'Core<->Core', 'Core [']):
                stats['icnt_lines'].append(line.rstrip())
    return stats


# ── Test: Large GEMM (1x4096 @ 4096x4096) across 4 Q32 cores ──
print("\n" + "=" * 70)
print("Test: Large GEMM (1x4096 @ 4096x4096) — 4-core round-robin")
print("=" * 70)
print("  Tensor core:  512 x 512 int4")
print("  vectorlane:   512")
print("  Output tiles:  4096 / 512 = 8 subgraphs")
print("  K tiles:       4096 / 512 = 8 iterations per subgraph")
print("  Distribution:  round-robin across 4 Q32 cores (2 subgraphs each)")
print()
print("  SRAM per lane: 1 KB")
print("  Total SRAM:    512 KB (512 lanes x 1 KB)")
print("  Weight tile:   512x512 int4 = 128 KB")
print("  Double buffer: 256 KB for 2 weight tiles — fits in 512 KB SRAM")
print()

torch._dynamo.reset()
a = torch.randn(1, 4096).to(device=device)
b = torch.randn(4096, 4096).to(device=device)

before_log = get_latest_log()
fn = torch.compile(dynamic=False)(lambda a, b: torch.matmul(a, b))
result = fn(a, b)
print(f"Result shape: {result.shape}")
time.sleep(0.5)
log_path = get_latest_log()

if log_path and log_path != before_log:
    stats = parse_log_stats(log_path)
    print(f"\n  Total cycles:          {stats['total_cycles']}")
    print(f"  Core-to-core packets:  {stats['core_to_core']}")
    print(f"  Core configs:          {stats['core_stats']}")
    print(f"\n  --- Relevant log lines ---")
    for line in stats['icnt_lines']:
        print(f"  {line}")

    # Sanity checks
    print(f"\n  --- Sanity checks ---")
    if stats['core_to_core'] > 0:
        print(f"  PASS: Core-to-core transfers = {stats['core_to_core']} (NoC routing active)")
    else:
        print(f"  FAIL: No core-to-core transfers detected")
else:
    print("  WARNING: No new log file generated")

print("\n" + "=" * 70)
print("Breakdown")
print("=" * 70)
print("""
Tensor core: 512 x 512, int4 (0.5 bytes per element)
Precision: data_precision_bytes = 0.5

SRAM (scratchpad) sizing:
  - vpu_spad_size_kb_per_lane = 1 KB per lane
  - vpu_num_lanes = 512
  - Total SPAD per core: 1 KB x 512 lanes = 512 KB
  - Double buffer half: 256 KB

  Per K-iteration, the SRAM holds:
    Weight tile (W_buffer): 512 x 512 x 0.5B = 128 KB
    Input tile  (X_buffer): 1 x 512 x 0.5B   = 256 B
    Output acc  (Y_buffer): 1 x 512 x 0.5B   = 256 B  (accum may be wider)
    ─────────────────────────────────────────────────
    Total one set:                            ~128.5 KB
    Fits in 256 KB half → other 256 KB prefetches next weight tile

Memory access:
  - Reads bypass NoC (local DRAM, 902 cycles per read)
  - Only writes (output tiles) go over NoC as core-to-core transfers

K-reduction loop (inner):
  for k in 0..4096 step 512:
    DMA weight B[k:k+512, n:n+512] -> W_buffer  (from local DRAM, 902 cyc)
    DMA input  A[0, k:k+512]       -> X_buffer  (from local DRAM, 902 cyc)
    matmul X_buffer @ W_buffer      -> Y_buffer  (accumulate in SRAM)
  DMA Y_buffer -> output via NoC to DSP core 4
""")
