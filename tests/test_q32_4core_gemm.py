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
import json
_cfg = json.load(open(os.environ.get('TOGSIM_CONFIG', '')))
_vlanes = _cfg['vpu_num_lanes']
_spad_kb = _cfg['vpu_spad_size_kb_per_lane']
_data_prec = _cfg.get('data_precision_bytes', 4)
_acc_prec = _cfg.get('acc_precision_bytes', _data_prec)
_dsp_core = _cfg.get('dsp_core_id', -1)
_num_q32 = _dsp_core if _dsp_core > 0 else _cfg['num_cores']
_M, _N, _K = 1, 4096, 4096
_n_subgraphs = _N // _vlanes
_k_iters = _K // _vlanes

print("\n" + "=" * 70)
print(f"Test: Large GEMM ({_M}x{_K} @ {_K}x{_N}) — {_num_q32}-core round-robin")
print("=" * 70)
print(f"  Tensor core:   {_vlanes} x {_vlanes}, {_data_prec}B data / {_acc_prec}B accumulator")
print(f"  vectorlane:    {_vlanes}")
print(f"  Output tiles:  {_N} / {_vlanes} = {_n_subgraphs} subgraphs")
print(f"  K tiles:       {_K} / {_vlanes} = {_k_iters} iterations per subgraph")
print(f"  Distribution:  round-robin across {_num_q32} Q32 cores ({_n_subgraphs // _num_q32} subgraphs each)")
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

_tile_M, _tile_N, _tile_K = 8, _vlanes, _vlanes  # padded M=8, tiles = vector_lane
_total_spad_kb = _spad_kb * _vlanes
_double_buf_kb = _total_spad_kb / 2
_x_kb = _tile_M * _tile_K * _data_prec / 1024
_w_kb = _tile_K * _tile_N * _data_prec / 1024
_y_kb = _tile_M * _tile_N * _acc_prec / 1024
_total_kb = _x_kb + _w_kb + _y_kb

print("\n" + "=" * 70)
print("Breakdown")
print("=" * 70)
print(f"""
Tensor core: {_vlanes} x {_vlanes}, {_data_prec}B data / {_acc_prec}B accumulator

SRAM (scratchpad) sizing:
  - vpu_spad_size_kb_per_lane = {_spad_kb} KB per lane
  - vpu_num_lanes = {_vlanes}
  - Total SPAD per core: {_spad_kb} KB x {_vlanes} lanes = {_total_spad_kb:.0f} KB
  - Double buffer half: {_double_buf_kb:.0f} KB

  Per K-iteration, the SRAM holds (tile {_tile_M}x{_tile_N}x{_tile_K}):
    Weight tile (W_buffer): {_tile_K} x {_tile_N} x {_data_prec}B = {_w_kb:.1f} KB
    Input tile  (X_buffer): {_tile_M} x {_tile_K} x {_data_prec}B = {_x_kb:.1f} KB
    Output acc  (Y_buffer): {_tile_M} x {_tile_N} x {_acc_prec}B  = {_y_kb:.1f} KB
    Total one set:          {_total_kb:.1f} KB / {_double_buf_kb:.0f} KB ({100*_total_kb/_double_buf_kb:.0f}%)

Memory access:
  - Reads bypass NoC (local DRAM, 902 cycles per read)
  - Only writes (output tiles) go over NoC as core-to-core transfers

K-reduction loop (inner):
  for k in 0..{_K} step {_tile_K}:
    DMA weight B[k:k+{_tile_K}, n:n+{_tile_N}] -> W_buffer  (from local DRAM, 902 cyc)
    DMA input  A[0, k:k+{_tile_K}]             -> X_buffer  (from local DRAM, 902 cyc)
    matmul X_buffer @ W_buffer                  -> Y_buffer  (accumulate in SRAM)
  DMA Y_buffer -> output via NoC to DSP core
""")
