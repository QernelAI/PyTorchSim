"""Test large GEMM distributed across 12 groups of 4 Q32 cores (48 Q32 + 12 DSP).

Runs a 1x24576 @ 4096x24576 GEMM on a 512x512 int4 tensor core config.
Output tiles along N: 24576 / 512 = 48 subgraphs, round-robin across 48 Q32 cores.

Layout (12 groups, 60 cores total):
  Group 0:  Q32 cores [0,1,2,3]     + DSP 4
  Group 1:  Q32 cores [5,6,7,8]     + DSP 9
  Group 2:  Q32 cores [10,11,12,13] + DSP 14
  ...
  Group 11: Q32 cores [55,56,57,58] + DSP 59

Expected with q32_12group_local_dram_dsp.json config (512-lane, int4):
- 48 subgraphs, 1 per Q32 core
- Reads: local DRAM (902 cycles each) -- bypass NoC
- Writes: core-to-core transfers over NoC to LOCAL DSP only

Usage:
    TOGSIM_CONFIG=configs/q32_12group_local_dram_dsp.json python tests/test_q32_12group_gemm.py
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


# ── Config ──
import json
_cfg = json.load(open(os.environ.get('TOGSIM_CONFIG', '')))
_vlanes = _cfg['vpu_num_lanes']
_spad_kb = _cfg['vpu_spad_size_kb_per_lane']
_data_prec = _cfg.get('data_precision_bytes', 4)
_acc_prec = _cfg.get('acc_precision_bytes', _data_prec)

# Compute Q32 core count from groups config
_q32_groups = _cfg.get('q32_groups', [])
if _q32_groups:
    _dsp_set = set(g['dsp_core'] for g in _q32_groups)
    _num_q32 = _cfg['num_cores'] - len(_dsp_set)
else:
    _dsp_core = _cfg.get('dsp_core_id', -1)
    _num_q32 = _dsp_core if _dsp_core > 0 else _cfg['num_cores']

# Size N so each Q32 core gets exactly 1 subgraph
_M, _K = 1, 4096
_N = _num_q32 * _vlanes  # 48 * 512 = 24576
_n_subgraphs = _N // _vlanes
_k_iters = _K // _vlanes

# ── Test: Large GEMM ──
print("\n" + "=" * 70)
print(f"Test: Large GEMM ({_M}x{_K} @ {_K}x{_N}) — {_num_q32}-core round-robin ({len(_q32_groups)} groups)")
print("=" * 70)
print(f"  Tensor core:   {_vlanes} x {_vlanes}, {_data_prec}B data / {_acc_prec}B accumulator")
print(f"  Output tiles:  {_N} / {_vlanes} = {_n_subgraphs} subgraphs")
print(f"  K tiles:       {_K} / {_vlanes} = {_k_iters} iterations per subgraph")
print(f"  Distribution:  round-robin across {_num_q32} Q32 cores ({_n_subgraphs // _num_q32} subgraphs each)")
if _q32_groups:
    for i, g in enumerate(_q32_groups):
        print(f"  Group {i:2d}: Q32 cores {g['q32_cores']} -> DSP {g['dsp_core']}")
print()

torch._dynamo.reset()
a = torch.randn(1, _K).to(device=device)
b = torch.randn(_K, _N).to(device=device)

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

    # Show latencies for a sample of cores
    q32_latencies = {k: v for k, v in stats['core_stats'].items() if not _q32_groups or k not in _dsp_set}
    dsp_latencies = {k: v for k, v in stats['core_stats'].items() if _q32_groups and k in _dsp_set}
    if q32_latencies:
        sample = dict(list(q32_latencies.items())[:4])
        print(f"  Q32 core latencies (sample): {sample}")
    if dsp_latencies:
        sample = dict(list(dsp_latencies.items())[:4])
        print(f"  DSP core latencies (sample): {sample}")

    # Sanity checks
    print(f"\n  --- Sanity checks ---")
    if stats['core_to_core'] > 0:
        print(f"  PASS: Core-to-core transfers = {stats['core_to_core']} (NoC routing active)")
    else:
        print(f"  FAIL: No core-to-core transfers detected")
else:
    print("  WARNING: No new log file generated")
