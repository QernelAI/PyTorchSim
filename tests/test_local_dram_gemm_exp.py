"""Test GEMM + exp with local DRAM config.

This exercises two separate kernels:
1. GEMM kernel — runs on Q32 core with local DRAM bypass (960ns reads).
   Writes go through NoC to DSP core (core-to-core transfer).
2. Exp kernel — runs as a separate pointwise kernel.

Without subgraph_map routing, both kernels run on core 0 by default.
The GEMM kernel should still show core-to-core NoC transfers
because WRITE_REQUESTs are routed to dsp_core_id in local_dram_mode.
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
    }
    with open(log_path) as f:
        for line in f:
            if 'Total execution cycles' in line:
                stats['total_cycles'] = int(line.split(':')[-1].strip())
            elif 'Core<->Core transfers' in line:
                stats['core_to_core'] = int(line.split('transfers:')[-1].strip())
            elif 'local_mem_latency_cycles' in line:
                # e.g. "[Config/Core] Core 0: local_mem_latency_cycles = 902"
                parts = line.split('Core ')[1]
                core_id = int(parts.split(':')[0])
                latency = int(parts.split('= ')[1].strip())
                stats['core_stats'][core_id] = {'local_mem_latency_cycles': latency}
            elif 'NUMA' in line and 'local' in line:
                # "[Core 0] NUMA local: X, remote: Y"
                pass  # We'll check the raw log
    return stats


# ── Test 1: matmul only (baseline) ──
print("\n" + "=" * 60)
print("Test 1: matmul only (1x512 @ 512x512)")
print("=" * 60)
torch._dynamo.reset()
a = torch.randn(1, 512).to(device=device)
b = torch.randn(512, 512).to(device=device)

before_log = get_latest_log()
fn1 = torch.compile(dynamic=False)(lambda a, b: torch.matmul(a, b))
result1 = fn1(a, b)
print(f"Result shape: {result1.shape}")
time.sleep(0.5)
matmul_log = get_latest_log()

if matmul_log and matmul_log != before_log:
    stats1 = parse_log_stats(matmul_log)
    print(f"  Total cycles: {stats1['total_cycles']}")
    print(f"  Core-to-core transfers: {stats1['core_to_core']}")
    print(f"  Core configs: {stats1['core_stats']}")
    # Print raw ICNT stats
    with open(matmul_log) as f:
        for line in f:
            if 'ICNT' in line or 'NUMA' in line or 'Core<->Core' in line:
                print(f"  LOG: {line.rstrip()}")
else:
    print("  WARNING: No new log file generated")
    stats1 = None


# ── Test 2: exp only ──
print("\n" + "=" * 60)
print("Test 2: exp only (1x512)")
print("=" * 60)
torch._dynamo.reset()
x = torch.randn(1, 512).to(device=device)

before_log = get_latest_log()
fn2 = torch.compile(dynamic=False)(lambda x: torch.exp(x))
result2 = fn2(x)
print(f"Result shape: {result2.shape}")
time.sleep(0.5)
exp_log = get_latest_log()

if exp_log and exp_log != before_log:
    stats2 = parse_log_stats(exp_log)
    print(f"  Total cycles: {stats2['total_cycles']}")
    print(f"  Core-to-core transfers: {stats2['core_to_core']}")
    with open(exp_log) as f:
        for line in f:
            if 'ICNT' in line or 'NUMA' in line or 'Core<->Core' in line:
                print(f"  LOG: {line.rstrip()}")
else:
    print("  WARNING: No new log file generated (exp may not produce a tile graph)")
    stats2 = None


# ── Test 3: matmul then exp (two separate kernels) ──
print("\n" + "=" * 60)
print("Test 3: matmul + exp (1x512 @ 512x512, then exp)")
print("=" * 60)
torch._dynamo.reset()
a = torch.randn(1, 512).to(device=device)
b = torch.randn(512, 512).to(device=device)

before_log = get_latest_log()
fn3 = torch.compile(dynamic=False)(lambda a, b: torch.exp(torch.matmul(a, b)))
result3 = fn3(a, b)
print(f"Result shape: {result3.shape}")
time.sleep(0.5)

# There might be TWO new log files (one per kernel)
all_logs = sorted(glob.glob(os.path.join(LOG_DIR, '*.log')))
new_logs = [l for l in all_logs if before_log is None or l > before_log]
print(f"  New log files generated: {len(new_logs)}")
for i, log_path in enumerate(new_logs):
    stats = parse_log_stats(log_path)
    print(f"\n  --- Kernel {i+1} log: {os.path.basename(log_path)} ---")
    print(f"  Total cycles: {stats['total_cycles']}")
    print(f"  Core-to-core transfers: {stats['core_to_core']}")
    with open(log_path) as f:
        for line in f:
            if 'ICNT' in line or 'NUMA' in line or 'Core<->Core' in line:
                print(f"  LOG: {line.rstrip()}")


print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print("If core-to-core transfers > 0 for GEMM, the NoC routing works.")
print("Exp kernel runs on core 0 by default (no subgraph_map routing to DSP yet).")
print("Next step: add subgraph_map to route exp subgraphs to DSP (core 4).")
