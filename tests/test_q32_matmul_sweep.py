"""Run matmul at various sizes on Q32 CIM config (timing only)."""
import os
import sys
import glob
import time

sys.path.append(os.environ.get('TORCHSIM_DIR', default='/workspace/PyTorchSim'))

import torch
import torch._dynamo

from Scheduler.scheduler import PyTorchSimRunner
module = PyTorchSimRunner.setup_device()
device = module.custom_device()

LOG_DIR = os.path.join(os.environ.get('TORCHSIM_DIR', '/workspace/PyTorchSim'), 'togsim_results')

def get_latest_log():
    logs = sorted(glob.glob(os.path.join(LOG_DIR, '*.log')))
    return logs[-1] if logs else None

def run_matmul(M, N, K):
    """Run a single matmul and return the log path."""
    def custom_matmul(a, b):
        return torch.matmul(a, b)

    torch.manual_seed(0)
    # Reset dynamo to force recompilation for each size
    torch._dynamo.reset()

    x = torch.randn(M, K).to(device=device)
    w = torch.randn(K, N).to(device=device)
    opt_fn = torch.compile(dynamic=False)(custom_matmul)

    before_log = get_latest_log()
    try:
        _ = opt_fn(x, w)
    except Exception:
        pass  # Ignore correctness failures
    time.sleep(0.5)
    after_log = get_latest_log()

    if after_log != before_log:
        return after_log
    return None

def parse_log(log_path):
    """Extract key stats from a TOGSim log."""
    stats = {}
    with open(log_path) as f:
        for line in f:
            if 'Total execution cycles' in line:
                stats['total_cycles'] = int(line.split(':')[-1].strip())
            elif 'Core [0]' in line and 'Total_cycles' in line:
                stats['core0_cycles'] = int(line.split('Total_cycles')[-1].strip())
            elif 'Core [1]' in line and 'Total_cycles' in line:
                stats['core1_cycles'] = int(line.split('Total_cycles')[-1].strip())
            elif 'Core [0]' in line and 'Systolic array [0] utilization' in line:
                stats['core0_sa_util'] = line.split('utilization(%)')[1].split(',')[0].strip()
            elif 'Core [0]' in line and 'DRAM BW' in line:
                bw = line.split('DRAM BW')[1].split('GB/s')[0].strip()
                stats['core0_dram_bw'] = bw
                reqs = line.split('(')[1].split('responses')[0].strip()
                stats['core0_dram_reqs'] = reqs
            elif 'Core [0]' in line and 'NUMA local' in line:
                local = line.split('local memory:')[1].split('requests')[0].strip()
                remote = line.split('remote memory:')[1].split('requests')[0].strip()
                stats['core0_numa_local'] = local
                stats['core0_numa_remote'] = remote
            elif 'Core [0]' in line and 'MOVIN' in line and 'inst_count' in line:
                stats['core0_movin'] = line.split('inst_count')[1].strip()
            elif 'Core [0]' in line and 'COMP' in line and 'inst_count' in line:
                stats['core0_comp'] = line.split('inst_count')[1].strip()
    return stats

# Sweep sizes
sizes = [
    (32, 32, 32),
    (128, 128, 128),
    (256, 256, 256),
    (512, 512, 512),
    (1024, 1024, 1024),
]

print(f"{'M':>6} {'N':>6} {'K':>6} | {'Cycles':>8} {'Time(us)':>9} | {'SA Util%':>9} {'DRAM BW':>9} {'DRAM Reqs':>10} | {'MOVIN':>6} {'COMP':>20}")
print("-" * 110)

for M, N, K in sizes:
    log_path = run_matmul(M, N, K)
    if log_path:
        stats = parse_log(log_path)
        cycles = stats.get('total_cycles', 0)
        time_us = cycles / 940.0  # 940 MHz -> microseconds
        print(f"{M:>6} {N:>6} {K:>6} | {cycles:>8} {time_us:>8.2f} | "
              f"{stats.get('core0_sa_util', 'N/A'):>9} {stats.get('core0_dram_bw', 'N/A'):>8} {stats.get('core0_dram_reqs', 'N/A'):>10} | "
              f"{stats.get('core0_movin', 'N/A'):>6} {stats.get('core0_comp', 'N/A'):>20}")
    else:
        print(f"{M:>6} {N:>6} {K:>6} | {'FAILED':>8}")
