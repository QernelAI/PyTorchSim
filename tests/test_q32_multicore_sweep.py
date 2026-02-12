"""Run matmul at various sizes on Q32 4-core + DSP config with BookSim2 NoC.

Exercises round-robin GEMM distribution across 4 Q32 tensor cores (cores 0-3)
with a DSP on core 4 and BookSim2 flattened butterfly interconnect.

Set TOGSIM_CONFIG to configs/q32_4core_dsp.json before running.
"""
import os
import sys
import glob
import re
import time

sys.path.append(os.environ.get('TORCHSIM_DIR', default='/workspace/PyTorchSim'))

import torch
import torch._dynamo

from Scheduler.scheduler import PyTorchSimRunner
module = PyTorchSimRunner.setup_device()
device = module.custom_device()

LOG_DIR = os.path.join(os.environ.get('TORCHSIM_DIR', '/workspace/PyTorchSim'), 'togsim_results')
NUM_Q32_CORES = 4
DSP_CORE_ID = 4
NUM_CORES = 5


def get_latest_log():
    logs = sorted(glob.glob(os.path.join(LOG_DIR, '*.log')))
    return logs[-1] if logs else None


def run_matmul(M, N, K):
    """Run a single matmul and return the log path."""
    def custom_matmul(a, b):
        return torch.matmul(a, b)

    torch.manual_seed(0)
    torch._dynamo.reset()

    x = torch.randn(M, K).to(device=device)
    w = torch.randn(K, N).to(device=device)
    opt_fn = torch.compile(dynamic=False)(custom_matmul)

    before_log = get_latest_log()
    try:
        _ = opt_fn(x, w)
    except Exception:
        pass
    time.sleep(0.5)
    after_log = get_latest_log()

    if after_log != before_log:
        return after_log
    return None


def parse_multicore_log(log_path, num_cores=NUM_CORES):
    """Extract per-core stats and BookSim2 stats from a TOGSim log."""
    stats = {'per_core': {i: {} for i in range(num_cores)}}

    with open(log_path) as f:
        for line in f:
            # Total execution cycles
            if 'Total execution cycles' in line:
                stats['total_cycles'] = int(line.split(':')[-1].strip())

            # Per-core stats: Core [N] ...
            core_match = re.search(r'Core \[(\d+)\]', line)
            if core_match:
                cid = int(core_match.group(1))
                if cid >= num_cores:
                    continue
                core = stats['per_core'][cid]

                if 'Total_cycles' in line:
                    core['total_cycles'] = int(line.split('Total_cycles')[-1].strip())

                if 'Systolic array [0] utilization' in line:
                    m = re.search(r'utilization\(%\)\s*([\d.]+)', line)
                    if m:
                        core['sa_util'] = m.group(1)

                if 'DRAM BW' in line:
                    m = re.search(r'DRAM BW\s*([\d.]+)\s*GB/s', line)
                    if m:
                        core['dram_bw'] = m.group(1)
                    m2 = re.search(r'\((\d+)\s*responses', line)
                    if m2:
                        core['dram_reqs'] = m2.group(1)

                if 'NUMA local' in line:
                    m = re.search(r'local memory:\s*(\d+)\s*requests.*remote memory:\s*(\d+)\s*requests', line)
                    if m:
                        core['numa_local'] = m.group(1)
                        core['numa_remote'] = m.group(2)

                if 'MOVIN' in line and 'inst_count' in line:
                    core['movin'] = line.split('inst_count')[-1].strip()

                if 'COMP' in line and 'inst_count' in line:
                    core['comp'] = line.split('inst_count')[-1].strip()

            # BookSim2 stats
            if 'Packet latency average' in line:
                m = re.search(r'Packet latency average\s*=\s*([\d.]+)', line)
                if m:
                    stats['booksim_pkt_lat'] = m.group(1)

            if 'accepted rate average' in line:
                m = re.search(r'accepted rate average\s*=\s*([\d.]+)', line)
                if m:
                    stats['booksim_accepted_rate'] = m.group(1)

            if 'injected rate average' in line:
                m = re.search(r'injected rate average\s*=\s*([\d.]+)', line)
                if m:
                    stats['booksim_injected_rate'] = m.group(1)

    return stats


def print_results(M, N, K, stats):
    """Print a summary table for one matmul run."""
    cycles = stats.get('total_cycles', 0)
    time_us = cycles / 940.0 if cycles else 0

    print(f"\n--- Matmul {M}x{N}x{K} ---")
    print(f"  Total cycles: {cycles}  ({time_us:.2f} us)")

    # BookSim2 stats
    pkt_lat = stats.get('booksim_pkt_lat', 'N/A')
    inj_rate = stats.get('booksim_injected_rate', 'N/A')
    acc_rate = stats.get('booksim_accepted_rate', 'N/A')
    print(f"  BookSim2: pkt_latency={pkt_lat}, injected_rate={inj_rate}, accepted_rate={acc_rate}")

    # Per-core table
    header = f"  {'Core':>6} {'Cycles':>8} {'SA Util%':>9} {'DRAM BW':>8} {'DRAM Reqs':>10} {'NUMA L':>7} {'NUMA R':>7} {'MOVIN':>6} {'COMP':>6}"
    print(header)
    print("  " + "-" * len(header.strip()))

    for cid in range(NUM_CORES):
        core = stats['per_core'].get(cid, {})
        role = f"Q32-{cid}" if cid < NUM_Q32_CORES else "DSP"
        print(f"  {role:>6} "
              f"{core.get('total_cycles', 'N/A'):>8} "
              f"{core.get('sa_util', 'N/A'):>9} "
              f"{core.get('dram_bw', 'N/A'):>8} "
              f"{core.get('dram_reqs', 'N/A'):>10} "
              f"{core.get('numa_local', 'N/A'):>7} "
              f"{core.get('numa_remote', 'N/A'):>7} "
              f"{core.get('movin', 'N/A'):>6} "
              f"{core.get('comp', 'N/A'):>6}")


# Sweep sizes
sizes = [
    (128, 128, 128),
    (512, 512, 512),
    (1024, 1024, 1024),
    (2048, 2048, 2048),
]

print("=" * 80)
print("Q32 Multi-Core Sweep: 4 Q32 + 1 DSP with BookSim2 NoC")
print("=" * 80)

for M, N, K in sizes:
    log_path = run_matmul(M, N, K)
    if log_path:
        stats = parse_multicore_log(log_path)
        print_results(M, N, K, stats)
    else:
        print(f"\n--- Matmul {M}x{N}x{K} --- FAILED (no log produced)")
