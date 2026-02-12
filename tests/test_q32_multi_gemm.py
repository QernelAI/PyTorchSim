"""Run 4 identical matmuls concurrently on 4 Q32 cores via interactive TOGSim.

Demonstrates multi-core utilization by:
1. Compiling a matmul through the standard path to generate the TOG
2. Launching the same TOG to 4 separate partitions (cores 0-3)
3. Running the simulation — all 4 cores execute in parallel
4. Comparing single-core baseline vs 4-core concurrent execution

Set TOGSIM_CONFIG to configs/q32_4core_dsp.json before running.
"""
import os
import sys
import glob
import re
import time
import json
import subprocess
import shlex
import tempfile
import datetime

base_path = os.environ.get('TORCHSIM_DIR', default='/workspace/PyTorchSim')
sys.path.append(base_path)

import torch
import torch._dynamo

from Scheduler.scheduler import PyTorchSimRunner
module = PyTorchSimRunner.setup_device()
device = module.custom_device()

from PyTorchSimFrontend import extension_config

CONFIG_PATH = extension_config.CONFIG_TOGSIM_CONFIG
LOG_DIR = os.path.join(base_path, 'togsim_results')
DUMP_PATH = extension_config.CONFIG_TORCHSIM_DUMP_PATH
NUM_Q32_CORES = 4
NUM_CORES = 5


def find_latest_tog():
    """Find the most recently created tile_graph.onnx."""
    search_dir = os.path.join(DUMP_PATH, 'outputs')
    files = glob.glob(os.path.join(search_dir, '**/tile_graph.onnx'), recursive=True)
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def get_latest_log():
    logs = sorted(glob.glob(os.path.join(LOG_DIR, '*.log')))
    return logs[-1] if logs else None


def compile_matmul(M, N, K):
    """Compile and run a single matmul (standard path) to generate the TOG.

    Returns the path to tile_graph.onnx.
    """
    def custom_matmul(a, b):
        return torch.matmul(a, b)

    torch.manual_seed(0)
    torch._dynamo.reset()

    x = torch.randn(M, K).to(device=device)
    w = torch.randn(K, N).to(device=device)
    opt_fn = torch.compile(dynamic=False)(custom_matmul)

    before_tog = find_latest_tog()
    before_mtime = os.path.getmtime(before_tog) if before_tog else 0

    try:
        _ = opt_fn(x, w)
    except Exception:
        pass
    time.sleep(0.5)

    after_tog = find_latest_tog()
    after_mtime = os.path.getmtime(after_tog) if after_tog else 0

    if after_tog and after_mtime > before_mtime:
        return after_tog
    return after_tog


def run_multicore_sim(tog_path, num_partitions=4):
    """Launch the same TOG on N partitions via interactive TOGSim.

    Creates an empty attribute file (no subgraph_map) so subgraphs have
    core_id=-1, meaning they get allocated to whichever core polls — since
    each partition has exactly one core, each core gets its own copy of the TOG.

    Returns path to the simulation log.
    """
    togsim_bin = os.path.join(base_path, "TOGSim/build/bin/Simulator")
    cmd = f"{togsim_bin} --config {CONFIG_PATH} --mode interactive"

    # Empty attribute: no subgraph_map → core_id=-1 → any core can pick up
    attr_dir = tempfile.mkdtemp(prefix="q32_multicore_")
    attr_path = os.path.join(attr_dir, "attr.json")
    with open(attr_path, "w") as f:
        json.dump({}, f)

    # Redirect stdout to log file for parsing
    os.makedirs(LOG_DIR, exist_ok=True)
    log_name = datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + "_multicore.log"
    log_path = os.path.join(LOG_DIR, log_name)
    log_handle = open(log_path, 'w')

    proc = subprocess.Popen(
        shlex.split(cmd),
        stdin=subprocess.PIPE,
        stdout=log_handle,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )

    def send(command):
        proc.stdin.write(command + '\n')
        proc.stdin.flush()
        return proc.stderr.readline().strip()

    # Launch the TOG to each Q32 partition
    for pid in range(num_partitions):
        resp = send(f"launch {CONFIG_PATH} {tog_path} {attr_path} 0 {pid}")
        print(f"    Partition {pid}: {resp}")

    # quit → C++ runs cycle() to completion then print_core_stat()
    send("quit")
    proc.wait()
    log_handle.close()

    return log_path


def parse_log(log_path, num_cores=NUM_CORES):
    """Extract per-core and BookSim2 stats from a TOGSim log."""
    stats = {'per_core': {i: {} for i in range(num_cores)}}

    with open(log_path) as f:
        for line in f:
            if 'Total execution cycles' in line:
                m = re.search(r'Total execution cycles:\s*(\d+)', line)
                if m:
                    stats['total_cycles'] = int(m.group(1))

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
                    m = re.search(
                        r'local memory:\s*(\d+)\s*requests.*remote memory:\s*(\d+)\s*requests',
                        line)
                    if m:
                        core['numa_local'] = m.group(1)
                        core['numa_remote'] = m.group(2)

                if 'MOVIN' in line and 'inst_count' in line:
                    core['movin'] = line.split('inst_count')[-1].strip()

                if 'COMP' in line and 'inst_count' in line:
                    core['comp'] = line.split('inst_count')[-1].strip()

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


def print_table(label, stats):
    cycles = stats.get('total_cycles', 0)
    time_us = cycles / 940.0 if cycles else 0

    print(f"\n  {label}")
    print(f"  Total cycles: {cycles}  ({time_us:.2f} us)")

    pkt_lat = stats.get('booksim_pkt_lat', 'N/A')
    inj_rate = stats.get('booksim_injected_rate', 'N/A')
    acc_rate = stats.get('booksim_accepted_rate', 'N/A')
    print(f"  BookSim2: pkt_latency={pkt_lat}, injected_rate={inj_rate}, accepted_rate={acc_rate}")

    header = (f"  {'Core':>6} {'Cycles':>8} {'SA Util%':>9} {'DRAM BW':>8} "
              f"{'DRAM Reqs':>10} {'NUMA L':>7} {'NUMA R':>7} {'MOVIN':>6} {'COMP':>6}")
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
print("=" * 80)
print("Q32 Multi-GEMM Test: 4 Concurrent Matmuls on 4 Q32 Cores")
print("=" * 80)

sizes = [
    (512, 512, 512),
    (1024, 1024, 1024),
]

for M, N, K in sizes:
    print(f"\n{'~'*70}")
    print(f"  Matmul {M}x{N}x{K}")
    print(f"{'~'*70}")

    # --- Step 1: Compile and run single-core baseline ---
    print(f"\n  [Step 1] Compiling matmul (single-core baseline) ...")
    baseline_log_before = get_latest_log()
    tog_path = compile_matmul(M, N, K)
    time.sleep(0.5)
    baseline_log = get_latest_log()

    if not tog_path:
        print(f"    FAILED: could not find tile_graph.onnx")
        continue
    print(f"    TOG: {tog_path}")

    baseline_stats = None
    if baseline_log and baseline_log != baseline_log_before:
        baseline_stats = parse_log(baseline_log)
        print_table(f"BASELINE: 1 matmul on 1 core", baseline_stats)

    # --- Step 2: Run 4 concurrent matmuls on 4 cores ---
    print(f"\n  [Step 2] Launching 4 concurrent matmuls (interactive TOGSim) ...")
    multicore_log = run_multicore_sim(tog_path, num_partitions=NUM_Q32_CORES)
    multicore_stats = parse_log(multicore_log)
    print_table(f"MULTI-CORE: 4 matmuls on 4 cores (concurrent)", multicore_stats)

    # --- Comparison ---
    b_cycles = baseline_stats.get('total_cycles', 0) if baseline_stats else 0
    m_cycles = multicore_stats.get('total_cycles', 0)

    if b_cycles and m_cycles:
        # 4 matmuls completed in m_cycles; 1 matmul takes b_cycles
        # Ideal parallel: m_cycles == b_cycles (4x throughput)
        overhead_pct = ((m_cycles - b_cycles) / b_cycles) * 100 if b_cycles else 0
        print(f"\n  Comparison:")
        print(f"    1 matmul  / 1 core:  {b_cycles:>8} cycles ({b_cycles/940:.2f} us)")
        print(f"    4 matmuls / 4 cores: {m_cycles:>8} cycles ({m_cycles/940:.2f} us)")
        print(f"    Overhead vs ideal parallel: {overhead_pct:+.1f}%")
        print(f"    Throughput: {4 * 940.0 / m_cycles:.2f} matmul/us")

        # Check per-core activity
        active_cores = sum(
            1 for cid in range(NUM_Q32_CORES)
            if multicore_stats['per_core'][cid].get('dram_reqs', '0') != '0'
        )
        print(f"    Active Q32 cores: {active_cores}/{NUM_Q32_CORES}")

print(f"\n{'='*80}")
print("Done")
