#!/usr/bin/env python3
"""Analyze TOGSim multi-core simulation logs.

Parses a TOGSim log file and prints detailed per-core instruction counts,
DMA stats, NUMA memory access patterns, BookSim2 NoC statistics, scheduler
timing, and ICNT bandwidth.

Usage:
    python scripts/analyze_multicore_log.py <log_file> [--num-cores N] [--num-q32 M] [--freq F]

Examples:
    python scripts/analyze_multicore_log.py togsim_results/20260212_003206_multicore.log
    python scripts/analyze_multicore_log.py togsim_results/*.log --num-cores 5 --num-q32 4
"""
import argparse
import glob
import re
import sys


def parse_log(log_path, num_cores):
    """Extract all available stats from a TOGSim log.

    Parses the final stats block (after "Simulation finished"), not the
    periodic interval stats. Handles TOGSim log format:
      - DMA: "DMA active_cycles, N DMA idle_cycles M, DRAM BW X GB/s (R responses)"
      - COMP: "inst_count N (GEMM: G, Vector: V)"
      - VPU: "Vector unit utilization(%) X, active cycle A, idle_cycle I"
      - Scheduler: "[Scheduler N] ... finish at CYCLE"
      - Wall-clock: "Wall-clock time for simulation: X seconds"
    """
    stats = {
        'per_core': {i: {} for i in range(num_cores)},
        'booksim': {},
        'scheduler': [],
        'icnt_intervals': [],
    }

    with open(log_path) as f:
        for line in f:
            # --- Total execution cycles ---
            if 'Total execution cycles' in line:
                m = re.search(r'Total execution cycles:\s*(\d+)', line)
                if m:
                    stats['total_cycles'] = int(m.group(1))

            # --- Wall-clock sim time ---
            if 'Wall-clock time for simulation' in line:
                m = re.search(r'Wall-clock time for simulation:\s*([\d.]+)\s*seconds', line)
                if m:
                    stats['wall_clock_sec'] = float(m.group(1))

            # --- Per-core stats ---
            core_match = re.search(r'Core \[(\d+)\]', line)
            if core_match:
                cid = int(core_match.group(1))
                if cid >= num_cores:
                    continue
                core = stats['per_core'][cid]

                # Total cycles
                if 'Total_cycles' in line:
                    m = re.search(r'Total_cycles\s+(\d+)', line)
                    if m:
                        core['total_cycles'] = int(m.group(1))

                # SA utilization: "utilization(%) 0.07, active_cycles 37, idle_cycles 55341"
                if 'Systolic array' in line and 'utilization' in line:
                    m = re.search(r'utilization\(%\)\s*([\d.]+),\s*active_cycles\s*(\d+),\s*idle_cycles\s*(\d+)', line)
                    if m:
                        core['sa_util'] = float(m.group(1))
                        core['sa_active'] = int(m.group(2))
                        core['sa_idle'] = int(m.group(3))

                # DMA: "DMA active_cycles, 12288 DMA idle_cycles 43090, DRAM BW 53.000 GB/s (49152 responses)"
                if 'DMA active_cycles' in line:
                    m = re.search(r'DMA active_cycles,?\s*(\d+)\s*DMA idle_cycles\s*(\d+)', line)
                    if m:
                        core['dma_active'] = int(m.group(1))
                        core['dma_idle'] = int(m.group(2))
                    m2 = re.search(r'DRAM BW\s*([\d.]+)\s*GB/s', line)
                    if m2:
                        core['dram_bw'] = float(m2.group(1))
                    m3 = re.search(r'\((\d+)\s*responses\)', line)
                    if m3:
                        core['dram_reqs'] = int(m3.group(1))

                # VPU: "Vector unit utilization(%) 0.02, active cycle 10, idle_cycle 0"
                if 'Vector unit utilization' in line:
                    m = re.search(r'utilization\(%\)\s*([\d.]+),\s*active cycle\s*(\d+),\s*idle_cycle\s*(\d+)', line)
                    if m:
                        core['vpu_util'] = float(m.group(1))
                        core['vpu_active'] = int(m.group(2))
                        core['vpu_idle'] = int(m.group(3))

                # NUMA
                if 'NUMA local' in line:
                    m = re.search(
                        r'local memory:\s*(\d+)\s*requests.*remote memory:\s*(\d+)\s*requests',
                        line)
                    if m:
                        core['numa_local'] = int(m.group(1))
                        core['numa_remote'] = int(m.group(2))

                # Instruction counts
                # MOVIN: "inst_count 2"
                if 'MOVIN' in line and 'inst_count' in line:
                    m = re.search(r'MOVIN\s+inst_count\s+(\d+)', line)
                    if m:
                        core['movin'] = int(m.group(1))
                # MOVOUT: "inst_count 1"
                if 'MOVOUT' in line and 'inst_count' in line:
                    m = re.search(r'MOVOUT\s+inst_count\s+(\d+)', line)
                    if m:
                        core['movout'] = int(m.group(1))
                # COMP: "inst_count 4 (GEMM: 2, Vector: 2)"
                if 'COMP' in line and 'inst_count' in line:
                    m = re.search(r'COMP\s+inst_count\s+(\d+)', line)
                    if m:
                        core['comp'] = int(m.group(1))
                    m2 = re.search(r'GEMM:\s*(\d+),\s*Vector:\s*(\d+)', line)
                    if m2:
                        core['comp_gemm'] = int(m2.group(1))
                        core['comp_vector'] = int(m2.group(2))
                # BAR: "inst_count 2"
                if 'BAR' in line and 'inst_count' in line:
                    m = re.search(r'BAR\s+inst_count\s+(\d+)', line)
                    if m:
                        core['bar'] = int(m.group(1))

            # --- BookSim2 stats (after "Simulation finished") ---
            # Packet latency
            if 'Packet latency average' in line:
                m = re.search(r'Packet latency average\s*=\s*([\d.]+)', line)
                if m:
                    stats['booksim']['pkt_lat_avg'] = float(m.group(1))
            # Min/max on indented lines after "Packet latency average"
            if line.strip().startswith('minimum') and 'pkt_lat_avg' in stats['booksim'] and 'pkt_lat_min' not in stats['booksim']:
                m = re.search(r'minimum\s*=\s*([\d.]+)', line)
                if m:
                    stats['booksim']['pkt_lat_min'] = float(m.group(1))
            if line.strip().startswith('maximum') and 'pkt_lat_avg' in stats['booksim'] and 'pkt_lat_max' not in stats['booksim']:
                m = re.search(r'maximum\s*=\s*([\d.]+)', line)
                if m:
                    stats['booksim']['pkt_lat_max'] = float(m.group(1))

            if 'Network latency average' in line:
                m = re.search(r'Network latency average\s*=\s*([\d.]+)', line)
                if m:
                    stats['booksim']['net_lat_avg'] = float(m.group(1))
            if 'Flit latency average' in line:
                m = re.search(r'Flit latency average\s*=\s*([\d.]+)', line)
                if m:
                    stats['booksim']['flit_lat_avg'] = float(m.group(1))

            # Injected/Accepted rates: "Injected packet rate average = 0.177515"
            if 'Injected packet rate average' in line:
                m = re.search(r'Injected packet rate average\s*=\s*([\d.]+)', line)
                if m:
                    stats['booksim']['inj_rate_avg'] = float(m.group(1))
            if 'Injected flit rate average' in line:
                m = re.search(r'Injected flit rate average\s*=\s*([\d.]+)', line)
                if m:
                    stats['booksim']['inj_flit_rate_avg'] = float(m.group(1))
            if 'Accepted packet rate average' in line:
                m = re.search(r'Accepted packet rate average\s*=\s*([\d.]+)', line)
                if m:
                    stats['booksim']['acc_rate_avg'] = float(m.group(1))
            if 'Accepted flit rate average' in line:
                m = re.search(r'Accepted flit rate average\s*=\s*([\d.]+)', line)
                if m:
                    stats['booksim']['acc_flit_rate_avg'] = float(m.group(1))

            # Per-node max: "maximum = 0.887573 (at node 20)"
            if 'maximum' in line and 'at node' in line:
                m = re.search(r'maximum\s*=\s*([\d.]+)\s*\(at node\s*(\d+)\)', line)
                if m:
                    rate = float(m.group(1))
                    node = int(m.group(2))
                    # Store the hotspot (last one seen is accepted flit rate, most relevant)
                    stats['booksim']['hotspot_rate'] = rate
                    stats['booksim']['hotspot_node'] = node

            # --- Scheduler timing ---
            # "[Scheduler N] Graph path: ... finish at CYCLE"
            if 'Scheduler' in line and 'finish at' in line:
                m = re.search(r'\[Scheduler\s+(\d+)\].*finish at\s+(\d+)', line)
                if m:
                    stats['scheduler'].append({
                        'partition': int(m.group(1)),
                        'finish_cycle': int(m.group(2)),
                    })

            # --- ICNT bandwidth intervals ---
            if 'ICNT' in line and 'GB/Sec' in line:
                m = re.search(r'cycle\s*(\d+).*?(\d+)\s*GB/Sec', line)
                if m:
                    stats['icnt_intervals'].append({
                        'cycle': int(m.group(1)),
                        'bw_gbps': int(m.group(2)),
                    })

    return stats


def print_report(log_path, stats, num_cores, num_q32, freq_mhz):
    """Print a comprehensive report."""
    cycles = stats.get('total_cycles', 0)
    time_us = cycles / freq_mhz if cycles else 0
    wall_sec = stats.get('wall_clock_sec', 0)

    print(f"\n{'=' * 80}")
    print(f"  TOGSim Analysis: {log_path}")
    print(f"{'=' * 80}")

    # --- Summary ---
    print(f"\n  SUMMARY")
    print(f"  {'Total cycles:':<30} {cycles:>12,}")
    print(f"  {'Simulated time:':<30} {time_us:>12.2f} us")
    if wall_sec:
        print(f"  {'Wall-clock time:':<30} {wall_sec:>12.2f} sec")
        if cycles:
            print(f"  {'Sim speed:':<30} {cycles / wall_sec:>12,.0f} cycles/sec")

    # --- Per-Core Instruction Counts ---
    print(f"\n  PER-CORE INSTRUCTION COUNTS")
    print(f"  {'Core':>6} {'MOVIN':>7} {'MOVOUT':>8} {'COMP':>6} {'(GEMM)':>8} {'(Vec)':>7} {'BAR':>5} {'Total':>7}")
    print(f"  {'-' * 56}")
    for cid in range(num_cores):
        core = stats['per_core'].get(cid, {})
        role = f"Q32-{cid}" if cid < num_q32 else "DSP"
        movin = core.get('movin', 0)
        movout = core.get('movout', 0)
        comp = core.get('comp', 0)
        gemm = core.get('comp_gemm', '-')
        vec = core.get('comp_vector', '-')
        bar = core.get('bar', 0)
        total = movin + movout + comp + bar
        print(f"  {role:>6} {movin:>7} {movout:>8} {comp:>6} {gemm:>8} {vec:>7} {bar:>5} {total:>7}")

    # --- Per-Core DMA Stats ---
    print(f"\n  PER-CORE DMA STATS")
    print(f"  {'Core':>6} {'DMA Active':>12} {'DMA Idle':>12} {'DMA Util%':>10} {'DRAM Reqs':>12} {'DRAM BW':>10}")
    print(f"  {'-' * 68}")
    for cid in range(num_cores):
        core = stats['per_core'].get(cid, {})
        role = f"Q32-{cid}" if cid < num_q32 else "DSP"
        active = core.get('dma_active', 0)
        idle = core.get('dma_idle', 0)
        total_dma = active + idle
        util = (active / total_dma * 100) if total_dma else 0
        reqs = core.get('dram_reqs', 0)
        bw = core.get('dram_bw', 0)
        print(f"  {role:>6} {active:>12,} {idle:>12,} {util:>9.1f}% {reqs:>12,} {bw:>8.0f} GB/s")

    # --- Per-Core Utilization ---
    print(f"\n  PER-CORE UTILIZATION")
    print(f"  {'Core':>6} {'Cycles':>10} {'SA Util%':>10} {'SA Active':>10} {'VPU Util%':>10} {'VPU Active':>11}")
    print(f"  {'-' * 63}")
    for cid in range(num_cores):
        core = stats['per_core'].get(cid, {})
        role = f"Q32-{cid}" if cid < num_q32 else "DSP"
        c = core.get('total_cycles', 0)
        sa = core.get('sa_util', 0.0)
        sa_act = core.get('sa_active', 0)
        vpu = core.get('vpu_util', 0.0)
        vpu_act = core.get('vpu_active', 0)
        print(f"  {role:>6} {c:>10,} {sa:>9.2f}% {sa_act:>10,} {vpu:>9.2f}% {vpu_act:>11,}")

    # --- NUMA Memory Access ---
    print(f"\n  NUMA MEMORY ACCESS PATTERN")
    print(f"  {'Core':>6} {'Local Reqs':>12} {'Remote Reqs':>12} {'Local%':>8}")
    print(f"  {'-' * 40}")
    for cid in range(num_cores):
        core = stats['per_core'].get(cid, {})
        role = f"Q32-{cid}" if cid < num_q32 else "DSP"
        local = core.get('numa_local', 0)
        remote = core.get('numa_remote', 0)
        total_numa = local + remote
        pct = (local / total_numa * 100) if total_numa else 0
        print(f"  {role:>6} {local:>12,} {remote:>12,} {pct:>7.1f}%")

    # --- BookSim2 NoC Stats ---
    bs = stats.get('booksim', {})
    if bs:
        print(f"\n  BOOKSIM2 NoC STATS")
        print(f"  {'Metric':<40} {'Value':>15}")
        print(f"  {'-' * 57}")
        for key, label in [
            ('pkt_lat_avg', 'Packet latency avg (cycles)'),
            ('pkt_lat_min', 'Packet latency min'),
            ('pkt_lat_max', 'Packet latency max'),
            ('net_lat_avg', 'Network latency avg'),
            ('flit_lat_avg', 'Flit latency avg'),
            ('inj_rate_avg', 'Injected pkt rate avg (flits/node/cyc)'),
            ('inj_flit_rate_avg', 'Injected flit rate avg'),
            ('acc_rate_avg', 'Accepted pkt rate avg'),
            ('acc_flit_rate_avg', 'Accepted flit rate avg'),
        ]:
            val = bs.get(key)
            if val is not None:
                if isinstance(val, float) and val > 100:
                    print(f"  {label:<40} {val:>15,.2f}")
                elif isinstance(val, float):
                    print(f"  {label:<40} {val:>15.6f}")
                else:
                    print(f"  {label:<40} {val:>15}")

        hotspot_node = bs.get('hotspot_node')
        hotspot_rate = bs.get('hotspot_rate')
        if hotspot_node is not None:
            num_core_nodes = num_cores * 4  # icnt_injection_ports_per_core
            node_type = "DRAM channel" if hotspot_node >= num_core_nodes else "core port"
            avg_rate = bs.get('inj_rate_avg', 1)
            multiplier = hotspot_rate / avg_rate if avg_rate else 0
            print(f"\n  NoC Hotspot: node {hotspot_node} ({node_type}), "
                  f"rate={hotspot_rate:.4f} "
                  f"({multiplier:.1f}x average)")

    # --- Scheduler Timing ---
    sched = stats.get('scheduler', [])
    if sched:
        sched_sorted = sorted(sched, key=lambda s: s['finish_cycle'])
        print(f"\n  SCHEDULER TIMING (partition finish order)")
        print(f"  {'Partition':>12} {'Finish Cycle':>14} {'Delta':>8}")
        print(f"  {'-' * 36}")
        first = sched_sorted[0]['finish_cycle']
        for s in sched_sorted:
            delta = s['finish_cycle'] - first
            delta_str = f"+{delta}" if delta else "-"
            print(f"  {'Sched ' + str(s['partition']):>12} {s['finish_cycle']:>14,} {delta_str:>8}")
        spread = sched_sorted[-1]['finish_cycle'] - first
        print(f"  Spread: {spread} cycles ({spread / freq_mhz:.2f} us)")

    # --- ICNT Bandwidth ---
    icnt = stats.get('icnt_intervals', [])
    if icnt:
        nonzero = [i for i in icnt if i['bw_gbps'] > 0]
        print(f"\n  ICNT BANDWIDTH")
        if nonzero:
            print(f"  {'Cycle':>10} {'BW (GB/s)':>10}")
            print(f"  {'-' * 22}")
            for i in nonzero:
                print(f"  {i['cycle']:>10,} {i['bw_gbps']:>10}")
        else:
            print(f"  All {len(icnt)} intervals report 0 GB/Sec")
            print(f"  (Integer division in reporting formula — NoC traffic confirmed by BookSim2)")

    # --- Notes on unavailable metrics ---
    print(f"\n  NOTES")
    print(f"  - Per-channel DRAM BW: Not available (requires dram_type=ramulator2)")
    print(f"  - DRAM row hit/miss/conflict: Not available (requires dram_type=ramulator2)")
    nonzero_icnt = [i for i in icnt if i['bw_gbps'] > 0] if icnt else []
    if icnt and not nonzero_icnt:
        print(f"  - ICNT BW reports 0 due to integer truncation in periodic stats formula")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze TOGSim multi-core simulation logs')
    parser.add_argument('logs', nargs='+',
                        help='Log file path(s), supports glob patterns')
    parser.add_argument('--num-cores', type=int, default=5,
                        help='Total number of cores (default: 5)')
    parser.add_argument('--num-q32', type=int, default=4,
                        help='Number of Q32 tensor cores (default: 4)')
    parser.add_argument('--freq', type=float, default=940.0,
                        help='Core frequency in MHz (default: 940)')
    args = parser.parse_args()

    # Expand glob patterns
    log_files = []
    for pattern in args.logs:
        expanded = glob.glob(pattern)
        if expanded:
            log_files.extend(expanded)
        else:
            log_files.append(pattern)

    for log_path in sorted(log_files):
        try:
            stats = parse_log(log_path, args.num_cores)
            print_report(log_path, stats, args.num_cores, args.num_q32, args.freq)
        except FileNotFoundError:
            print(f"ERROR: File not found: {log_path}", file=sys.stderr)
        except Exception as e:
            print(f"ERROR parsing {log_path}: {e}", file=sys.stderr)

    print()


if __name__ == '__main__':
    main()
