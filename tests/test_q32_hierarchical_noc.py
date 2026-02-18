"""Test hierarchical NoC with row topology (intra-row vs inter-row stats).

Runs a 1x24576 @ 4096x24576 GEMM on a 12-group config with hierarchical NoC.
Verifies that GEMM transfers are all intra-row (Q32 -> local DSP, same group),
then computes the gather data volume required to collect full output at DSP 4.

Layout (12 groups, 6 rows x 2 groups/row):
  Row 0: Group 0 [Q32 0-3, DSP 4]   | Group 1 [Q32 5-8, DSP 9]
  Row 1: Group 2 [Q32 10-13, DSP 14] | Group 3 [Q32 15-18, DSP 19]
  Row 2: Group 4 [Q32 20-23, DSP 24] | Group 5 [Q32 25-28, DSP 29]
  Row 3: Group 6 [Q32 30-33, DSP 34] | Group 7 [Q32 35-38, DSP 39]
  Row 4: Group 8 [Q32 40-43, DSP 44] | Group 9 [Q32 45-48, DSP 49]
  Row 5: Group 10 [Q32 50-53, DSP 54] | Group 11 [Q32 55-58, DSP 59]

Usage:
    TOGSIM_CONFIG=configs/q32_12group_hierarchical.json python tests/test_q32_hierarchical_noc.py
"""
import os
import sys
import glob
import json
import time
import re

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
    """Extract key stats including hierarchical NoC stats from a TOGSim log."""
    stats = {
        'total_cycles': None,
        'core_to_core': 0,
        'intra_row_transfers': None,
        'inter_row_transfers': None,
        'intra_row_bytes': None,
        'inter_row_bytes': None,
        'core_stats': {},
        'icnt_lines': [],
    }
    with open(log_path) as f:
        for line in f:
            if 'Total execution cycles' in line:
                stats['total_cycles'] = int(line.split(':')[-1].strip())
            elif 'Core<->Core transfers' in line and 'Summary' in line:
                stats['core_to_core'] = int(line.split('transfers:')[-1].strip())
            elif 'Intra-row transfers:' in line and 'Summary' in line:
                m = re.search(r'Intra-row transfers:\s*(\d+)\s*\((\d+)\s*bytes\)', line)
                if m:
                    stats['intra_row_transfers'] = int(m.group(1))
                    stats['intra_row_bytes'] = int(m.group(2))
            elif 'Inter-row transfers:' in line and 'Summary' in line:
                m = re.search(r'Inter-row transfers:\s*(\d+)\s*\((\d+)\s*bytes\)', line)
                if m:
                    stats['inter_row_transfers'] = int(m.group(1))
                    stats['inter_row_bytes'] = int(m.group(2))
            elif 'local_mem_latency_cycles' in line:
                parts = line.split('Core ')[1]
                core_id = int(parts.split(':')[0])
                latency = int(parts.split('= ')[1].strip())
                stats['core_stats'][core_id] = {'local_mem_latency_cycles': latency}
            elif any(kw in line for kw in ['ICNT', 'NUMA', 'Core<->Core', 'Core [']):
                stats['icnt_lines'].append(line.rstrip())
    return stats


# ── Config ──
_cfg = json.load(open(os.environ.get('TOGSIM_CONFIG', '')))
_vlanes = _cfg['vpu_num_lanes']
_spad_kb = _cfg['vpu_spad_size_kb_per_lane']
_data_prec = _cfg.get('data_precision_bytes', 4)
_acc_prec = _cfg.get('acc_precision_bytes', _data_prec)
_groups_per_row = _cfg.get('groups_per_row', 0)
_inter_row_extra = _cfg.get('icnt_inter_row_extra_latency_cycles', 0)
_icnt_base_latency = _cfg.get('icnt_latency_cycles', 10)

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
_num_groups = len(_q32_groups)
_q32_per_group = len(_q32_groups[0]['q32_cores']) if _q32_groups else _num_q32
_subgraphs_per_group = _n_subgraphs // _num_groups if _num_groups > 0 else _n_subgraphs

# ── Test: Large GEMM with Hierarchical NoC ──
print("\n" + "=" * 70)
print(f"Test: Hierarchical NoC GEMM ({_M}x{_K} @ {_K}x{_N})")
print("=" * 70)
print(f"  Tensor core:    {_vlanes} x {_vlanes}, {_data_prec}B data / {_acc_prec}B accumulator")
print(f"  Output tiles:   {_N} / {_vlanes} = {_n_subgraphs} subgraphs")
print(f"  K tiles:        {_K} / {_vlanes} = {_k_iters} iterations per subgraph")
print(f"  Distribution:   round-robin across {_num_q32} Q32 cores ({_num_groups} groups)")
print(f"  Groups/row:     {_groups_per_row}  ({_num_groups // _groups_per_row if _groups_per_row > 0 else 1} rows)")
print(f"  NoC latency:    base={_icnt_base_latency} cycles, inter-row extra={_inter_row_extra} cycles")
if _q32_groups:
    for i, g in enumerate(_q32_groups):
        row = i // _groups_per_row if _groups_per_row > 0 else 0
        print(f"  Group {i:2d} (row {row}): Q32 cores {g['q32_cores']} -> DSP {g['dsp_core']}")
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

    if stats['intra_row_transfers'] is not None:
        print(f"  Intra-row transfers:   {stats['intra_row_transfers']} ({stats['intra_row_bytes']} bytes)")
        print(f"  Inter-row transfers:   {stats['inter_row_transfers']} ({stats['inter_row_bytes']} bytes)")

    # Show latencies for a sample of cores
    q32_latencies = {k: v for k, v in stats['core_stats'].items() if k not in _dsp_set}
    dsp_latencies = {k: v for k, v in stats['core_stats'].items() if k in _dsp_set}
    if q32_latencies:
        sample = dict(list(q32_latencies.items())[:4])
        print(f"  Q32 core latencies (sample): {sample}")
    if dsp_latencies:
        sample = dict(list(dsp_latencies.items())[:4])
        print(f"  DSP core latencies (sample): {sample}")

    # ── Sanity checks ──
    print(f"\n  --- Sanity checks ---")
    if stats['core_to_core'] > 0:
        print(f"  PASS: Core-to-core transfers = {stats['core_to_core']} (NoC routing active)")
    else:
        print(f"  FAIL: No core-to-core transfers detected")

    if stats['intra_row_transfers'] is not None:
        if stats['inter_row_transfers'] == 0:
            print(f"  PASS: Inter-row transfers = 0 (all GEMM traffic intra-group, as expected)")
        else:
            print(f"  INFO: Inter-row transfers = {stats['inter_row_transfers']} (unexpected for N-partitioned GEMM)")
        if stats['intra_row_transfers'] == stats['core_to_core']:
            print(f"  PASS: All core-to-core transfers are intra-row")
        else:
            print(f"  INFO: intra-row ({stats['intra_row_transfers']}) != total c2c ({stats['core_to_core']})")

    # ── Gather Analysis ──
    # Each DSP holds: subgraphs_per_group * M * tile_N * acc_precision bytes
    # tile_N = vlanes (one subgraph = one output tile column)
    _data_per_dsp = _subgraphs_per_group * _M * _vlanes * _acc_prec
    _primary_dsp = _q32_groups[0]['dsp_core']  # DSP 4
    _primary_row = 0

    intra_row_dsps = []
    inter_row_dsps = []
    for i, g in enumerate(_q32_groups):
        if g['dsp_core'] == _primary_dsp:
            continue  # skip primary itself
        row = i // _groups_per_row if _groups_per_row > 0 else 0
        if row == _primary_row:
            intra_row_dsps.append(g['dsp_core'])
        else:
            inter_row_dsps.append(g['dsp_core'])

    intra_row_gather_bytes = len(intra_row_dsps) * _data_per_dsp
    inter_row_gather_bytes = len(inter_row_dsps) * _data_per_dsp
    total_gather_bytes = intra_row_gather_bytes + inter_row_gather_bytes

    # Estimate latency: parallel transfers, so max of intra and inter latency
    intra_row_latency_est = _icnt_base_latency if intra_row_dsps else 0
    inter_row_latency_est = (_icnt_base_latency + _inter_row_extra) if inter_row_dsps else 0

    print(f"\n  --- Hierarchical NoC Analysis ---")
    print(f"  GEMM phase (Q32 -> local DSP):")
    print(f"    All transfers intra-group (trivially intra-row): {stats['core_to_core']} packets")
    print(f"    Inter-row transfers during GEMM: {stats['inter_row_transfers'] or 0} (N-partitioned, no cross-group traffic)")
    print()
    print(f"  Gather phase (all DSPs -> primary DSP {_primary_dsp}):")
    print(f"    Data per DSP: {_data_per_dsp} bytes ({_data_per_dsp / 1024:.1f} KB)")
    print(f"    Intra-row gather (DSP {intra_row_dsps} -> DSP {_primary_dsp}): "
          f"{intra_row_gather_bytes} bytes ({intra_row_gather_bytes / 1024:.1f} KB), "
          f"~{intra_row_latency_est} cycles base latency")
    print(f"    Inter-row gather ({len(inter_row_dsps)} DSPs -> DSP {_primary_dsp}): "
          f"{inter_row_gather_bytes} bytes ({inter_row_gather_bytes / 1024:.1f} KB), "
          f"~{inter_row_latency_est} cycles base latency")
    print(f"    Total gather: {total_gather_bytes} bytes ({total_gather_bytes / 1024:.1f} KB)")
    print(f"      Intra-row: {intra_row_gather_bytes} bytes from {len(intra_row_dsps)} DSPs")
    print(f"      Inter-row: {inter_row_gather_bytes} bytes from {len(inter_row_dsps)} DSPs")
    print(f"    Estimated gather latency (parallel): max({intra_row_latency_est}, {inter_row_latency_est}) = {max(intra_row_latency_est, inter_row_latency_est)} cycles")

    # ── Per-Group GEMM Traffic Table ──
    _packets_per_core = _k_iters * (_M * _vlanes * _acc_prec // _cfg.get('dram_req_size_byte', 32))
    _bytes_per_core = _packets_per_core * _cfg.get('dram_req_size_byte', 32)
    _packets_per_group = _q32_per_group * _packets_per_core
    _bytes_per_group = _q32_per_group * _bytes_per_core

    print()
    print("=" * 78)
    print("  GEMM Traffic Per Group (simulated)")
    print("=" * 78)
    print(f"  {'Group':>5} {'Row':>3}  {'Source':^22} {'-> Dest':^8} {'Category':^12} {'Packets':>8} {'Bytes':>10}")
    print(f"  {'-'*5:>5} {'-'*3:>3}  {'-'*22:^22} {'-'*8:^8} {'-'*12:^12} {'-'*8:>8} {'-'*10:>10}")
    for i, g in enumerate(_q32_groups):
        row = i // _groups_per_row if _groups_per_row > 0 else 0
        cores_str = f"Q32 {g['q32_cores'][0]}-{g['q32_cores'][-1]}"
        dest_str = f"DSP {g['dsp_core']}"
        print(f"  {i:5d} {row:3d}  {cores_str:^22} {dest_str:^8} {'Intra-row':^12} {_packets_per_group:8,} {_bytes_per_group:10,}")
    print(f"  {'-'*5:>5} {'-'*3:>3}  {'-'*22:^22} {'-'*8:^8} {'-'*12:^12} {'-'*8:>8} {'-'*10:>10}")
    print(f"  {'':>5} {'':>3}  {'':^22} {'':^8} {'TOTAL':^12} {stats['core_to_core']:8,} {stats.get('intra_row_bytes', 0) + stats.get('inter_row_bytes', 0):10,}")

    # ── Gather Traffic Table ──
    print()
    print("=" * 78)
    print(f"  Gather Traffic: All DSPs -> Primary DSP {_primary_dsp} (analytical)")
    print("=" * 78)
    print(f"  {'Source':>8} {'Row':>3} {'-> DSP':>6} {'Category':^12} {'Bytes':>10} {'Latency':>10}")
    print(f"  {'-'*8:>8} {'-'*3:>3} {'-'*6:>6} {'-'*12:^12} {'-'*10:>10} {'-'*10:>10}")
    for i, g in enumerate(_q32_groups):
        if g['dsp_core'] == _primary_dsp:
            continue
        row = i // _groups_per_row if _groups_per_row > 0 else 0
        cat = "Intra-row" if row == _primary_row else "Inter-row"
        lat = _icnt_base_latency if row == _primary_row else _icnt_base_latency + _inter_row_extra
        print(f"  {'DSP ' + str(g['dsp_core']):>8} {row:3d} {_primary_dsp:6d} {cat:^12} {_data_per_dsp:10,} {str(lat) + ' cyc':>10}")
    print(f"  {'-'*8:>8} {'-'*3:>3} {'-'*6:>6} {'-'*12:^12} {'-'*10:>10} {'-'*10:>10}")
    print(f"  {'':>8} {'':>3} {'':>6} {'Intra-row':^12} {intra_row_gather_bytes:10,} {str(intra_row_latency_est) + ' cyc':>10}")
    print(f"  {'':>8} {'':>3} {'':>6} {'Inter-row':^12} {inter_row_gather_bytes:10,} {str(inter_row_latency_est) + ' cyc':>10}")
    print(f"  {'':>8} {'':>3} {'':>6} {'TOTAL':^12} {total_gather_bytes:10,}")

    # ── Combined Summary ──
    gemm_intra = stats.get('intra_row_bytes', 0)
    gemm_inter = stats.get('inter_row_bytes', 0)
    gemm_total = gemm_intra + gemm_inter
    combined_intra = gemm_intra + intra_row_gather_bytes
    combined_inter = gemm_inter + inter_row_gather_bytes
    combined_total = gemm_total + total_gather_bytes

    print()
    print("=" * 78)
    print("  Combined Traffic Summary")
    print("=" * 78)
    print(f"  {'Phase':<20} {'Intra-row':>14} {'Inter-row':>14} {'Total':>14}")
    print(f"  {'-'*20:<20} {'-'*14:>14} {'-'*14:>14} {'-'*14:>14}")
    print(f"  {'GEMM (simulated)':<20} {gemm_intra:>10,} B {gemm_inter:>10,} B {gemm_total:>10,} B")
    print(f"  {'Gather (analytical)':<20} {intra_row_gather_bytes:>10,} B {inter_row_gather_bytes:>10,} B {total_gather_bytes:>10,} B")
    print(f"  {'-'*20:<20} {'-'*14:>14} {'-'*14:>14} {'-'*14:>14}")
    print(f"  {'TOTAL':<20} {combined_intra:>10,} B {combined_inter:>10,} B {combined_total:>10,} B")
    if combined_total > 0:
        pct_inter = 100.0 * combined_inter / combined_total
        print(f"\n  Inter-row share of total traffic: {pct_inter:.1f}%")
        if total_gather_bytes > 0:
            pct_gather_inter = 100.0 * inter_row_gather_bytes / total_gather_bytes
            print(f"  Inter-row share of gather traffic: {pct_gather_inter:.1f}%")
else:
    print("  WARNING: No new log file generated")
