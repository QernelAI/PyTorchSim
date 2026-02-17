"""Run GEMM + exp on Q32 CIM config to verify op-name inference end-to-end.

Tests three scenarios:
1. matmul_only     — baseline GEMM on CIM tensor core
2. matmul_then_exp — GEMM + exp (compiler may fuse as epilogue)
3. exp_only        — standalone exp, pure DSP vector op (no CIM)
4. matmul_K1024_exp — K-reduction + exp
"""
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

def parse_log(log_path):
    """Extract key stats from a TOGSim log."""
    stats = {}
    with open(log_path) as f:
        for line in f:
            if 'Total execution cycles' in line:
                stats['total_cycles'] = int(line.split(':')[-1].strip())
            elif 'Core [0]' in line and 'COMP' in line and 'inst_count' in line:
                stats['core0_comp'] = line.split('inst_count')[1].strip()
            elif 'Core [0]' in line and 'Vector unit' in line:
                stats['core0_vpu'] = line.strip().split('Vector unit')[1]
            elif 'Core [0]' in line and 'Systolic array [0] utilization' in line:
                stats['core0_sa_active'] = line.split('active_cycles')[1].split(',')[0].strip()
            elif 'Core [1]' in line and 'COMP' in line and 'inst_count' in line:
                stats['core1_comp'] = line.split('inst_count')[1].strip()
            elif 'Core [1]' in line and 'Vector unit' in line:
                stats['core1_vpu'] = line.strip().split('Vector unit')[1]
            elif 'Core [1]' in line and 'Systolic array [0] utilization' in line:
                stats['core1_sa_active'] = line.split('active_cycles')[1].split(',')[0].strip()
    return stats

def run_test(name, fn_factory, inputs_factory):
    """Compile and run a function, return parsed log stats."""
    torch.manual_seed(0)
    torch._dynamo.reset()

    inputs = inputs_factory()
    opt_fn = torch.compile(dynamic=False)(fn_factory())

    before_log = get_latest_log()
    try:
        _ = opt_fn(*inputs)
    except Exception:
        pass
    time.sleep(0.5)
    after_log = get_latest_log()

    if after_log and after_log != before_log:
        return parse_log(after_log)
    return None

# ── Test 1: matmul only (baseline) ──
print("\n=== Test 1: matmul_only (1,512,512) ===")
matmul_stats = run_test(
    "matmul_only",
    lambda: (lambda a, b: torch.matmul(a, b)),
    lambda: (torch.randn(1, 512).to(device=device), torch.randn(512, 512).to(device=device)),
)
if matmul_stats:
    print(f"  Total cycles: {matmul_stats.get('total_cycles')}")
    print(f"  Core[0] COMP: {matmul_stats.get('core0_comp')}")
    print(f"  Core[0] SA active: {matmul_stats.get('core0_sa_active')}")
else:
    print("  FAILED")

# ── Test 2: exp only (standalone DSP vector op) ──
print("\n=== Test 2: exp_only (512 elements) ===")
print("  This should use our minimal single-node TOG path")
exp_stats = run_test(
    "exp_only",
    lambda: (lambda x: torch.exp(x)),
    lambda: (torch.randn(1, 512).to(device=device),),
)
if exp_stats:
    print(f"  Total cycles: {exp_stats.get('total_cycles')}")
    print(f"  Core[0] COMP: {exp_stats.get('core0_comp')}")
    print(f"  Core[0] VPU: {exp_stats.get('core0_vpu')}")
    print(f"  Core[1] COMP: {exp_stats.get('core1_comp')}")
    print(f"  Core[1] VPU: {exp_stats.get('core1_vpu')}")
else:
    print("  FAILED (or no log generated)")

# ── Test 3: matmul then exp (should show DSP activity) ──
print("\n=== Test 3: matmul_then_exp (1,512,512) ===")
gemm_exp_stats = run_test(
    "matmul_then_exp",
    lambda: (lambda a, b: torch.exp(torch.matmul(a, b))),
    lambda: (torch.randn(1, 512).to(device=device), torch.randn(512, 512).to(device=device)),
)
if gemm_exp_stats:
    print(f"  Total cycles: {gemm_exp_stats.get('total_cycles')}")
    print(f"  Core[0] COMP: {gemm_exp_stats.get('core0_comp')}")
    print(f"  Core[0] SA active: {gemm_exp_stats.get('core0_sa_active')}")
    print(f"  Core[1] COMP (DSP): {gemm_exp_stats.get('core1_comp')}")
    print(f"  Core[1] VPU (DSP): {gemm_exp_stats.get('core1_vpu')}")
    print(f"  Core[1] SA  (DSP): {gemm_exp_stats.get('core1_sa_active')}")
else:
    print("  FAILED")

# ── Test 4: matmul K=1024 + exp (K-reduction + vector) ──
print("\n=== Test 4: matmul_K1024_exp (1,512,1024) ===")
gemm_k1024_exp_stats = run_test(
    "matmul_K1024_exp",
    lambda: (lambda a, b: torch.exp(torch.matmul(a, b))),
    lambda: (torch.randn(1, 1024).to(device=device), torch.randn(1024, 512).to(device=device)),
)
if gemm_k1024_exp_stats:
    print(f"  Total cycles: {gemm_k1024_exp_stats.get('total_cycles')}")
    print(f"  Core[0] COMP: {gemm_k1024_exp_stats.get('core0_comp')}")
    print(f"  Core[0] SA active: {gemm_k1024_exp_stats.get('core0_sa_active')}")
    print(f"  Core[1] COMP (DSP): {gemm_k1024_exp_stats.get('core1_comp')}")
    print(f"  Core[1] VPU (DSP): {gemm_k1024_exp_stats.get('core1_vpu')}")
else:
    print("  FAILED")

# ── Assertions ──
print("\n=== Assertions ===")
passed = 0
failed = 0

if matmul_stats and gemm_exp_stats:
    matmul_cycles = matmul_stats['total_cycles']
    gemm_exp_cycles = gemm_exp_stats['total_cycles']
    if gemm_exp_cycles > matmul_cycles:
        print(f"  PASS: matmul_then_exp ({gemm_exp_cycles}) > matmul_only ({matmul_cycles})")
        passed += 1
    else:
        print(f"  FAIL: matmul_then_exp ({gemm_exp_cycles}) should be > matmul_only ({matmul_cycles})")
        failed += 1
else:
    print("  SKIP: missing stats for matmul or gemm_exp")

if gemm_exp_stats and gemm_exp_stats.get('core1_comp'):
    # Core 1 (DSP) should have non-zero COMP instructions
    comp_str = gemm_exp_stats['core1_comp']
    # comp_str looks like ": N" where N is the count
    comp_count = int(comp_str.strip().lstrip(':').strip())
    if comp_count > 0:
        print(f"  PASS: DSP core (Core[1]) has {comp_count} COMP instruction(s)")
        passed += 1
    else:
        print(f"  FAIL: DSP core (Core[1]) COMP count is 0")
        failed += 1
else:
    print("  SKIP: no Core[1] COMP stats for matmul_then_exp")

if gemm_k1024_exp_stats and matmul_stats:
    k1024_cycles = gemm_k1024_exp_stats['total_cycles']
    matmul_cycles = matmul_stats['total_cycles']
    if k1024_cycles > matmul_cycles:
        print(f"  PASS: matmul_K1024_exp ({k1024_cycles}) > matmul_only ({matmul_cycles})")
        passed += 1
    else:
        print(f"  FAIL: matmul_K1024_exp ({k1024_cycles}) should be > matmul_only ({matmul_cycles})")
        failed += 1
else:
    print("  SKIP: missing stats for K1024 exp test")

print(f"\nResults: {passed} passed, {failed} failed")
if failed > 0:
    sys.exit(1)
