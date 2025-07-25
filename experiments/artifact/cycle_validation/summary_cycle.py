import os
import math
import csv
import re

LOG_DIR = os.path.join(os.environ.get("TORCHSIM_DIR", "."), "experiments/artifact/logs")
BASELINE_CSV = os.path.join(os.environ.get("TORCHSIM_DIR", "."), "experiments/artifact/baseline_cycle.csv")

def format_with_error(value, ref, error_list=None):
    try:
        if value == "" or ref == "" or float(ref) == 0:
            return "N/A"
        val = float(value)
        ref = float(ref)
        err = ((val - ref) / ref) * 100
        if error_list is not None:
            error_list.append(abs(err))
        val_str = f"{int(val):>7}"
        err_str = f"{err:+.2f}%"
        return f"{val_str} ({err_str:>8})"
    except (ValueError, TypeError):
        return "N/A"

def compute_mean(errors):
    if not errors:
        return "N/A"
    abs_errors = [abs(err) for err in errors]
    return f"{sum(abs_errors) / len(errors):.2f}%"

if __name__ == "__main__":
    # 1. Generate cycle_map
    cycle_map = {}
    for file in os.listdir(LOG_DIR):
        if file.endswith(".log"):
            full_path = os.path.join(LOG_DIR, file)
            name = file[:-4]
            with open(full_path) as f:
                for line in f:
                    match = re.search(r"Total execution cycle:\s*([0-9]+)", line)
                    if match:
                        cycle_map[name] = int(match.group(1))
                        break

    # Error list init
    mnpusim_errors = []
    timeloop_errors = []
    maestro_errors = []
    scalesim_errors = []
    togsim_errors = []

    # Header
    print("[*] Summary of Total Execution Cycles with TPUv3-relative (%) Error")
    print("=" * 190)
    print(f"{'Workload':>30} {'TPUv3':>25} {'mNPUSim':>25} {'Timeloop':>25} {'Maestro':>25} {'SCALE-Sim v3':>25} {'TOGSim(Ours)':>25}")
    print("=" * 190)

    with open(BASELINE_CSV, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            workload = row["Workload"].lstrip('\ufeff')
            tpv3 = row["TPUv3"]
    
            mnpusim  = format_with_error(row["mNPUSim"], tpv3, mnpusim_errors)
            timeloop = format_with_error(row["Timeloop"], tpv3, timeloop_errors)
            maestro  = format_with_error(row["Maestro"], tpv3, maestro_errors)
            scalesim = format_with_error(row["SCALE-Sim v3"], tpv3, scalesim_errors)
    
            togsim_val = cycle_map.get(workload, "")
            if "softmax" in workload or "layernorm" in workload:
                togsim_str = format_with_error(str(togsim_val), tpv3, [])
            else:
                togsim_str = format_with_error(str(togsim_val), tpv3, togsim_errors)
    
            print(f"{workload:>30} {tpv3:>25} {mnpusim:>25} {timeloop:>25} {maestro:>25} {scalesim:>25} {togsim_str:>25}")

    # MAE row
    print("=" * 190)
    print(f"{'[*] Mean Absolute Error(%)':>30} {'0.00%':>25} {compute_mean(mnpusim_errors):>25} {compute_mean(timeloop_errors):>25} {compute_mean(maestro_errors):>25} {compute_mean(scalesim_errors):>25} {compute_mean(togsim_errors):>25}")