import os
import csv
import re

LOG_DIR = os.path.join(os.environ.get("TORCHSIM_DIR", "."), "experiments/artifact/speedup/results")
BASELINE_CSV = os.path.join(os.environ.get("TORCHSIM_DIR", "."), "experiments/artifact/baseline_latency.csv")

def format_with_speedup(value, ref, speedup_list=None):
    try:
        if value == "" or ref == "" or float(value) == 0:
            return "N/A"
        val = float(value)
        ref = float(ref)
        spd = ref / val
        if speedup_list is not None:
            speedup_list.append(spd)
        val_str = f"{float(val):>7.3f}"
        spd_str = f"{spd:.2f}×"
        return f"{val_str} ({spd_str:>7})"
    except (ValueError, TypeError):
        return "N/A"

def compute_geomean(errors):
    if not errors:
        return "N/A"
    filtered = [abs(e) for e in errors if e > 0]
    if not filtered:
        return "0.00x"
    prod = 1.0
    for e in filtered:
        prod *= e
    geo = prod ** (1.0 / len(filtered))
    return f"{geo:.2f}x"

if __name__ == "__main__":
    # 1. Generate cycle_map
    average_time_map = {}
    for file in os.listdir(LOG_DIR):
        if file.endswith(".txt"):
            full_path = os.path.join(LOG_DIR, file)
            full_name = file[:-4]
            name = full_name.split("_systolic", 1)[0]
            if "ils" in full_name:
                name = name
            elif "booksim" in full_name:
                name = name +"cn"
            elif "simple_noc" in full_name:
                name = name +"sn"
            else:
                raise ValueError(f"Unsupported file name format: {file}")
            with open(full_path) as f:
                for line in f:
                    match = re.search(r"Average simulation time\s*=\s*([0-9]+(?:\.[0-9]+)?)", line)
                    if match:
                        average_time_map[name] = float(match.group(1))
                        break

    # Speedup list init
    accelsim_speedup = []
    mnpusim_speedup = []
    torchsim_ils_sn_speedup = []
    torchsim_sn_speedup = []
    torchsim_cn_speedup = []

    # Header
    print("[*] Summary of Latency (Seconds) and Speedup (vs Accel-Sim)")
    print("=" * 165)
    print(f"{'Workload':>30} {'Accel-Sim':>25} {'mNPUSim':>25} {'PyTorchSim(ILS)-SN':>25} {'PyTorchSim-SN':>25} {'PyTorchSim-CN':>25}")
    print("=" * 165)

    with open(BASELINE_CSV, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            workload = row["Workload"].lstrip('\ufeff')
            accelsim = row["Accel-Sim"]
    
            mnpusim = format_with_speedup(row["mNPUSim"], accelsim, mnpusim_speedup)

            togsim_ils_sn_val = average_time_map.get("ils_" + workload, "")
            togsim_sn_val = average_time_map.get(workload+"sn", "")
            togsim_cn_val = average_time_map.get(workload+"cn", "")
            torchsim_ils_sn = format_with_speedup(togsim_ils_sn_val, accelsim, torchsim_ils_sn_speedup)
            torchsim_sn = format_with_speedup(togsim_sn_val, accelsim, torchsim_sn_speedup)
            torchsim_cn = format_with_speedup(togsim_cn_val, accelsim, torchsim_cn_speedup)

            print(f"{workload:>30} {accelsim:>25} {mnpusim:>25} {torchsim_ils_sn:>25} {torchsim_sn:>25} {torchsim_cn:>25}")

    # MAE row
    print("=" * 165)
    print(f"{'[*] Geomean Speedup':>30} {'1x':>25} {compute_geomean(mnpusim_speedup):>25} {compute_geomean(torchsim_ils_sn_speedup):>25} {compute_geomean(torchsim_sn_speedup):>25} {compute_geomean(torchsim_cn_speedup):>25}")