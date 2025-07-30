import os
import csv
import re
import matplotlib.pyplot as plt
import numpy as np

TORCHSIM_DIR = os.environ.get("TORCHSIM_DIR", ".")
LOG_DIR = os.path.join(TORCHSIM_DIR, "experiments/artifact/speedup/results")
BASELINE_CSV = os.path.join(TORCHSIM_DIR, "experiments/artifact/baseline_latency.csv")


def plot_speedup_bars(data: dict, filename: str):
    colors = {
        'Accel-Sim': '#A6A6A6',
        'mNPUSim': '#E97132',
        'PyTorchSim(ILS)-SN': '#4EA72E',
        'PyTorchSim-SN': '#0070C0',
        'PyTorchSim-CN': '#A6CAEC',
    }

    labels = list(data.keys())
    num_sims = len(colors)
    bar_width = 1
    fig, ax = plt.subplots(figsize=(48, 16))

    grouped_data = {sim: [[], []] for sim in colors}
    x_pos = []
    x_offset = 0

    for key, value in data.items():
        for i, (sim, color) in enumerate(colors.items()):
            grouped_data[sim][0].append(value[i])
            grouped_data[sim][1].append(x_offset + bar_width * i)
        x_pos.append(x_offset + bar_width * (num_sims // 2))
        x_offset += bar_width * (num_sims + 2)

    for sim, (heights, xpos) in grouped_data.items():
        bars = ax.bar(xpos, heights, width=bar_width, color=colors[sim], label=sim, edgecolor='black')
        mae_val = heights[-1]
        ax.text(
            xpos[-1],
            mae_val + 2 if mae_val >= 0 else mae_val - 6,
            f'{mae_val:.1f}x',
            ha='center',
            va='bottom' if mae_val >= 0 else 'top',
            fontsize=9,
            rotation=90
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=20, ha='right')
    ax.set_yscale('log')
    ax.set_ylim(0.1, 150)
    ax.set_yticks([0.1, 1, 10, 100])
    ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())
    ax.yaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend()

    plt.savefig(filename)
    plt.close()
    print(f"Saved plot to {filename}")

def format_with_speedup(value, ref, speedup_list=None):
    try:
        if value == "" or ref == "" or float(value) == 0:
            return "N/A", 0.0
        val = float(value)
        ref = float(ref)
        spd = ref / val
        if speedup_list is not None:
            speedup_list.append(spd)
        val_str = f"{float(val):>7.3f}"
        spd_str = f"{spd:.2f}×"
        return f"{val_str} ({spd_str:>7})", spd
    except (ValueError, TypeError):
        return "N/A", 0.0

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
    return geo

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
            with open(full_path, errors="ignore") as f:
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

    # Plot data
    plot_data ={}

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
    
            mnpusim, mnpusim_spd = format_with_speedup(row["mNPUSim"], accelsim, mnpusim_speedup)

            togsim_ils_sn_val = average_time_map.get("ils_" + workload, "")
            togsim_sn_val = average_time_map.get(workload+"sn", "")
            togsim_cn_val = average_time_map.get(workload+"cn", "")
            torchsim_ils_sn, ils_sn_spd = format_with_speedup(togsim_ils_sn_val, accelsim, torchsim_ils_sn_speedup)
            torchsim_sn, sn_spd = format_with_speedup(togsim_sn_val, accelsim, torchsim_sn_speedup)
            torchsim_cn, cn_spd = format_with_speedup(togsim_cn_val, accelsim, torchsim_cn_speedup)
            plot_data[workload] = [1.0, mnpusim_spd, ils_sn_spd, sn_spd, cn_spd]
            print(f"{workload:>30} {accelsim:>25} {mnpusim:>25} {torchsim_ils_sn:>25} {torchsim_sn:>25} {torchsim_cn:>25}")

    # MAE row
    geomean_accelsim = 1.0
    geomean_mnpusim = compute_geomean(mnpusim_speedup)
    geomean_torchsim_ils_sn = compute_geomean(torchsim_ils_sn_speedup)
    geomean_torchsim_sn = compute_geomean(torchsim_sn_speedup)
    geomean_torchsim_cn = compute_geomean(torchsim_cn_speedup)
    plot_data["Geomean"] = [geomean_accelsim, geomean_mnpusim, geomean_torchsim_ils_sn, geomean_torchsim_sn, geomean_torchsim_cn]
    print("=" * 165)
    print(f"{'Geomean Speedup':>30} {'1x':>25} {geomean_mnpusim:>24.2f}x {geomean_torchsim_ils_sn:>24.2f}x {geomean_torchsim_sn:>24.2f}x {geomean_torchsim_cn:>24.2f}x")
    path = os.path.join(TORCHSIM_DIR, "experiments/artifact/speedup/speedup.png")
    plot_speedup_bars(plot_data, path)
