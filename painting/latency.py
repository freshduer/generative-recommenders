import os
import re
import matplotlib.pyplot as plt

log_base_dir = "../logs/debug"
folders = ["single_stream", "2_streams", "4_streams", "8_streams"]
colors = ["tab:blue", "tab:red", "tab:orange", "tab:green"]   # 增加对比度
markers = ["o", "x", "s", "^"]  # 每个stream一个明显不同的marker

# 用于存储每个folder的数据
all_results = {}

pattern = re.compile(r"debug_(\d+)\.txt")

for folder in folders:
    folder_path = os.path.join(log_base_dir, folder)
    results = []
    for fname in sorted(os.listdir(folder_path)):
        m = pattern.match(fname)
        if not m:
            continue
        seq_len = int(m.group(1))
        with open(os.path.join(folder_path, fname), "r") as f:
            last_line = f.readlines()[-1].strip()
        
        qps_match = re.search(r"qps=([\d\.]+)", last_line)
        avg_match = re.search(r"avg_query_time=([\d\.]+)", last_line)
        
        if qps_match and avg_match:
            qps = float(qps_match.group(1))
            avg_time = float(avg_match.group(1))
            results.append((seq_len, qps, avg_time))
    
    results.sort(key=lambda x: x[0])
    all_results[folder] = results

# --- 图1: QPS ---
plt.figure(figsize=(8, 5))
for idx, folder in enumerate(folders):
    seq_lens = [r[0] for r in all_results[folder]]
    qps_vals = [r[1] for r in all_results[folder]]
    plt.plot(seq_lens, qps_vals, marker=markers[idx], color=colors[idx], label=folder)
plt.xlabel("uih_seq_len")
plt.ylabel("QPS")
plt.title("QPS vs uih_seq_len")
plt.grid(True)
plt.legend()
plt.savefig("qps_comparison.png", dpi=300)

# --- 图2: 平均查询时间 ---
plt.figure(figsize=(8, 5))
for idx, folder in enumerate(folders):
    seq_lens = [r[0] for r in all_results[folder]]
    avg_times = [r[2] for r in all_results[folder]]
    plt.plot(seq_lens, avg_times, marker=markers[idx], color=colors[idx], label=folder)
plt.xlabel("uih_seq_len")
plt.ylabel("Avg Query Time (s)")
plt.title("Avg Query Time vs uih_seq_len")
plt.grid(True)
plt.legend()
plt.savefig("avg_query_time_comparison.png", dpi=300)
