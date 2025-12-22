import numpy as np
import matplotlib.pyplot as plt

# ---- 场景参数 ----
# 延迟锚点 (ms)
lat_gpu_hit = 0.9    # 100% hit
lat_cpu_miss = 31.0  # 100% miss (10ms lookup + 21ms transfer)

# 数据分布参数
total_emb_size_gb = 200.0   # 2TB (20亿行 * 1KB)
hotspot_ratio = 0.1          # 10% items
hotspot_access_ratio = 0.9   # 90% requests

hotspot_size_gb = total_emb_size_gb * hotspot_ratio # 200 GB

# 变量: GPU Cache 从 0GB 到 80GB，间隔 2GB
cache_sizes = np.array(list(range(0, 82, 2)))

# ---- 计算逻辑 ----
hit_rates = []
latencies = []

# 对比组：假设热点不是均匀分布，而是更集中的 Zipf (幂律) 分布
# 这种情况下，头部 20% 的热点可能占据了 80% 的热点流量
zipf_hit_rates = []
zipf_latencies = []

for cache in cache_sizes:
    # --- 1. 均匀热点模型 (Uniform Hotspot) ---
    # 算出 cache 能装下百分之多少的热点
    fraction_hotspot_cached = min(1.0, cache / hotspot_size_gb)
    
    # 命中率 = 流量落入热点的概率 * 热点在缓存里的概率
    hit_rate = hotspot_access_ratio * fraction_hotspot_cached
    hit_rates.append(hit_rate)
    
    # 延迟 = GPU基准 + (CPU惩罚 * Miss率)
    # Miss率越高，传输的数据量越大，延迟呈线性增长
    lat = lat_gpu_hit + (lat_cpu_miss - lat_gpu_hit) * (1 - hit_rate)
    latencies.append(lat)
    
    # --- 2. Zipf 优化模型 (更接近真实推荐系统) ---
    # 假设：虽然 Cache 只占热点的 20%，但这 20% 是“热点中的热点”
    # 经验法则：前 20% 的热点数据往往承载了热点内部 60%-70% 的流量
    zipf_scaling = fraction_hotspot_cached ** 0.4  # 简单的幂律模拟
    zipf_hit_rate = hotspot_access_ratio * min(1.0, zipf_scaling)
    zipf_hit_rates.append(zipf_hit_rate)
    zipf_lat = lat_gpu_hit + (lat_cpu_miss - lat_gpu_hit) * (1 - zipf_hit_rate)
    zipf_latencies.append(zipf_lat)

# ---- 绘图 ----
plt.figure(figsize=(10, 6))

# 绘制均匀分布线
plt.plot(cache_sizes, latencies, 'o-', label='Uniform Hotspot (Worst Case)', color='tab:red', linewidth=2)
# 绘制 Zipf 分布线
plt.plot(cache_sizes, zipf_latencies, 'o--', label='Zipf Hotspot (Real World)', color='tab:blue', linewidth=2)

plt.axhline(y=31.0, color='gray', linestyle=':', label='Baseline (Pure CPU)')
plt.axhline(y=0.9, color='green', linestyle=':', label='Ideal (Pure GPU)')

plt.title(f'Batch Size 8 Latency vs GPU Cache Size\n(Hotspot Size: {hotspot_size_gb}GB)', fontsize=12)
plt.xlabel('GPU Cache Size (GB)', fontsize=10)
plt.ylabel('Avg Latency (ms)', fontsize=10)
plt.grid(True, alpha=0.3)
plt.legend()

# 标注关键点
plt.annotate(f"{latencies[-1]:.1f}ms", (cache_sizes[-1], latencies[-1]), textcoords="offset points", xytext=(0,10), ha='center', color='red')
plt.annotate(f"{zipf_latencies[-1]:.1f}ms", (cache_sizes[-1], zipf_latencies[-1]), textcoords="offset points", xytext=(0,-15), ha='center', color='blue')

plt.savefig("figures/latency_simulation.png")
plt.show()

print(f"Cache {int(cache_sizes[-1])}GB 时 (Uniform): Hit Rate = {hit_rates[-1]:.1%}, Latency = {latencies[-1]:.2f} ms")
print(f"Cache {int(cache_sizes[-1])}GB 时 (Zipf):    Hit Rate = {zipf_hit_rates[-1]:.1%}, Latency = {zipf_latencies[-1]:.2f} ms")

# ---- 格式化输出 ----
print("\n-- Latency Averages (ms) --\n")
for cache, lat in zip(cache_sizes, zipf_latencies):
    if cache == 0:
        print(f"      CPU avg={lat:.2f}")
    else:
        print(f"     {int(cache):>2}GB avg={lat:.2f}")