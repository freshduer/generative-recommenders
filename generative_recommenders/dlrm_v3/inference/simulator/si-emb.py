import numpy as np
import matplotlib.pyplot as plt

# ---- 场景参数 ----
# Batch 和请求参数（这些参数用于说明延迟值对应的配置）
batch_size = 8              # Batch 大小
sequence_length = 14000     # Sequence length (1.4w)

# 延迟锚点 (ms)
# 注意：这些延迟值已经考虑了 batch_size=8 和 sequence_length=1.4w 的情况
# CPU miss: lookup 10ms + H2D transfer 21ms = 31ms
# GPU replicated: lookup 0.9ms
lat_gpu_hit = 0.9    # GPU hit 延迟（batch_size=8, seq_len=1.4w）
lat_cpu_miss = 31.0  # CPU miss 延迟（batch_size=8, seq_len=1.4w: 10ms lookup + 21ms H2D transfer）

# 数据分布参数
total_emb_size_gb = 12000.0   # 2TB (20亿行 * 1KB)
hotspot_ratio = 0.1          # 10% items
hotspot_access_ratio = 0.9   # 90% requests

hotspot_size_gb = total_emb_size_gb * hotspot_ratio # 200 GB

# 硬件配置
num_gpus = 8                # GPU 数量（例如 8 张卡）
gpu_memory_per_card = 82    # 每张 GPU 卡的显存大小 (GB)
cpu_memory_gb = 2048        # CPU 内存大小 (GB)，None 表示无限制
# 总显存 = num_gpus * gpu_memory_per_card
total_gpu_memory_gb = num_gpus * gpu_memory_per_card

# 变量: GPU Cache 从 0GB 到 80GB，间隔 2GB（单卡显存大小）
cache_sizes = np.array(list(range(0, 82, 2)))

# ---- 计算逻辑 ----
hit_rates = []
latencies = []

# 对比组：假设热点不是均匀分布，而是更集中的 Zipf (幂律) 分布
# 这种情况下，头部 20% 的热点可能占据了 80% 的热点流量
zipf_hit_rates = []
zipf_latencies = []

# CPU内存限制：如果CPU内存有限，可能无法存储所有embedding
# 假设CPU内存主要用于存储不在GPU缓存中的embedding
cpu_available_for_emb = cpu_memory_gb if cpu_memory_gb is not None else total_emb_size_gb
cpu_can_store_all = cpu_available_for_emb >= total_emb_size_gb

print(f"=== Embedding Cache Simulation ===")
print(f"\nRequest Parameters (for latency baseline):")
print(f"  - Batch Size: {batch_size}")
print(f"  - Sequence Length: {sequence_length}")
print(f"\nHardware Configuration:")
print(f"  - Number of GPUs: {num_gpus}")
print(f"  - GPU Memory per Card: {gpu_memory_per_card} GB")
print(f"  - Total GPU Memory: {total_gpu_memory_gb} GB ({num_gpus} × {gpu_memory_per_card} GB)")
if cpu_memory_gb is not None:
    print(f"  - CPU Memory: {cpu_memory_gb} GB")
    if cpu_can_store_all:
        print(f"    → CPU can store all embeddings ({total_emb_size_gb} GB)")
    else:
        print(f"    → CPU can only store {cpu_available_for_emb} GB / {total_emb_size_gb} GB embeddings")
        print(f"    → {total_emb_size_gb - cpu_available_for_emb} GB must be stored elsewhere (e.g., disk)")
else:
    print(f"  - CPU Memory: Unlimited")
print(f"\nData Distribution:")
print(f"  - Total Embedding Size: {total_emb_size_gb} GB")
print(f"  - Hotspot Size: {hotspot_size_gb} GB ({hotspot_ratio:.0%} of total)")
print(f"  - Hotspot Access Ratio: {hotspot_access_ratio:.0%}")
print(f"\nLatency Parameters (batch_size={batch_size}, seq_len={sequence_length}):")
print(f"  - GPU Hit: {lat_gpu_hit} ms (GPU replicated lookup)")
print(f"  - CPU Miss: {lat_cpu_miss} ms (10ms lookup + 21ms H2D transfer)")
print("-" * 60)

for cache in cache_sizes:
    # 考虑多GPU：总显存 = num_gpus * cache（单卡显存）
    total_cache_gb = num_gpus * cache
    
    # --- 1. 均匀热点模型 (Uniform Hotspot) ---
    # 算出 cache 能装下百分之多少的热点
    fraction_hotspot_cached = min(1.0, total_cache_gb / hotspot_size_gb)
    
    # 命中率 = 流量落入热点的概率 * 热点在缓存里的概率
    hit_rate = hotspot_access_ratio * fraction_hotspot_cached
    hit_rates.append(hit_rate)
    
    # 延迟计算：考虑 batch 中混合 hit/miss 的情况
    # 在一个 batch 中，有 hit_rate 比例的 embedding 是 hit，1-hit_rate 比例是 miss
    
    # miss 时的延迟：lat_cpu_miss 是固定的基准值（从 CPU 内存到 GPU 的传输时间）
    # 这个延迟值已经考虑了 batch_size 和 sequence_length，不应该因为 total_emb_size_gb 而改变
    # 假设：即使 CPU 内存有限，通过某种缓存策略（如 LRU），miss 的数据仍然在 CPU 内存中
    # 因此 miss 延迟始终是 lat_cpu_miss
    miss_latency = lat_cpu_miss
    
    # 延迟计算：lat_gpu_hit 和 lat_cpu_miss 已经是 batch 级别的延迟
    # 当 hit_rate 为 h 时，平均延迟 = h * lat_gpu_hit + (1-h) * miss_latency
    lat = lat_gpu_hit * hit_rate + miss_latency * (1 - hit_rate)
    latencies.append(lat)
    
    # --- 2. Zipf 优化模型 (更接近真实推荐系统) ---
    # 假设：虽然 Cache 只占热点的 20%，但这 20% 是"热点中的热点"
    # 经验法则：前 20% 的热点数据往往承载了热点内部 60%-70% 的流量
    # 调整幂律指数：0.6 比 0.4 更保守，意味着需要更多 cache 才能获得高命中率
    zipf_exponent = 0.6  # 幂律指数，越小越激进（需要更少 cache），越大越保守（需要更多 cache）
    zipf_scaling = fraction_hotspot_cached ** zipf_exponent  # 简单的幂律模拟
    zipf_hit_rate = hotspot_access_ratio * min(1.0, zipf_scaling)
    zipf_hit_rates.append(zipf_hit_rate)
    
    # 使用相同的延迟计算逻辑
    zipf_lat = lat_gpu_hit * zipf_hit_rate + miss_latency * (1 - zipf_hit_rate)
    zipf_latencies.append(zipf_lat)

# ---- 绘图 ----
plt.figure(figsize=(12, 6))

# 绘制均匀分布线
plt.plot(cache_sizes, latencies, 'o-', label='Uniform Hotspot (Worst Case)', color='tab:red', linewidth=2)
# 绘制 Zipf 分布线
plt.plot(cache_sizes, zipf_latencies, 'o--', label='Zipf Hotspot (Real World)', color='tab:blue', linewidth=2)

plt.axhline(y=31.0, color='gray', linestyle=':', label='Baseline (Pure CPU)')
plt.axhline(y=0.9, color='green', linestyle=':', label='Ideal (Pure GPU)')

# 标题和标签
x_label = f'GPU Cache Size per Card (GB) (Total: {num_gpus}×{gpu_memory_per_card}GB = {total_gpu_memory_gb}GB)' if num_gpus > 1 else 'GPU Cache Size (GB)'
cpu_mem_str = f", CPU: {cpu_memory_gb}GB" if cpu_memory_gb is not None else ", CPU: Unlimited"
title = f'Batch Size {batch_size}, Seq Len {sequence_length} Latency vs GPU Cache Size\n({num_gpus} GPUs, Hotspot Size: {hotspot_size_gb}GB{cpu_mem_str})'
plt.title(title, fontsize=11)
plt.xlabel(x_label, fontsize=10)
plt.ylabel('Avg Latency (ms)', fontsize=10)
plt.grid(True, alpha=0.3)
plt.legend()

# 标注关键点
plt.annotate(f"{latencies[-1]:.1f}ms", (cache_sizes[-1], latencies[-1]), textcoords="offset points", xytext=(0,10), ha='center', color='red')
plt.annotate(f"{zipf_latencies[-1]:.1f}ms", (cache_sizes[-1], zipf_latencies[-1]), textcoords="offset points", xytext=(0,-15), ha='center', color='blue')

plt.savefig("figures/latency_simulation.png")
plt.show()

total_cache_max = num_gpus * cache_sizes[-1]
print(f"\nCache {int(cache_sizes[-1])}GB/card ({int(total_cache_max)}GB total) 时 (Uniform): Hit Rate = {hit_rates[-1]:.1%}, Latency = {latencies[-1]:.2f} ms")
print(f"Cache {int(cache_sizes[-1])}GB/card ({int(total_cache_max)}GB total) 时 (Zipf):    Hit Rate = {zipf_hit_rates[-1]:.1%}, Latency = {zipf_latencies[-1]:.2f} ms")

# ---- 格式化输出 ----
print("\n-- Latency Averages (ms) with Hit Rates --\n")
for cache, lat, hit_rate in zip(cache_sizes, zipf_latencies, zipf_hit_rates):
    if cache == 0:
        print(f"      CPU avg={lat:.2f} (Hit Rate: {hit_rate:.1%})")
    else:
        total_cache = num_gpus * cache
        fraction_hotspot = min(1.0, total_cache / hotspot_size_gb)
        print(f"     {int(cache):>2}GB/card ({int(total_cache)}GB total) avg={lat:.2f} | Hit Rate: {hit_rate:.1%} | Cache covers {fraction_hotspot:.1%} of hotspot")