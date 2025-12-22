import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

# ---- 核心场景参数 ----
batch_size = 8
num_users = 200  # 用户池

# 模型参数（用于计算 KV cache 大小）
user_length = 5000  # 用户序列长度（历史交互数量）
num_layers = 3       # Transformer 层数
num_heads = 4        # Attention head 数量
attention_dim = 64   # K 的维度（每个 head 的 attention 维度，通常是 qk_dim）
hidden_dim = 1024     # V 的维度（每个 head 的 hidden 维度，通常是 linear_dim）
kv_dtype_bytes = 2   # KV cache 数据类型大小（2=float16, 4=float32）

# 自动计算每个用户的 KV cache 大小
# 注意：K 和 V 的维度可能不同！
# K cache: seq_len * num_layers * num_heads * attention_dim * dtype_bytes
# V cache: seq_len * num_layers * num_heads * hidden_dim * dtype_bytes
k_size_per_user_bytes = user_length * num_layers * num_heads * attention_dim * kv_dtype_bytes
v_size_per_user_bytes = user_length * num_layers * num_heads * hidden_dim * kv_dtype_bytes
kv_size_per_user_bytes = k_size_per_user_bytes + v_size_per_user_bytes
kv_size_per_user_mb = kv_size_per_user_bytes / (1024.0 * 1024.0)
kv_size_per_user_gb = kv_size_per_user_mb / 1024.0

# 延迟参数 (根据 benchmark 数据反推)
# Benchmark: 0% cache=70ms, 50% cache=43ms, 75% cache=25ms
# 每个用户 KV Cache = 180MB，batch=8 时总数据量 = 1.44GB
# 
# 延迟组成：
# - latency_base_compute: GPU 计算时间（当数据已在 GPU 时）
# - latency_transfer_per_user: 每个用户从 CPU 传输到 GPU 的时间（自动计算）
#
# 计算逻辑：
#   batch_latency = compute_time + (miss_count × transfer_per_user)
#   例如：8个用户全miss = 10ms + 8×7.5ms = 70ms ✓
#
# 传输带宽参数
pcie_bandwidth_gbps = 32.0    # CPU→GPU 传输带宽 (GB/s)，PCIe Gen4 典型值
# 自动计算每个用户的传输延迟：延迟(ms) = 数据大小(MB) / 带宽(GB/s) * 1000 / 1024
latency_transfer_per_user = (kv_size_per_user_gb / pcie_bandwidth_gbps) * 1000.0

if user_length == 15000:
    if num_layers == 3:
        latency_recompute_batch = 42.0
    elif num_layers == 6:
        latency_recompute_batch = 63.0
    else:
        raise ValueError(f"Unsupported number of layers: {num_layers}")
elif user_length == 5000:
    if num_layers == 3:
        latency_recompute_batch = 15.0
    elif num_layers == 6:
        latency_recompute_batch = 27.5
    else:
        raise ValueError(f"Unsupported number of layers: {num_layers}")

latency_base_compute = latency_recompute_batch/3.5      # GPU 计算时间（数据已在 GPU）

# 流量分布参数
hotspot_user_ratio = 0.1    # 20% 的用户是热门用户
hotspot_access_ratio = 0.1  # 50% 的流量访问热门用户

# 显存配置
gpu_memory_sizes = list(range(0, 82, 2))  # 0 到 80GB，间隔 2GB

# ---- 流量生成：支持热点分布 ----
def generate_traffic_with_hotspots(num_users, batch_size, num_batches, hotspot_user_ratio, hotspot_access_ratio):
    """
    生成带热点分布的流量
    - hotspot_user_ratio: 热门用户比例（如 0.2 表示 20% 的用户是热门）
    - hotspot_access_ratio: 热门用户流量占比（如 0.5 表示 50% 的流量访问热门用户）
    """
    num_hotspot_users = int(num_users * hotspot_user_ratio)
    hotspot_users = set(range(num_hotspot_users))
    
    traffic = []
    for _ in range(num_batches * batch_size):
        # 决定访问热门用户还是非热门用户
        if np.random.rand() < hotspot_access_ratio:
            # 访问热门用户（均匀分布）
            user_id = np.random.randint(0, num_hotspot_users)
        else:
            # 访问非热门用户（均匀分布）
            user_id = np.random.randint(num_hotspot_users, num_users)
        traffic.append(user_id)
    
    return np.array(traffic).reshape(num_batches, batch_size)

# ---- 模拟主循环 ----
num_hotspot_users = int(num_users * hotspot_user_ratio)
print(f"=== KV Cache Simulation with Hotspot Distribution ===")
print(f"Total Users: {num_users}")
print(f"Hotspot Users: {num_hotspot_users} ({hotspot_user_ratio:.0%} of total)")
print(f"Hotspot Traffic Ratio: {hotspot_access_ratio:.0%}")
print(f"\nModel Parameters:")
print(f"  - User Length (seq_len): {user_length}")
print(f"  - Num Layers: {num_layers}, Num Heads: {num_heads}")
print(f"  - Attention Dim (K): {attention_dim}, Hidden Dim (V): {hidden_dim}")
print(f"  - KV Cache Size per User: {kv_size_per_user_mb:.2f} MB (auto-calculated)")
print(f"    - K cache: {k_size_per_user_bytes / (1024*1024):.2f} MB")
print(f"    - V cache: {v_size_per_user_bytes / (1024*1024):.2f} MB")
print(f"Total Data Needed: {num_users * kv_size_per_user_gb:.1f} GB")
print(f"\nLatency Parameters:")
print(f"  - GPU Compute Time: {latency_base_compute}ms (data already in GPU)")
print(f"  - PCIe Bandwidth: {pcie_bandwidth_gbps}GB/s")
print(f"  - Transfer Time per User: {latency_transfer_per_user:.2f}ms (CPU→GPU, auto-calculated from {kv_size_per_user_mb:.2f}MB / {pcie_bandwidth_gbps}GB/s)")
print(f"  - Recompute Time per Batch: {latency_recompute_batch}ms (entire batch recompute when KV cache not in GPU)")
print(f"  - Strategy: Compare batch_transfer_time ({batch_size}×{latency_transfer_per_user:.2f}={batch_size*latency_transfer_per_user:.2f}ms) vs batch_recompute_time ({latency_recompute_batch}ms), choose min")
transfer_time_all_miss = batch_size * latency_transfer_per_user
transfer_strategy_all_miss = latency_base_compute + transfer_time_all_miss
recompute_strategy_all_miss = latency_recompute_batch
if transfer_strategy_all_miss < recompute_strategy_all_miss:
    print(f"  - Batch={batch_size}, all miss: {latency_base_compute}ms + {transfer_time_all_miss:.2f}ms (transfer) = {transfer_strategy_all_miss:.2f}ms")
else:
    print(f"  - Batch={batch_size}, all miss: {recompute_strategy_all_miss}ms (recompute, no KV cache needed)")
print("-" * 60)

avg_latencies = []
hit_rates = []
capacity_users_list = []  # 保存每个配置的容量信息

# 生成带热点分布的流量
num_batches_sim = 2000
np.random.seed(42)
traffic_batches = generate_traffic_with_hotspots(num_users, batch_size, num_batches_sim, hotspot_user_ratio, hotspot_access_ratio)

for gpu_mem in gpu_memory_sizes:
    # 容量计算
    capacity_users = int(gpu_mem / kv_size_per_user_gb)
    
    # 模拟 LRU Cache
    gpu_cache = OrderedDict()
    
    total_latency = 0.0
    total_hits = 0
    total_lookups = 0
    total_transfers = 0
    total_recomputes = 0
    
    for batch in traffic_batches:
        batch_miss_count = 0
        
        for user_id in batch:
            total_lookups += 1
            
            if user_id in gpu_cache:
                # HIT
                gpu_cache.move_to_end(user_id)
                total_hits += 1
            else:
                # MISS
                batch_miss_count += 1
                
                # 如果有位置且没满
                if capacity_users > 0:
                    if len(gpu_cache) >= capacity_users:
                        gpu_cache.popitem(last=False) # 踢出最早的
                    gpu_cache[user_id] = True
        
        # 计算当前 batch 的延迟
        # 对于整个 batch，比较两种策略：
        # 1. Transfer 策略：每个 miss 的用户都传输，总时间 = base_compute + miss_count * transfer_time_per_user
        # 2. Recompute 策略：整个 batch 一起 recompute，总时间 = latency_recompute_batch（不需要 base_compute，因为不需要 KV cache）
        # 选择较小的那个
        if batch_miss_count == 0:
            # 全部 hit，使用基础计算时间（KV cache 在 GPU）
            batch_latency = latency_base_compute
        else:
            transfer_time_total = batch_miss_count * latency_transfer_per_user
            transfer_strategy_latency = latency_base_compute + transfer_time_total
            recompute_strategy_latency = latency_recompute_batch
            
            if transfer_strategy_latency < recompute_strategy_latency:
                # 选择传输策略
                batch_latency = transfer_strategy_latency
                total_transfers += batch_miss_count
            else:
                # 选择 recompute 策略（整个 batch 一起 recompute，不需要 KV cache，所以不需要 base_compute）
                batch_latency = recompute_strategy_latency
                total_recomputes += 1  # 整个 batch 算一次 recompute
        
        total_latency += batch_latency
        
    avg_lat = total_latency / num_batches_sim
    avg_hit_rate = total_hits / total_lookups
    total_misses = total_lookups - total_hits
    transfer_ratio = total_transfers / total_misses if total_misses > 0 else 0.0
    
    avg_latencies.append(avg_lat)
    hit_rates.append(avg_hit_rate)
    capacity_users_list.append(capacity_users)
    
    # 格式化输出
    strategy_str = f"T:{transfer_ratio:.0%}" if batch_size * latency_transfer_per_user < latency_recompute_batch else f"R:{1-transfer_ratio:.0%}"
    print(f"GPU {gpu_mem:>3}GB | Cap {capacity_users:>4} ({capacity_users/num_users:.1%}) | Hit {avg_hit_rate:.1%} | Lat {avg_lat:.2f} ms | {strategy_str}")

# ---- 绘图 ----
fig, ax1 = plt.subplots(figsize=(10, 6))

# 延迟曲线
color = 'tab:red'
ax1.set_xlabel('GPU Memory Size (GB)')
ax1.set_ylabel('Avg Batch Latency (ms)', color=color, fontweight='bold')
ax1.plot(gpu_memory_sizes, avg_latencies, 'o-', color=color, linewidth=2, label='Latency')
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim(0, 80)
ax1.grid(True, alpha=0.3)

# 理论线性参考线 (虚线)
# 理论上 Latency = 70 - 60 * (Mem / Total_Data)
total_data_gb = num_users * kv_size_per_user_gb
theoretical_lat = [max(10, 70 - 60 * (m / total_data_gb)) for m in gpu_memory_sizes]
ax1.plot(gpu_memory_sizes, theoretical_lat, '--', color='gray', alpha=0.5, label='Theoretical Linear')

# 命中率
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Cache Hit Rate (%)', color=color, fontweight='bold')
ax2.plot(gpu_memory_sizes, [h*100 for h in hit_rates], 'x--', color=color, label='Hit Rate')
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim(0, 105)

plt.title(f'KV Cache Performance with Hotspot Distribution\n({num_users} Users, {hotspot_user_ratio:.0%} Hotspot Users, {hotspot_access_ratio:.0%} Hotspot Traffic)', fontsize=12)
plt.tight_layout()
plt.savefig("figures/kv_cache_hotspot.png")
plt.show()

# ---- 格式化输出 ----
print("\n-- Latency Averages (ms) --\n")
for gpu_mem, lat, hit_rate, cap_users in zip(gpu_memory_sizes, avg_latencies, hit_rates, capacity_users_list):
    if gpu_mem == 0:
        print(f"      CPU avg={lat:.2f}")
    else:
        print(f"     {int(gpu_mem):>2}GB avg={lat:.2f} | Capacity: {cap_users:>4} users | Hit Rate: {hit_rate:.1%}")