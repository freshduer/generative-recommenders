import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

"""
Embedding 和 KV Cache 动态平衡模拟器

场景：GPU 显存有限，需要同时存储 Embedding Cache 和 KV Cache
- Embedding: 用户/物品 embedding，访问模式是热点分布（Zipf）
- KV Cache: 用户历史序列，访问模式也是热点分布
- 两者共享 GPU 显存池，需要动态平衡

动态平衡的场景：
1. GPU 显存不足以同时存下所有需要的 embedding 和 KV cache
2. 不同时间段/请求类型，对两者的需求不同
3. 需要根据访问频率和延迟目标，动态调整分配比例
"""

# ---- 场景参数 ----
total_gpu_memory_gb = 40.0  # GPU 总显存

# Embedding 参数
total_emb_size_gb = 2000.0   # 总 embedding 大小
emb_hotspot_ratio = 0.1      # 10% 的 embedding 是热点
emb_hotspot_access_ratio = 0.9  # 90% 的请求访问热点 embedding
emb_lat_gpu_hit = 0.9        # Embedding GPU hit 延迟 (ms)
emb_lat_cpu_miss = 31.0      # Embedding CPU miss 延迟 (ms)

# KV Cache 参数
num_users = 2000              # 用户池大小

# 模型参数（用于计算 KV cache 大小）
user_length = 15000  # 用户序列长度（历史交互数量）
num_layers = 6       # Transformer 层数
num_heads = 4        # Attention head 数量
attention_dim = 64   # K 的维度（每个 head 的 attention 维度，通常是 qk_dim）
hidden_dim = 195     # V 的维度（每个 head 的 hidden 维度，根据实际测试调整，layer=6时约195可得178MB）
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

kv_hotspot_user_ratio = 0.1  # 10% 的用户是热点
kv_hotspot_access_ratio = 0.1  # 10% 的流量访问热点用户
kv_lat_base_compute = 10.0   # KV Cache GPU 计算延迟 (ms)
# 传输带宽参数
pcie_bandwidth_gbps = 24.0    # CPU→GPU 传输带宽 (GB/s)，PCIe Gen4 典型值
# 自动计算每个用户的传输延迟：延迟(ms) = 数据大小(MB) / 带宽(GB/s) * 1000 / 1024
kv_lat_transfer_per_user = (kv_size_per_user_mb / pcie_bandwidth_gbps) * 1000.0 / 1024.0
# Recompute 时间：当所有 KV cache 都不在显存时，整个 batch 的 recompute 时间
kv_lat_recompute_batch = 42.0  # 整个 batch 的 recompute 时间 (ms)，不管有多少个 miss，整个 batch 一起 recompute 需要 42ms

# 请求混合比例
# 不同请求类型对 embedding 和 KV cache 的需求不同
# 例如：推荐请求需要更多 embedding，对话请求需要更多 KV cache
request_type_ratio = {
    'embedding_heavy': 0.6,  # 60% 的请求主要需要 embedding（如推荐）
    'kv_heavy': 0.4,         # 40% 的请求主要需要 KV cache（如对话）
}

# 测试不同的分配比例
# emb_ratio: 分配给 embedding cache 的显存比例
# 5% 步长：0%, 5%, 10%, ..., 95%, 100%
emb_ratios = np.arange(0, 101, 5) / 100.0  # 从 0% 到 100%，步长 5%

print("=== Embedding 和 KV Cache 动态平衡模拟 ===")
print(f"Total GPU Memory: {total_gpu_memory_gb} GB")
print(f"Embedding Total Size: {total_emb_size_gb} GB")
print(f"KV Cache Total Size: {num_users * kv_size_per_user_gb:.1f} GB")
print(f"Request Mix: {request_type_ratio['embedding_heavy']:.0%} embedding-heavy, {request_type_ratio['kv_heavy']:.0%} KV-heavy")
print(f"\nKV Cache Model Parameters:")
print(f"  - User Length (seq_len): {user_length}")
print(f"  - Num Layers: {num_layers}, Num Heads: {num_heads}")
print(f"  - Attention Dim (K): {attention_dim}, Hidden Dim (V): {hidden_dim}")
print(f"  - KV Cache Size per User: {kv_size_per_user_mb:.2f} MB (auto-calculated)")
print(f"    - K cache: {k_size_per_user_bytes / (1024*1024):.2f} MB")
print(f"    - V cache: {v_size_per_user_bytes / (1024*1024):.2f} MB")
print(f"\nKV Cache Latency Parameters:")
print(f"  - Base Compute Time: {kv_lat_base_compute}ms")
print(f"  - PCIe Bandwidth: {pcie_bandwidth_gbps}GB/s")
print(f"  - Transfer Time per User: {kv_lat_transfer_per_user:.2f}ms (auto-calculated from {kv_size_per_user_mb:.2f}MB / {pcie_bandwidth_gbps}GB/s)")
print(f"  - Recompute Time per Batch: {kv_lat_recompute_batch}ms (entire batch recompute when KV cache not in GPU)")
transfer_time_all_miss = batch_size * kv_lat_transfer_per_user
transfer_strategy_all_miss = kv_lat_base_compute + transfer_time_all_miss
recompute_strategy_all_miss = kv_lat_recompute_batch
print(f"  - Strategy: Compare batch_transfer_time ({kv_lat_base_compute}ms + {batch_size}×{kv_lat_transfer_per_user:.2f}={transfer_strategy_all_miss:.2f}ms) vs batch_recompute_time ({recompute_strategy_all_miss}ms), choose min")
if transfer_strategy_all_miss < recompute_strategy_all_miss:
    print(f"  - Batch={batch_size}, all miss: {transfer_strategy_all_miss:.2f}ms (transfer)")
else:
    print(f"  - Batch={batch_size}, all miss: {recompute_strategy_all_miss}ms (recompute, no KV cache needed)")
print("-" * 70)

# 模拟参数
num_batches_sim = 1000
batch_size = 8
np.random.seed(42)

# 生成请求序列（混合类型）
request_types = []
for _ in range(num_batches_sim):
    if np.random.rand() < request_type_ratio['embedding_heavy']:
        request_types.append('embedding_heavy')
    else:
        request_types.append('kv_heavy')

# 生成 embedding 访问（Zipf 分布）
emb_hotspot_size_gb = total_emb_size_gb * emb_hotspot_ratio
emb_accesses = []
for _ in range(num_batches_sim * batch_size):
    if np.random.rand() < emb_hotspot_access_ratio:
        # 访问热点 embedding（简化：假设热点在前 10%）
        emb_id = np.random.randint(0, int(total_emb_size_gb * emb_hotspot_ratio))
    else:
        emb_id = np.random.randint(int(total_emb_size_gb * emb_hotspot_ratio), int(total_emb_size_gb))
    emb_accesses.append(emb_id)

# 生成 KV cache 访问（热点用户）
num_kv_hotspot_users = int(num_users * kv_hotspot_user_ratio)
kv_accesses = []
for _ in range(num_batches_sim * batch_size):
    if np.random.rand() < kv_hotspot_access_ratio:
        user_id = np.random.randint(0, num_kv_hotspot_users)
    else:
        user_id = np.random.randint(num_kv_hotspot_users, num_users)
    kv_accesses.append(user_id)

# 模拟不同分配比例
results = []
for emb_ratio in emb_ratios:
    kv_ratio = 1.0 - emb_ratio
    
    # 计算分配给 embedding 和 KV cache 的显存
    emb_memory_gb = total_gpu_memory_gb * emb_ratio
    kv_memory_gb = total_gpu_memory_gb * kv_ratio
    
    # Embedding cache 容量（能存多少 embedding）
    emb_cache_capacity_gb = emb_memory_gb
    emb_fraction_cached = min(1.0, emb_cache_capacity_gb / emb_hotspot_size_gb)
    
    # KV cache 容量（能存多少用户）
    kv_cache_capacity_users = int(kv_memory_gb / kv_size_per_user_gb) if kv_size_per_user_gb > 0 else 0
    
    # 模拟 LRU cache
    emb_cache = OrderedDict()  # 简化的 embedding cache
    kv_cache = OrderedDict()   # KV cache
    
    total_latency = 0.0
    emb_hits = 0
    kv_hits = 0
    total_emb_lookups = 0
    total_kv_lookups = 0
    
    batch_idx = 0
    for req_type in request_types:
        batch_emb_latency = 0.0
        batch_kv_latency = 0.0
        
        # 处理当前 batch 的 embedding 访问
        for i in range(batch_size):
            emb_id = emb_accesses[batch_idx * batch_size + i]
            total_emb_lookups += 1
            
            # 简化的 embedding cache 检查
            # 假设 embedding 按 ID 范围缓存（热点在前）
            if emb_id < emb_cache_capacity_gb:
                emb_hits += 1
                batch_emb_latency += emb_lat_gpu_hit
            else:
                batch_emb_latency += emb_lat_cpu_miss
        
        # 处理当前 batch 的 KV cache 访问
        batch_kv_miss_count = 0
        for i in range(batch_size):
            user_id = kv_accesses[batch_idx * batch_size + i]
            total_kv_lookups += 1
            
            if user_id in kv_cache:
                kv_cache.move_to_end(user_id)
                kv_hits += 1
            else:
                batch_kv_miss_count += 1
                if kv_cache_capacity_users > 0:
                    if len(kv_cache) >= kv_cache_capacity_users:
                        kv_cache.popitem(last=False)
                    kv_cache[user_id] = True
        
        # 对于整个 batch，比较两种策略：
        # 1. Transfer 策略：每个 miss 的用户都传输，总时间 = base_compute + miss_count * transfer_time_per_user
        # 2. Recompute 策略：整个 batch 一起 recompute，总时间 = kv_lat_recompute_batch（不需要 base_compute，因为不需要 KV cache）
        # 选择较小的那个
        if batch_kv_miss_count == 0:
            # 全部 hit，使用基础计算时间（KV cache 在 GPU）
            batch_kv_latency = kv_lat_base_compute
        else:
            transfer_time_total = batch_kv_miss_count * kv_lat_transfer_per_user
            transfer_strategy_latency = kv_lat_base_compute + transfer_time_total
            recompute_strategy_latency = kv_lat_recompute_batch
            
            if transfer_strategy_latency < recompute_strategy_latency:
                # 选择传输策略
                batch_kv_latency = transfer_strategy_latency
            else:
                # 选择 recompute 策略（整个 batch 一起 recompute，不需要 KV cache，所以不需要 base_compute）
                batch_kv_latency = recompute_strategy_latency
        
        # 根据请求类型计算延迟
        if req_type == 'embedding_heavy':
            # Embedding-heavy 请求：主要延迟来自 embedding
            batch_latency = batch_emb_latency / batch_size + batch_kv_latency * 0.3
        else:
            # KV-heavy 请求：主要延迟来自 KV cache
            batch_latency = batch_emb_latency / batch_size * 0.3 + batch_kv_latency
        
        total_latency += batch_latency
        batch_idx += 1
    
    avg_latency = total_latency / num_batches_sim
    emb_hit_rate = emb_hits / total_emb_lookups if total_emb_lookups > 0 else 0.0
    kv_hit_rate = kv_hits / total_kv_lookups if total_kv_lookups > 0 else 0.0
    
    results.append({
        'emb_ratio': emb_ratio,
        'kv_ratio': kv_ratio,
        'emb_memory_gb': emb_memory_gb,
        'kv_memory_gb': kv_memory_gb,
        'avg_latency': avg_latency,
        'emb_hit_rate': emb_hit_rate,
        'kv_hit_rate': kv_hit_rate,
    })
    
    # 每10%打印一次（0%, 10%, 20%, ..., 100%）
    if int(emb_ratio * 100) % 10 == 0:
        print(f"Emb={emb_ratio:.0%} ({emb_memory_gb:>4.0f}GB) | KV={kv_ratio:.0%} ({kv_memory_gb:>4.0f}GB) | "
              f"Lat={avg_latency:.2f}ms | Emb_Hit={emb_hit_rate:.1%} | KV_Hit={kv_hit_rate:.1%}")

# 提取结果
emb_ratios_plot = [r['emb_ratio'] for r in results]
latencies = [r['avg_latency'] for r in results]
emb_hit_rates = [r['emb_hit_rate'] for r in results]
kv_hit_rates = [r['kv_hit_rate'] for r in results]

# 找到最优分配点
optimal_idx = np.argmin(latencies)
optimal_emb_ratio = emb_ratios_plot[optimal_idx]
optimal_latency = latencies[optimal_idx]

# ---- 绘图 ----
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# 上图：延迟曲线
ax1.plot([r*100 for r in emb_ratios_plot], latencies, 'o-', color='tab:red', linewidth=2, label='Total Latency')
ax1.axvline(x=optimal_emb_ratio*100, color='green', linestyle='--', alpha=0.7, label=f'Optimal ({optimal_emb_ratio:.0%})')
ax1.axhline(y=optimal_latency, color='green', linestyle='--', alpha=0.7)
ax1.set_xlabel('Embedding Cache Memory Ratio (%)', fontsize=11)
ax1.set_ylabel('Average Latency (ms)', color='tab:red', fontsize=11, fontweight='bold')
ax1.tick_params(axis='y', labelcolor='tab:red')
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.set_title(f'Optimal Allocation: {optimal_emb_ratio:.0%} Embedding, {1-optimal_emb_ratio:.0%} KV Cache\n'
              f'Min Latency: {optimal_latency:.2f}ms', fontsize=12, fontweight='bold')

# 下图：命中率曲线
ax2.plot([r*100 for r in emb_ratios_plot], [h*100 for h in emb_hit_rates], 'o-', color='tab:blue', linewidth=2, label='Embedding Hit Rate')
ax2.plot([r*100 for r in emb_ratios_plot], [h*100 for h in kv_hit_rates], 's-', color='tab:orange', linewidth=2, label='KV Cache Hit Rate')
ax2.axvline(x=optimal_emb_ratio*100, color='green', linestyle='--', alpha=0.7)
ax2.set_xlabel('Embedding Cache Memory Ratio (%)', fontsize=11)
ax2.set_ylabel('Hit Rate (%)', fontsize=11, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend()
ax2.set_ylim(0, 105)

plt.tight_layout()
plt.savefig("figures/emb_kv_balance.png")
plt.show()

print(f"\n=== 结论 ===")
print(f"最优分配: Embedding={optimal_emb_ratio:.0%} ({results[optimal_idx]['emb_memory_gb']:.0f}GB), "
      f"KV Cache={1-optimal_emb_ratio:.0%} ({results[optimal_idx]['kv_memory_gb']:.0f}GB)")
print(f"最小延迟: {optimal_latency:.2f}ms")
print(f"此时 Embedding Hit Rate: {emb_hit_rates[optimal_idx]:.1%}")
print(f"此时 KV Cache Hit Rate: {kv_hit_rates[optimal_idx]:.1%}")
