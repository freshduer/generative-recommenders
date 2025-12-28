import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import sys
import os

# ---- 核心场景参数 ----
batch_size = 8
# num_users 将在循环中设置

# 模型参数（用于计算 KV cache 大小）
user_length = 15000  # 用户序列长度（历史交互数量）
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

# 硬件配置
num_gpus = 8                # GPU 数量（例如 8 张卡）
gpu_memory_per_card = 82    # 每张 GPU 卡的显存大小 (GB)
cpu_memory_gb = 2048        # CPU 内存大小 (GB)，None 表示无限制
# 总显存 = num_gpus * gpu_memory_per_card
total_gpu_memory_gb = num_gpus * gpu_memory_per_card

# 显存配置（单卡显存大小，用于扫描不同配置）
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

# ---- 模拟函数 ----
def run_simulation(num_users, output_file=None, append_mode=False):
    """
    运行 KV cache 模拟
    Args:
        num_users: 用户数量
        output_file: 输出文件路径，如果为None则输出到stdout
        append_mode: 是否以追加模式打开文件
    """
    # 保存原始的stdout
    original_stdout = sys.stdout
    
    # 如果指定了输出文件，重定向stdout
    if output_file:
        mode = 'a' if append_mode else 'w'
        f = open(output_file, mode, encoding='utf-8')
        sys.stdout = f
    else:
        f = None
    
    try:
        num_hotspot_users = int(num_users * hotspot_user_ratio)
        print(f"=== KV Cache Simulation with Hotspot Distribution ===")
        print(f"Total Users: {num_users}")
        print(f"Hotspot Users: {num_hotspot_users} ({hotspot_user_ratio:.0%} of total)")
        print(f"Hotspot Traffic Ratio: {hotspot_access_ratio:.0%}")
        print(f"\nHardware Configuration:")
        print(f"  - Number of GPUs: {num_gpus}")
        print(f"  - GPU Memory per Card: {gpu_memory_per_card} GB")
        print(f"  - Total GPU Memory: {total_gpu_memory_gb} GB ({num_gpus} × {gpu_memory_per_card} GB)")
        if cpu_memory_gb is not None:
            print(f"  - CPU Memory: {cpu_memory_gb} GB (limited)")
        else:
            print(f"  - CPU Memory: Unlimited")
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
            # 容量计算：考虑多GPU情况
            # gpu_mem 是单卡显存，总显存 = num_gpus * gpu_mem
            total_gpu_mem = num_gpus * gpu_mem
            capacity_users_gpu = int(total_gpu_mem / kv_size_per_user_gb)
            
            # CPU 内存容量（如果有限制）
            if cpu_memory_gb is not None:
                capacity_users_cpu = int(cpu_memory_gb / kv_size_per_user_gb)
            else:
                capacity_users_cpu = num_users  # 无限制，可以存储所有用户
            
            # 模拟 LRU Cache
            gpu_cache: OrderedDict[int, bool] = OrderedDict()  # GPU 缓存
            cpu_cache: OrderedDict[int, bool] = OrderedDict()  # CPU 缓存（如果CPU内存有限）
            
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
                        # GPU HIT
                        gpu_cache.move_to_end(user_id)
                        total_hits += 1
                    elif user_id in cpu_cache:
                        # CPU HIT（需要从CPU传输到GPU）
                        cpu_cache.move_to_end(user_id)
                        batch_miss_count += 1  # 需要传输，算作miss
                    else:
                        # MISS（既不在GPU也不在CPU）
                        batch_miss_count += 1
                        
                        # 尝试放入CPU缓存（如果CPU内存还有空间）
                        if capacity_users_cpu > 0:
                            if len(cpu_cache) >= capacity_users_cpu:
                                cpu_cache.popitem(last=False)  # 踢出最早的
                            cpu_cache[user_id] = True
                        
                        # 尝试放入GPU缓存（如果GPU显存还有空间）
                        if capacity_users_gpu > 0:
                            if len(gpu_cache) >= capacity_users_gpu:
                                gpu_cache.popitem(last=False)  # 踢出最早的
                            gpu_cache[user_id] = True
                
                # 计算当前 batch 的延迟
                # 对于整个 batch，比较两种策略：
                # 1. Transfer 策略：传输所有 miss 的用户，然后计算整个 batch（hit 的用户用 cache，miss 的用户用传输的数据）
                #    总时间 = base_compute + miss_count * transfer_time_per_user
                # 2. Recompute 策略：recompute 整个 batch（当 miss 多时，这比传输更划算）
                #    总时间 = latency_recompute_batch（不需要传输，直接 recompute）
                # 选择较小的那个
                if batch_miss_count == 0:
                    # 全部 hit，使用基础计算时间（KV cache 在 GPU，只需要 forward pass）
                    # 这是最优情况，延迟应该是最低的
                    batch_latency = latency_base_compute
                else:
                    # 有部分 miss，需要处理 miss 的用户
                    batch_hit_count = batch_size - batch_miss_count
                    hit_ratio = batch_hit_count / batch_size
                    miss_ratio = batch_miss_count / batch_size
                    
                    # Transfer 策略：传输所有 miss 的用户，然后计算整个 batch
                    # hit 的用户用 cache（只需要 forward pass），miss 的用户用传输的数据（也需要 forward pass）
                    # 由于整个 batch 是并行计算的，计算时间主要取决于 batch 大小，而不是 miss 比例
                    # 当有 cache 时，计算时间接近 base_compute（因为 hit 的部分很快）
                    transfer_time_total = batch_miss_count * latency_transfer_per_user
                    # 计算时间：当有部分命中时，计算时间应该接近 base_compute
                    # 因为 hit 的部分用 cache（很快），miss 的部分用传输的数据（也需要计算，但整个 batch 并行）
                    # 简化：计算时间 = base_compute（因为整个 batch 并行，主要取决于 batch 大小，而不是 miss 比例）
                    transfer_strategy_latency = latency_base_compute + transfer_time_total
                    
                    # Recompute 策略：recompute 整个 batch（当 miss 多时，这比传输更划算）
                    # 但是，hit 的部分不需要 recompute，所以延迟应该降低
                    # 简化模型：recompute 时间与 miss 比例相关
                    # 如果全部 miss，需要完整的 recompute；如果有部分 hit，只需要 recompute miss 的部分
                    # 更准确：recompute 时间 = base_compute * hit_ratio + recompute_time * miss_ratio
                    # 但考虑到 batch 并行，实际可能更接近：recompute_time * miss_ratio + base_compute * hit_ratio
                    # 或者更简单：如果 hit 比例高，recompute 时间应该明显降低
                    recompute_strategy_latency = latency_base_compute * hit_ratio + latency_recompute_batch * miss_ratio
                    
                    if transfer_strategy_latency < recompute_strategy_latency:
                        # 选择传输策略：传输 miss 的用户，hit 的用户用 cache
                        batch_latency = transfer_strategy_latency
                        total_transfers += batch_miss_count
                    else:
                        # 选择 recompute 策略：但考虑 hit 的部分不需要 recompute
                        batch_latency = recompute_strategy_latency
                        total_recomputes += 1  # 整个 batch 算一次 recompute
                
                total_latency += batch_latency
            
            avg_lat = total_latency / num_batches_sim
            avg_hit_rate = total_hits / total_lookups
            total_misses = total_lookups - total_hits
            transfer_ratio = total_transfers / total_misses if total_misses > 0 else 0.0
            
            avg_latencies.append(avg_lat)
            hit_rates.append(avg_hit_rate)
            capacity_users_list.append((capacity_users_gpu, capacity_users_cpu))
            
            # 格式化输出
            strategy_str = f"T:{transfer_ratio:.0%}" if batch_size * latency_transfer_per_user < latency_recompute_batch else f"R:{1-transfer_ratio:.0%}"
            total_gpu_mem_str = f"{total_gpu_mem}GB" if num_gpus > 1 else f"{gpu_mem}GB"
            cpu_mem_str = f"CPU:{capacity_users_cpu}" if cpu_memory_gb is not None else "CPU:∞"
            print(f"GPU {gpu_mem:>3}GB/card ({total_gpu_mem_str} total) | GPU Cap {capacity_users_gpu:>4} ({capacity_users_gpu/num_users:.1%}) | {cpu_mem_str} | Hit {avg_hit_rate:.1%} | Lat {avg_lat:.2f} ms | {strategy_str}")

        # ---- 格式化输出 ----
        print("\n-- Latency Averages (ms) --\n")
        for gpu_mem, lat, hit_rate, (cap_users_gpu, cap_users_cpu) in zip(gpu_memory_sizes, avg_latencies, hit_rates, capacity_users_list):
            if gpu_mem == 0:
                print(f"      CPU avg={lat:.2f}")
            else:
                total_gpu_mem = num_gpus * gpu_mem
                cpu_str = f" | CPU Cap: {cap_users_cpu}" if cpu_memory_gb is not None else " | CPU Cap: ∞"
                print(f"     {int(gpu_mem):>2}GB/card ({int(total_gpu_mem)}GB total) avg={lat:.2f} | GPU Capacity: {cap_users_gpu:>4} users{cpu_str} | Hit Rate: {hit_rate:.1%}")
    
    except Exception as e:
        # 如果出错，恢复stdout并重新抛出异常
        if f:
            sys.stdout = original_stdout
            f.close()
        raise e
    finally:
        # 恢复原始的stdout
        if f:
            sys.stdout = original_stdout
            f.close()
            print(f"Results saved to {output_file}")


# ---- 主循环：测试不同的 num_users ----
if __name__ == "__main__":
    # 创建输出目录（相对于脚本位置）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "results")
    os.makedirs(output_dir, exist_ok=True)
    
    # num_users 从 100 开始，间隔 200，到 5000
    num_users_list = list(range(100, 5001, 200))
    
    # 所有结果保存到一个文件
    output_file = os.path.join(output_dir, "kv_cache_sim_all_num_users.txt")
    
    print(f"Running simulations for {len(num_users_list)} different num_users values...")
    print(f"num_users range: {num_users_list[0]} to {num_users_list[-1]}, step=200")
    print(f"All results will be saved to: {output_file}\n")
    
    for i, num_users in enumerate(num_users_list):
        print(f"Running simulation for num_users={num_users}...")
        
        # 如果是第一次，使用写入模式；否则使用追加模式，并添加分隔符
        if i > 0:
            # 在追加模式下，先添加分隔符
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write("\n" + "=" * 80 + "\n")
                f.write("=" * 80 + "\n")
                f.write("\n")
        
        # 运行模拟，将输出写入文件
        run_simulation(num_users, output_file, append_mode=(i > 0))
        
        print(f"Completed num_users={num_users}\n")
    
    print(f"All simulations completed! Results saved to {output_file}")