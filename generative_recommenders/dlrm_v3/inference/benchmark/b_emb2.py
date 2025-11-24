import os
import sys
import time
import torch
import torch.distributed as dist
from typing import Dict, List
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from torchrec.modules.embedding_configs import EmbeddingConfig, DataType
from torchrec.distributed.planner.types import ParameterConstraints

# ============================================================================
# 1. 路径与环境设置
# ============================================================================
# 请确保此路径下包含 custom_sharding.py
PROJECT_ROOT = os.path.abspath("/home/comp/cswjyu/orion-yuwenjun/generative-recommenders/generative_recommenders/dlrm_v3/inference")
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

try:
    from custom_sharding import CustomEmbeddingCollection, ShardingType
except ImportError as e:
    print(f"Error: 无法导入 custom_sharding。请确认路径 {PROJECT_ROOT} 是否正确。")
    raise e

# ============================================================================
# 2. 全局常量配置
# ============================================================================
NUM_EMBEDDINGS = 5_000_000  # 模拟 500万 items
EMBEDDING_DIM = 256
HOT_RATIO = 0.10            # 10% 的热点 ID
ACCESS_RATIO = 0.90         # 90% 的访问集中在热点 ID
WARMUP_STEPS = 5
BENCH_STEPS = 50

# 定义特征组及其模拟长度（基于你的日志分析）
FEATURE_GROUPS = {
    # A类：单值上下文 (Length = 1)
    "scalars": {
        "names": ["v", "d"],
        "avg_len": 1,
        "variance": 0
    },
    # B类：超长用户行为序列 (Length ~ 15,000)
    "history": {
        "names": [
            "uih_post_id", "uih_action_time", "uih_weight", 
            "uih_owner_id", "uih_watchtime", "uih_surface_type", 
            "uih_video_length", "viewer_id", "dummy_contexual"
        ],
        "avg_len": 15000,
        "variance": 1000  # 长度在 14000 ~ 16000 之间波动
    },
    # C类：候选物品特征 (Length ~ 5)
    "items": {
        "names": [
            "item_post_id", "item_owner_id", "item_surface_type", 
            "item_video_length", "item_action_weight", 
            "item_target_watchtime", "item_query_time"
        ],
        "avg_len": 5,
        "variance": 2
    }
}

# ============================================================================
# 3. 数据生成逻辑 (模拟真实 Jagged 数据)
# ============================================================================
def generate_skewed_indices(num_embeddings: int, total_count: int, device: torch.device) -> torch.Tensor:
    """生成符合 Zipfian/热点分布的 Embedding ID"""
    if total_count == 0:
        return torch.empty(0, dtype=torch.long, device=device)
        
    num_hot = int(num_embeddings * HOT_RATIO)
    # 随机决定每个位置是否访问热点 ID
    is_hot_access = torch.rand(total_count, device=device) < ACCESS_RATIO
    num_hot_access = is_hot_access.sum().item()
    num_cold_access = total_count - num_hot_access
    
    indices = torch.zeros(total_count, dtype=torch.long, device=device)
    
    if num_hot_access > 0:
        indices[is_hot_access] = torch.randint(0, num_hot, (num_hot_access,), device=device)
    if num_cold_access > 0:
        indices[~is_hot_access] = torch.randint(num_hot, num_embeddings, (num_cold_access,), device=device)
    
    return indices

def get_kjt_batch(batch_size: int, device: torch.device, feature_vocab_sizes: Dict[str, int]) -> KeyedJaggedTensor:
    """构造一个完全模拟 HSTU 输入结构的 KeyedJaggedTensor

    修复：不同特征对应的 embedding 表大小不同，不能统一使用 NUM_EMBEDDINGS，
    否则会对较小表（如 video_meta）生成越界 ID 导致 CUDA gather 断言失败。
    """
    all_keys = []
    all_lengths_list = []
    all_values_list = []

    for group_name, spec in FEATURE_GROUPS.items():
        base_len = spec["avg_len"]
        var = spec["variance"]
        
        for feat_name in spec["names"]:
            all_keys.append(feat_name)
            
            # 生成 Lengths
            if var > 0:
                # 随机长度: base_len +/- var
                low = max(1, base_len - var)
                high = base_len + var
                lengths = torch.randint(low, high, (batch_size,), dtype=torch.int32, device=device)
            else:
                # 固定长度
                lengths = torch.full((batch_size,), base_len, dtype=torch.int32, device=device)
            
            # 计算该特征在这个 Batch 里总共有多少个值
            total_vals = lengths.sum().item()
            
            # 生成 Values (IDs) —— 按特征所属表的 vocab size
            vocab_size = feature_vocab_sizes.get(feat_name, NUM_EMBEDDINGS)
            values = generate_skewed_indices(vocab_size, total_vals, device)
            
            all_lengths_list.append(lengths)
            all_values_list.append(values)

    # 拼装 KJT
    # values 和 lengths 必须展平
    final_values = torch.cat(all_values_list)
    final_lengths = torch.cat(all_lengths_list)

    return KeyedJaggedTensor.from_lengths_sync(
        keys=all_keys,
        values=final_values,
        lengths=final_lengths
    ).to(device)

# ============================================================================
# 4. Benchmark 核心函数
# ============================================================================
def run_benchmark(rank: int, model: torch.nn.Module, batch_size: int, device: torch.device, feature_vocab_sizes: Dict[str, int]):
    # 生成数据 (模拟 IO/Preproc)
    input_kjt = get_kjt_batch(batch_size, device, feature_vocab_sizes)
    # print(f"get_kjt_batch done for batch_size:{batch_size}")
    
    # Warmup
    with torch.no_grad():
        for _ in range(WARMUP_STEPS):
            _ = model(input_kjt)
    # print(f"warmup done for batch_size:{batch_size}")
    
    dist.barrier()
    
    # Timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    torch.cuda.synchronize()
    start_event.record()
    
    with torch.no_grad():
        for i in range(BENCH_STEPS):
            # print(f"Rank {rank} - Step {i+1}/{BENCH_STEPS} for batch_size:{batch_size}")
            _ = model(input_kjt)
        
    end_event.record()
    torch.cuda.synchronize()
    
    total_time_ms = start_event.elapsed_time(end_event)
    avg_time_ms = total_time_ms / BENCH_STEPS
    qps = batch_size / (avg_time_ms / 1000.0)
    
    return avg_time_ms, qps

# ============================================================================
# 5. 主程序
# ============================================================================
def main():
    # 分布式初始化
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"
        
    # 使用 device_id 显式绑定本进程 GPU，避免 NCCL 映射不明确的警告
    # PyTorch>=2.1 支持 device_id 参数；若版本较低可忽略但需确保 set_device 已调用。
    try:
        dist.init_process_group(backend=backend, device_id=local_rank)
    except TypeError:
        # 旧版本不支持 device_id，退化为原始调用
        dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # 固定随机种子，保证各 rank 生成的 lengths 和 values 形状一致
    seed = 1337
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 打印环境信息
    if rank == 0:
        print(f"\nBenchmark | world_size={world_size} | device={device}")
        print(f"Data Pattern: HSTU-like Long Sequence (Avg Len ~15k)")
        print(f"Table size: {NUM_EMBEDDINGS} rows x {EMBEDDING_DIM} dim")
        print("-" * 80)

    # 这里的 Batch Size 比较小，因为每个样本包含 ~1.5万 个 token
    batch_sizes = [16, 32]
    
    # 测试模式
    sharding_modes = [
        ShardingType.ROW_WISE,
        ShardingType.TABLE_WISE,
        ShardingType.REPLICATED,
        ShardingType.CPU_OFFLOAD, # 如果显存不够可以开启
    ]

    for mode in sharding_modes:
        # --------------------------------------------------------------------
        # 定义 Embedding Tables 配置
        # 关键策略：模拟真实的表共享 (Table Sharing)
        # --------------------------------------------------------------------
        tables_config: Dict[str, EmbeddingConfig] = {
            # 1. 帖子/视频 ID 表：同时处理超长历史(uih)和候选集(item)
            "table_post_id": EmbeddingConfig(
                num_embeddings=NUM_EMBEDDINGS,
                embedding_dim=EMBEDDING_DIM,
                name="table_post_id",
                data_type=DataType.FP16,
                feature_names=["uih_post_id", "item_post_id"] 
            ),
            # 2. 用户/作者 ID 表
            "table_user_id": EmbeddingConfig(
                num_embeddings=NUM_EMBEDDINGS,
                embedding_dim=EMBEDDING_DIM,
                name="table_user_id",
                data_type=DataType.FP16,
                feature_names=["uih_owner_id", "item_owner_id", "viewer_id"]
            ),
            # 3. 视频属性表 (时长、类型等)
            "table_video_meta": EmbeddingConfig(
                num_embeddings=NUM_EMBEDDINGS // 10, # 属性类 ID 通常少一点
                embedding_dim=EMBEDDING_DIM,
                name="table_video_meta",
                data_type=DataType.FP16,
                feature_names=[
                    "uih_surface_type", "item_surface_type",
                    "uih_video_length", "item_video_length"
                ]
            ),
             # 4. 上下文与统计特征表
            "table_context_stats": EmbeddingConfig(
                num_embeddings=NUM_EMBEDDINGS,
                embedding_dim=EMBEDDING_DIM,
                name="table_context_stats",
                data_type=DataType.FP16,
                feature_names=[
                    "v", "d", "dummy_contexual",
                    "uih_action_time", "uih_weight", "uih_watchtime",
                    "item_action_weight", "item_target_watchtime", "item_query_time"
                ]
            ),
        }

        # 设置 Sharding 约束
        constraints = {
            name: ParameterConstraints(sharding_types=[mode.value])
            for name in tables_config.keys()
        }

        # 初始化你的 Custom Model
        try:
            model = CustomEmbeddingCollection(
                table_config=tables_config,
                constraints=constraints,
                device=device,
            )
        except Exception as e:
            if rank == 0:
                print(f"Skipping mode {mode.value} due to init error: {e}")
            continue

        # 打印 Sharding 布局信息
        if rank == 0:
            print(f"\n[Placement | mode={mode.value}]")
            for name, tbl in model.tables.items():
                dev_str = str(tbl.weight.weight.device)
                print(f"  {name}: rows=({tbl.row_start},{tbl.row_end}) local={tbl.row_end - tbl.row_start} "
                      f"sharding={tbl.sharding_type.value} device={dev_str}")
            print(f"\nMode {mode.value.upper()} results:")
            print(f"{'Batch Size':<12} | {'Latency ms':<12} | {'QPS':<10} | Note")
            print("-" * 60)


        # 构造特征 -> vocab_size 映射，供数据生成使用
        feature_vocab_sizes: Dict[str, int] = {}
        for tname, cfg in tables_config.items():
            for feat in cfg.feature_names:
                feature_vocab_sizes[feat] = cfg.num_embeddings

        # 运行测试循环
        for bs in batch_sizes:
            try:
                dist.barrier()
                latency, qps = run_benchmark(rank, model, bs, device, feature_vocab_sizes)
                
                if rank == 0:
                    note = mode.value
                    print(f"{bs:<12} | {latency:<12.4f} | {int(qps):<10,} | {note}")
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    if rank == 0:
                        print(f"{bs:<12} | {'OOM':<12} | {'0':<10} | {mode.value}")
                    torch.cuda.empty_cache()
                else:
                    raise e

        del model
        torch.cuda.empty_cache()

    dist.destroy_process_group()

if __name__ == "__main__":
    main()