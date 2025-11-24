import os
import torch
import torch.distributed as dist
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from typing import Dict
import sys

PROJECT_ROOT = os.path.abspath("/home/comp/cswjyu/orion-yuwenjun/generative-recommenders/generative_recommenders/dlrm_v3/inference")
sys.path.append(PROJECT_ROOT)

from custom_sharding import CustomEmbeddingCollection, ShardingType
from torchrec.distributed.planner.types import ParameterConstraints
# 引入 EmbeddingConfig 与 DataType
from torchrec.modules.embedding_configs import EmbeddingConfig
from torchrec.modules.embedding_configs import DataType  # 若你在 configs 中已重新导出，可省略

NUM_EMBEDDINGS = 10000000
EMBEDDING_DIM = 256
HOT_RATIO = 0.10
ACCESS_RATIO = 0.90
WARMUP_STEPS = 10
BENCH_STEPS = 100

def generate_skewed_indices(num_embeddings: int, batch_size: int, device: torch.device) -> torch.Tensor:
    num_hot = int(num_embeddings * HOT_RATIO)
    is_hot_access = torch.rand(batch_size, device=device) < ACCESS_RATIO
    num_hot_access = is_hot_access.sum().item()
    num_cold_access = batch_size - num_hot_access
    indices = torch.zeros(batch_size, dtype=torch.long, device=device)
    if num_hot_access > 0:
        indices[is_hot_access] = torch.randint(0, num_hot, (num_hot_access,), device=device)
    if num_cold_access > 0:
        indices[~is_hot_access] = torch.randint(num_hot, num_embeddings, (num_cold_access,), device=device)
    return indices

def get_kjt_batch(batch_size: int, device: torch.device) -> KeyedJaggedTensor:
    keys = ["feat_a", "feat_b"]
    idx_a = generate_skewed_indices(NUM_EMBEDDINGS, batch_size, device)
    idx_b = generate_skewed_indices(NUM_EMBEDDINGS, batch_size, device)
    values = torch.cat([idx_a, idx_b])
    lengths = torch.ones(batch_size * 2, dtype=torch.int32, device=device)
    return KeyedJaggedTensor.from_lengths_sync(keys=keys, values=values, lengths=lengths).to(device)

def run_benchmark(rank: int, model: torch.nn.Module, batch_size: int, device: torch.device):
    input_kjt = get_kjt_batch(batch_size, device)
    for _ in range(WARMUP_STEPS):
        _ = model(input_kjt)
    dist.barrier()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start_event.record()
    for _ in range(BENCH_STEPS):
        _ = model(input_kjt)
    end_event.record()
    torch.cuda.synchronize()
    total_time_ms = start_event.elapsed_time(end_event)
    avg_time_ms = total_time_ms / BENCH_STEPS
    qps = batch_size / (avg_time_ms / 1000.0)
    return avg_time_ms, qps

def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"
    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == 0:
        print(f"\nBenchmark | world_size={world_size} | device={device}")
        print(f"Skew: {HOT_RATIO*100:.1f}% items get {ACCESS_RATIO*100:.1f}% traffic")
        print(f"Table size: {NUM_EMBEDDINGS} rows x {EMBEDDING_DIM} dim")
        print("-" * 80)

    batch_sizes = [2048, 8192, 32768]
    sharding_modes = [
        ShardingType.ROW_WISE,
        ShardingType.TABLE_WISE,
        ShardingType.REPLICATED,
        ShardingType.CPU_OFFLOAD,
    ]

    for mode in sharding_modes:
        # 使用 EmbeddingConfig（示例与你提供的 movie_id 配置形式一致）
        tables_config: Dict[str, EmbeddingConfig] = {
            "table_a": EmbeddingConfig(
                num_embeddings=NUM_EMBEDDINGS,
                embedding_dim=EMBEDDING_DIM,
                name="table_a",
                data_type=DataType.FP16,
                feature_names=["feat_a"],
            ),
            "table_b": EmbeddingConfig(
                num_embeddings=NUM_EMBEDDINGS,
                embedding_dim=EMBEDDING_DIM,
                name="table_b",
                data_type=DataType.FP16,
                feature_names=["feat_b"],
            ),
        }

        # 通过 constraints 指定当前基准循环的 sharding 类型
        constraints = {
            name: ParameterConstraints(sharding_types=[mode.value])
            for name in tables_config.keys()
        }

        model = CustomEmbeddingCollection(
            table_config=tables_config,
            constraints=constraints,
            device=device,
        )

        if rank == 0:
            print(f"\n[Placement | mode={mode.value}]")
            for name, tbl in model.tables.items():
                dev_str = str(tbl.weight.weight.device)
                print(f"  {name}: rows=({tbl.row_start},{tbl.row_end}) local={tbl.row_end - tbl.row_start} "
                      f"sharding={tbl.sharding_type.value} device={dev_str}")
            print(f"\nMode {mode.value.upper()} results:")
            print(f"{'Batch Size':<12} | {'Latency ms':<12} | {'QPS':<10} | Note")
            print("-" * 60)

        for bs in batch_sizes:
            dist.barrier()
            latency, qps = run_benchmark(rank, model, bs, device)
            if rank == 0:
                if mode == ShardingType.CPU_OFFLOAD:
                    note = "PCIe"
                elif mode == ShardingType.ROW_WISE:
                    note = "AllReduce"
                elif mode == ShardingType.TABLE_WISE:
                    note = "LoadImb"
                elif mode == ShardingType.REPLICATED:
                    note = "Mem BW"
                else:
                    note = "-"
                print(f"{bs:<12} | {latency:<12.4f} | {int(qps):<10,} | {note}")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    dist.destroy_process_group()

if __name__ == "__main__":
    main()