import os
from typing import Dict, List
import torch
import torch.distributed as dist
import torchrec
from torchrec import KeyedJaggedTensor
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec import EmbeddingBagConfig, PoolingType
from torchrec.distributed.embeddingbag import EmbeddingBagCollectionSharder
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
from torchrec.distributed.types import (
    ModuleSharder,
    ShardingEnv,
    ShardingPlan,
    ShardingType,
)
from torchrec.distributed.planner.types import ParameterConstraints
from torchrec.models.dlrm import DLRM, DLRM_DCN, DLRM_Projection, DLRMTrain
from torchrec.distributed.model_parallel import (
    DistributedModelParallel,
)
# ---------------------------
# 1. Distributed initializer
# ---------------------------
def init_distributed(rank: int, world_size: int, backend: str) -> dist.ProcessGroup:
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
    )
    return dist.group.WORLD


# ---------------------------
# 2. Generate parameter constraints
# ---------------------------
def gen_constraints(sharding_type: ShardingType,
    large_table_cnt: int,
    small_table_cnt: int,
    ) -> Dict[str, ParameterConstraints]:
    large_table_constraints = {
        "large_table_" + str(i): ParameterConstraints(
        sharding_types=[sharding_type.value],
        ) for i in range(large_table_cnt)
    }
    small_table_constraints = {
        "small_table_" + str(i): ParameterConstraints(
        sharding_types=[sharding_type.value],
        ) for i in range(small_table_cnt)
    }
    constraints = {**large_table_constraints, **small_table_constraints}
    return constraints


def print_constraints(constraints: Dict[str, ParameterConstraints]) -> None:
    """打印约束的概览与详细字段"""
    try:
        print(f"[Constraints] 总数: {len(constraints)}")
        print(f"[Constraints] 表名列表: {list(constraints.keys())}")
        for name, pc in constraints.items():
            print(
                "[Constraint] table=%s | sharding_types=%s | compute_kernels=%s | pooling_factors=%s | rank=%s | local_size=%s"
                % (
                    name,
                    getattr(pc, "sharding_types", None),
                    getattr(pc, "compute_kernels", None),
                    getattr(pc, "pooling_factors", None),
                    getattr(pc, "rank", None),
                    getattr(pc, "local_size", None),
                )
            )
    except Exception as e:
        print(f"[Constraints] 打印失败: {e}")

# ---------------------------
# 3. Single-rank execution function
# ---------------------------
def single_rank_execution(
    rank: int,
    world_size: int,
    constraints: Dict[str, ParameterConstraints],
    module: torch.nn.Module,
    backend: str,
) -> None:
    try:
        # 先打印收到的 constraints
        print("[Rank %d] 接收到 constraints：" % rank)
        print_constraints(constraints)

        # process group init
        pg = init_distributed(rank, world_size, backend)

        # device setup
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            torch.cuda.set_device(device)

        # topology config
        topology = Topology(world_size=world_size, compute_device="cuda")


        # planner & sharders
        planner = EmbeddingShardingPlanner(
            topology=topology,
            constraints=constraints,
        )
        sharders: List[ModuleSharder] = [EmbeddingBagCollectionSharder()]

        # generate plan
        plan: ShardingPlan = planner.collective_plan(module, sharders, pg)
        print(f"[Rank {rank}] sharding plan _serialize:{plan._serialize()}.")
        

        # build sharded model
        sharded_model = DistributedModelParallel(
            module,
            env=ShardingEnv.from_process_group(pg),
            plan=plan,
            sharders=sharders,
            device=device,
        )

        print(f"[Rank {rank}] Sharding plan:\n{plan}")
        
        kjt = KeyedJaggedTensor.from_offsets_sync(
            keys=["large_table_feature_0", "large_table_feature_1",
                "small_table_feature_0", "small_table_feature_1"],
            values=torch.arange(16, dtype=torch.int32),
            offsets=torch.tensor([0, 4, 8, 12, 16], dtype=torch.int32)
        ).to(device)

        # Forward pass
        output = sharded_model(kjt)
        print(f"[Rank {rank}] Forward output: {output}")
        embeddings = output.wait()  # EmbeddingBagCollection
        for key in embeddings.keys():
            emb = embeddings[key]  # embeddings[key] 返回 tensor
            print(f"[Rank {rank}] Feature: {key}, embedding shape: {emb.shape}")


    finally:
        # 销毁进程组，避免资源泄漏
        if dist.is_initialized():
            dist.destroy_process_group()


# ---------------------------
# 4. Main entry for torchrun
# ---------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--world_size", type=int, default=2)
    parser.add_argument("--backend", type=str, default="nccl")
    args = parser.parse_args()

    # master 配置
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    world_size = args.world_size
    backend = args.backend

    # Build tables
    large_table_cnt = 2
    small_table_cnt = 2

    large_tables = [
        EmbeddingBagConfig(
            name=f"large_table_{i}",
            embedding_dim=64,
            num_embeddings=4096,
            feature_names=[f"large_table_feature_{i}"],
            pooling=PoolingType.SUM,
        )
        for i in range(large_table_cnt)
    ]
    small_tables = [
        EmbeddingBagConfig(
            name=f"small_table_{i}",
            embedding_dim=64,
            num_embeddings=1024,
            feature_names=[f"small_table_feature_{i}"],
            pooling=PoolingType.SUM,
        )
        for i in range(small_table_cnt)
    ]

    # build EmbeddingBagCollection
    ebc = EmbeddingBagCollection(
        tables=large_tables + small_tables,
        device="meta",
    )

    # torchrun 会自动为每个进程设置 RANK 和 LOCAL_RANK
    rank = int(os.environ.get("RANK", 0))

    # generate constraints
    constraints = gen_constraints(
        sharding_type=ShardingType.TABLE_WISE,
        large_table_cnt=large_table_cnt,
        small_table_cnt=small_table_cnt,
    )

    # 主进程也打印一次 constraints
    print("[Main] 生成的 constraints：")
    print_constraints(constraints)

    # run
    single_rank_execution(
        rank=rank,
        world_size=world_size,
        constraints=constraints,
        module=ebc,
        backend=backend,
    )
