import os
import time
import torch
import torch.distributed as dist
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
from enum import Enum
import logging
import zlib
from torchrec.distributed.planner.types import ParameterConstraints
from torchrec.modules.embedding_configs import EmbeddingConfig, DataType  # 增加 DataType 以支持 dtype 映射

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ShardingType(Enum):
    ROW_WISE = "row_wise"
    TABLE_WISE = "table_wise"
    REPLICATED = "replicated"
    CPU_OFFLOAD = "cpu_offload"

@dataclass
class LocalEmbeddingTable:
    weight: torch.nn.Embedding
    row_start: int
    row_end: int
    sharding_type: ShardingType

class SimpleKeyedTensor:
    def __init__(self, lengths: torch.Tensor, values: torch.Tensor) -> None:
        self._lengths = lengths
        self._values = values

def _string_to_sharding(s: str) -> ShardingType:
    s = (s or "").lower()
    if s in ("row_wise", "rowwise", "row-wise"):
        return ShardingType.ROW_WISE
    if s in ("table_wise", "tablewise", "table-wise"):
        return ShardingType.TABLE_WISE
    if s in ("replicated", "data_parallel", "dp"):
        return ShardingType.REPLICATED
    if s in ("cpu_offload", "cpu-offload", "cpu"):
        return ShardingType.CPU_OFFLOAD
    return ShardingType.ROW_WISE

def get_partition_bounds(
    sharding_type: ShardingType,
    num_rows: int,
    world_size: int,
    rank: int,
    table_name: str
) -> Tuple[int, int]:
    if sharding_type == ShardingType.ROW_WISE:
        base = num_rows // world_size
        rem = num_rows % world_size
        if rank < rem:
            start = rank * (base + 1)
            end = start + base + 1
        else:
            start = rem * (base + 1) + (rank - rem) * base
            end = start + base
        return start, end
    elif sharding_type == ShardingType.TABLE_WISE:
        owner_rank = zlib.adler32(table_name.encode()) % world_size
        if rank == owner_rank:
            return 0, num_rows
        else:
            return 0, 0
    elif sharding_type in [ShardingType.REPLICATED, ShardingType.CPU_OFFLOAD]:
        return 0, num_rows
    else:
        raise ValueError(f"Unsupported sharding type: {sharding_type}")

class CustomEmbeddingCollection(torch.nn.Module):
    """
    参数:
      table_config: Dict[str, EmbeddingConfig] (来自 get_embedding_table_config)
      constraints: Dict[str, ParameterConstraints] (提供 sharding_types 等)
      device: 目标 GPU device
    """
    def __init__(
        self,
        table_config: Dict[str, EmbeddingConfig],
        constraints: Optional[Dict[str, ParameterConstraints]],
        device: torch.device,
    ) -> None:
        super().__init__()
        assert dist.is_initialized()
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.device = device
        self.tables: Dict[str, LocalEmbeddingTable] = {}
        self.feature_to_table: Dict[str, str] = {}
        self._constraints = constraints or {}

        for name, cfg in table_config.items():
            num_embeddings = cfg.num_embeddings
            embedding_dim = cfg.embedding_dim
            feature_names = list(cfg.feature_names or [])

            # 根据 constraints 推断 sharding
            sharding_type = self._infer_sharding_type(name)

            start, end = get_partition_bounds(
                sharding_type, num_embeddings, self.world_size, self.rank, name
            )
            local_rows = end - start
            target_device = (
                torch.device("cpu")
                if sharding_type == ShardingType.CPU_OFFLOAD
                else self.device
            )
            # logger.info(f"[rank {self.rank}] Table '{name}' assigned rows ({start}, {end}) with sharding {sharding_type.value} on device {target_device}")
            # logger.info(f"local_rows: {local_rows}, embedding_dim: {embedding_dim}")
            # 根据 EmbeddingConfig.data_type 设置 dtype，降低显存占用
            dtype_map = {
                getattr(DataType, "FP32", None): torch.float32,
                getattr(DataType, "FP16", None): torch.float16,
                getattr(DataType, "BF16", None): torch.bfloat16,
            }
            target_dtype = dtype_map.get(getattr(cfg, "data_type", getattr(DataType, "FP32", None)), torch.float32)
            emb = torch.nn.Embedding(local_rows, embedding_dim, device=target_device, dtype=target_dtype)
            if local_rows > 0:
                torch.nn.init.uniform_(emb.weight, -0.01, 0.01)

            self.tables[name] = LocalEmbeddingTable(
                weight=emb,
                row_start=start,
                row_end=end,
                sharding_type=sharding_type,
            )
            for feat in feature_names:
                self.feature_to_table[feat] = name

            if self.rank == 0:
                logger.info(
                    f"Table '{name}' init: num_embeddings={num_embeddings}, dim={embedding_dim}, "
                    f"range=({start},{end}), local_rows={local_rows}, sharding={sharding_type.value}, device={target_device}"
                )

    def _infer_sharding_type(self, table_name: str) -> ShardingType:
        pc = self._constraints.get(table_name)
        if pc is not None:
            sts = getattr(pc, "sharding_types", None)
            if sts:
                logger.info(f"Inferred sharding type for table '{table_name}': {sts[0]}")
                return _string_to_sharding(sts[0])
        return ShardingType.ROW_WISE

    def _lookup_one_table(self, global_indices: torch.Tensor, table: LocalEmbeddingTable) -> torch.Tensor:
        logger.info(
            f"[rank {self.rank}] Lookup table='{table.weight}' sharding={table.sharding_type.value} "
            f"indices_shape={tuple(global_indices.shape)} device={global_indices.device}"
        )
        if table.sharding_type == ShardingType.CPU_OFFLOAD:
            indices_cpu = global_indices.cpu()
            out_cpu = table.weight(indices_cpu)
            return out_cpu.to(self.device, non_blocking=True)
        logger.info(f"[rank {self.rank}] 111")
        if table.sharding_type == ShardingType.REPLICATED:
            # 安全越界检测：REPLICATED 模式直接使用全局 ID，需要确保范围合法
            if global_indices.numel() > 0:
                vocab_size = table.weight.num_embeddings
                max_id = int(global_indices.max())
                min_id = int(global_indices.min())
                if max_id >= vocab_size or min_id < 0:
                    bad_mask = (global_indices < 0) | (global_indices >= vocab_size)
                    bad_count = int(bad_mask.sum())
                    sample_bad = global_indices[bad_mask][:10].tolist()
                    raise RuntimeError(
                        f"[rank {self.rank}] Replicated table index out of bounds: bad_count={bad_count} "
                        f"min={min_id} max={max_id} vocab_size={vocab_size} sample_bad={sample_bad}"
                    )
            return table.weight(global_indices)
        logger.info(f"[rank {self.rank}] 222")

        # ROW_WISE / TABLE_WISE
        mask = (global_indices >= table.row_start) & (global_indices < table.row_end)
        # mask_hits = int(mask.sum().item())
        logger.info(
            f"[rank {self.rank}] Table rows=({table.row_start},{table.row_end}) "
            # f"mask_hits={mask_hits}/{global_indices.numel()}"
        )
        # 用本地 weight dtype 以减少转换
        out = torch.zeros(
            global_indices.shape[0],
            table.weight.embedding_dim,
            device=self.device,
            dtype=table.weight.weight.dtype,
        )
        logger.info(f"[rank {self.rank}] torch.zeros done")
        # 分块处理命中索引，降低一次性内存占用
        if mask.any():
            hit_pos = torch.nonzero(mask, as_tuple=False).squeeze(1)
            total_hits = hit_pos.numel()
            logger.info(f"[rank {self.rank}] 333 hits={total_hits}")
            if total_hits > 0:
                chunk_size = 64_000  # 可调参数
                vocab_size = table.weight.num_embeddings
                for start in range(0, total_hits, chunk_size):
                    end = min(start + chunk_size, total_hits)
                    pos_slice = hit_pos[start:end]
                    local_slice = global_indices[pos_slice] - table.row_start
                    # 越界安全检测（按本地 embedding 大小）
                    bad_mask_local = (local_slice < 0) | (local_slice >= vocab_size)
                    if bad_mask_local.any():
                        bad_count = int(bad_mask_local.sum())
                        sample_bad = local_slice[bad_mask_local][:10].tolist()
                        raise RuntimeError(
                            f"[rank {self.rank}] Local row-wise/table-wise index out of bounds: bad_count={bad_count} "
                            f"vocab_size={vocab_size} sample_bad={sample_bad} row_start={table.row_start} row_end={table.row_end}"
                        )
                    emb_slice = table.weight(local_slice)
                    out[pos_slice] = emb_slice
                logger.info(f"[rank {self.rank}] 666 chunked lookup done")
        # 确保参与 all_reduce 的张量在所有 rank 形状一致；必要时按最大长度进行零填充
        n_local = torch.tensor([out.shape[0]], device=self.device, dtype=torch.int64)
        n_list = [torch.zeros_like(n_local) for _ in range(self.world_size)]
        dist.all_gather(n_list, n_local)
        sizes = torch.stack(n_list).squeeze(-1)
        max_n = int(sizes.max().item())

        unpad = False
        if out.shape[0] != max_n:
            pad_rows = max_n - out.shape[0]
            pad = torch.zeros(
                pad_rows,
                table.weight.embedding_dim,
                device=self.device,
                dtype=out.dtype,
            )
            out = torch.cat([out, pad], dim=0)
            unpad = True

        start = time.time()
        logger.info(f"[rank {self.rank}] Starting all_reduce... (padded_rows={max_n})")
        dist.all_reduce(out, op=dist.ReduceOp.SUM)
        logger.info(
            f"[rank {self.rank}] all_reduce completed in {(time.time() - start)*1e3:.3f} ms"
        )
        if unpad:
            out = out[: int(n_local.item()), :]
        return out

    def forward(self, kjt: KeyedJaggedTensor) -> Dict[str, torch.Tensor]:
        result: Dict[str, torch.Tensor] = {}
        keys_obj = kjt.keys()
        feature_keys = [keys_obj] if isinstance(keys_obj, str) else list(keys_obj)

        for feat_key in feature_keys:
            table_name = self.feature_to_table.get(feat_key)
            if not table_name:
                logger.warning(f"[rank {self.rank}] Feature '{feat_key}' missing table mapping, skipped")
                continue
            table = self.tables[table_name]
            indices = kjt[feat_key].values()
            logger.info(
                f"[rank {self.rank}] Forward feature='{feat_key}', table='{table_name}', "
                f"indices_shape={tuple(indices.shape)} device={indices.device}"
            )
            if indices.device != self.device:
                indices = indices.to(self.device)
            embeddings = self._lookup_one_table(indices, table)
            result[feat_key] = embeddings
        return result