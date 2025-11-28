import os
import torch
import torch.distributed as dist
import logging
import zlib
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

# 依赖 torchrec，如环境缺失请自行替换相关类型定义
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from torchrec.distributed.planner.types import ParameterConstraints
from torchrec.modules.embedding_configs import EmbeddingConfig, DataType


# 配置 Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | Rank %(process)d | %(message)s'
)
logger = logging.getLogger(__name__)

# --- Enums & Data Structures ---

class ShardingType(Enum):
    ROW_WISE = "row_wise"
    TABLE_WISE = "table_wise"
    REPLICATED = "replicated"
    CPU_OFFLOAD = "cpu_offload"

@dataclass
class LocalEmbeddingTable:
    name: str
    weight: torch.nn.Embedding
    sharding_type: ShardingType
    global_num_embeddings: int
    row_start: int  # Global index start for this rank (RowWise)
    row_end: int    # Global index end for this rank (RowWise)

# --- Sharding Utilities (纯逻辑计算) ---

class ShardingUtils:
    @staticmethod
    def parse_type(s: str) -> ShardingType:
        s = (s or "").lower()
        if s in ("row_wise", "rowwise", "row-wise"): return ShardingType.ROW_WISE
        if s in ("table_wise", "tablewise", "table-wise"): return ShardingType.TABLE_WISE
        if s in ("replicated", "data_parallel", "dp"): return ShardingType.REPLICATED
        if s in ("cpu_offload", "cpu-offload", "cpu"): return ShardingType.CPU_OFFLOAD
        return ShardingType.ROW_WISE

    @staticmethod
    def get_partition_bounds(sharding_type: ShardingType, total_rows: int, world_size: int, rank: int, table_name: str) -> Tuple[int, int]:
        """计算当前 Rank 负责的全局行号范围 [start, end)"""
        if sharding_type == ShardingType.ROW_WISE:
            base = total_rows // world_size
            rem = total_rows % world_size
            if rank < rem:
                start = rank * (base + 1)
                end = start + base + 1
            else:
                start = rem * (base + 1) + (rank - rem) * base
                end = start + base
            return start, end
        elif sharding_type == ShardingType.TABLE_WISE:
            owner = zlib.adler32(table_name.encode()) % world_size
            return (0, total_rows) if rank == owner else (0, 0)
        return 0, total_rows # Replicated / CPU Offload

    @staticmethod
    def get_dest_ranks_row_wise(indices: torch.Tensor, total_rows: int, world_size: int) -> torch.Tensor:
        """计算 Row-Wise 模式下每个 index 归属的 rank"""
        base = total_rows // world_size
        rem = total_rows % world_size
        cutoff = rem * (base + 1)
        
        owners = torch.empty_like(indices, dtype=torch.int64)
        mask_lower = indices < cutoff
        
        # Vectorized calculation
        if mask_lower.any():
            owners[mask_lower] = indices[mask_lower] // (base + 1)
        
        mask_upper = ~mask_lower
        if mask_upper.any():
            owners[mask_upper] = rem + (indices[mask_upper] - cutoff) // base
        return owners

    @staticmethod
    def global_to_local(global_indices: torch.Tensor, table: LocalEmbeddingTable) -> torch.Tensor:
        """将接收到的全局 ID 转换为本地 Lookup ID"""
        if table.sharding_type == ShardingType.ROW_WISE:
            # 将全局索引映射到本地分片区间
            local = global_indices - table.row_start
            local_rows = table.row_end - table.row_start
            if local_rows > 0:
                # 使用 remainder 确保范围在 [0, local_rows)
                local = torch.remainder(local, local_rows)
            return local
        return global_indices

# --- Main Module ---

class CustomEmbeddingCollection(torch.nn.Module):
    def __init__(
        self,
        table_config: Dict[str, EmbeddingConfig],
        constraints: Optional[Dict[str, ParameterConstraints]] = None,
        device: torch.device = None,
    ) -> None:
        super().__init__()
        
        # --- 1. 智能环境检测 (支持单卡 Debug) ---
        if dist.is_available() and dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1
            logger.info("Distributed environment not initialized, running in SINGLE CARD mode.")

        # --- 2. 设备绑定 ---
        if device is None:
            if torch.cuda.is_available():
                local_rank = int(os.environ.get("LOCAL_RANK", 0))
                self.device = torch.device(f"cuda:{local_rank}")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device

        self.tables: Dict[str, LocalEmbeddingTable] = {}
        self.feature_to_table: Dict[str, str] = {}
        self._constraints = constraints or {}

        # --- 3. 初始化表 ---
        for name, cfg in table_config.items():
            self._init_table(name, cfg)

    def _init_table(self, name: str, cfg: EmbeddingConfig):
        sharding_type = self._infer_sharding_type(name)
        num_embeddings = cfg.num_embeddings
        embedding_dim = cfg.embedding_dim

        # 计算分区 (单卡模式下 start=0, end=total)
        start, end = ShardingUtils.get_partition_bounds(
            sharding_type, num_embeddings, self.world_size, self.rank, name
        )
        local_rows = end - start

        # 确定设备与精度
        is_offload = sharding_type == ShardingType.CPU_OFFLOAD
        target_device = torch.device("cpu") if is_offload else self.device
        
        dtype_map = {
            getattr(DataType, "FP32", None): torch.float32,
            getattr(DataType, "FP16", None): torch.float16,
            getattr(DataType, "BF16", None): torch.bfloat16,
        }
        target_dtype = dtype_map.get(getattr(cfg, "data_type", None), torch.float32)

        # 创建 Layer
        # 注意: 即使 local_rows=0 (我不持有该表), 也要创建 layer 结构以保持代码一致性, 但不占显存
        emb_layer = torch.nn.Embedding(local_rows, embedding_dim, device=target_device, dtype=target_dtype)
        if local_rows > 0:
            torch.nn.init.uniform_(emb_layer.weight, -0.01, 0.01)

        self.tables[name] = LocalEmbeddingTable(
            name=name,
            weight=emb_layer,
            sharding_type=sharding_type,
            global_num_embeddings=num_embeddings,
            row_start=start,
            row_end=end
        )
        
        for feat in (cfg.feature_names or []):
            self.feature_to_table[feat] = name
            
        if self.rank == 0:
            logger.info(f"Initialized table '{name}': {sharding_type.value}, dim={embedding_dim}, total_rows={num_embeddings}")

    def _infer_sharding_type(self, table_name: str) -> ShardingType:
        if pc := self._constraints.get(table_name):
            if sts := getattr(pc, "sharding_types", None):
                return ShardingUtils.parse_type(sts[0])
        return ShardingType.ROW_WISE

    # --------------------------------------------------------------------------
    # Forward Pass Logic
    # --------------------------------------------------------------------------

    def forward(self, kjt: KeyedJaggedTensor) -> Dict[str, torch.Tensor]:
        results = {}
        feature_keys = kjt.keys() if isinstance(kjt.keys(), list) else list(kjt.keys())
        
        # --- [单卡优化] ---
        # 只要是单卡，无论配置成什么切分策略，数据一定全在本地。
        # 直接走 local lookup，跳过所有通信和排序开销。
        is_single_card = (self.world_size == 1)

        for key in feature_keys:
            table_name = self.feature_to_table.get(key)
            if not table_name: 
                continue

            table = self.tables[table_name]
            indices = kjt[key].values()
            
            if indices.device != self.device:
                indices = indices.to(self.device, non_blocking=True)

            # 判定是否可以直接本地查询
            should_direct_lookup = (
                is_single_card or 
                table.sharding_type in (ShardingType.REPLICATED, ShardingType.CPU_OFFLOAD)
            )

            if should_direct_lookup:
                # 即使是 Direct Lookup，也建议做一次 OOB 检查 (开发阶段)
                # self._check_oob(indices, table) 
                embeddings = self._lookup_direct(indices, table)
            else:
                # 分布式查询
                embeddings = self._lookup_distributed(indices, table)
            
            results[key] = embeddings
        
        return results

    def _lookup_direct(self, indices: torch.Tensor, table: LocalEmbeddingTable) -> torch.Tensor:
        """本地直接查询"""
        if table.sharding_type == ShardingType.CPU_OFFLOAD:
            return table.weight(indices.cpu()).to(self.device, non_blocking=True)
        return table.weight(indices)

    def _lookup_distributed(self, global_indices: torch.Tensor, table: LocalEmbeddingTable) -> torch.Tensor:
        """分布式查询流水线"""
        
        # 0. 预处理
        global_indices = global_indices.long()
        # 发送端检查：尽量在源头修复或报警
        self._check_oob(global_indices, table)

        # 1. Routing
        dest_ranks = self._get_dest_ranks(global_indices, table)

        # 2. Shuffle
        indices_sorted, sort_idx, send_splits = self._sort_by_dest(global_indices, dest_ranks)

        # 3. Exchange Indices
        recv_indices, recv_splits = self._exchange_data(indices_sorted, send_splits, dtype=torch.int64)

        # 4. Local Lookup (Owner Side)
        local_indices = ShardingUtils.global_to_local(recv_indices, table)
        
        # ==================== [DEBUG: CRASH DETECTION] ====================
        # 在真正查表前，执行最后一道防线检查
        if local_indices.numel() > 0:
            real_limit = table.weight.num_embeddings
            # 快速获取最大最小值 (会有少量 GPU-CPU 同步开销，但为了 Debug 值得)
            # 生产环境如果追求极致性能，可以注释掉下面几行
            min_idx = local_indices.min().item()
            max_idx = local_indices.max().item()
            
            if min_idx < 0 or max_idx >= real_limit:
                bad_indices = local_indices[(local_indices < 0) | (local_indices >= real_limit)]
                sample_bad = bad_indices[:10].tolist()
                
                error_msg = (
                    f"\n>>> [CRASH DETECTED] Rank {self.rank} | Table '{table.name}' <<<\n"
                    f"    Sharding Type: {table.sharding_type}\n"
                    f"    Table Weight Shape: {table.weight.weight.shape}\n"
                    f"    Local Indices Shape: {local_indices.shape}\n"
                    f"    Detected Range: min={min_idx}, max={max_idx}\n"
                    f"    Allowed Range: [0, {real_limit - 1}]\n"
                    f"    Sample Bad Indices: {sample_bad}\n"
                    f"    Hints: Check if Global-to-Local logic is consistent with partitioning.\n"
                )
                logger.info(error_msg)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize() # 强制刷新日志
                
                raise RuntimeError(f"Index out of bounds detected in table '{table.name}' on Rank {self.rank}")
        # ==================== [DEBUG END] ====================

        if table.weight.num_embeddings == 0:
            # 如果我不持有该表数据，返回零向量
            emb_dim = table.weight.embedding_dim
            local_embs = torch.zeros((recv_indices.shape[0], emb_dim), dtype=table.weight.weight.dtype, device=self.device)
        else:
            local_embs = table.weight(local_indices)

        # 5. Exchange Embeddings
        recv_embs_sorted, _ = self._exchange_data(
            local_embs, 
            send_splits=recv_splits, 
            output_splits_hint=send_splits,
            dtype=local_embs.dtype
        )

        # 6. Restore
        return self._restore_order(recv_embs_sorted, sort_idx)

    # --------------------------------------------------------------------------
    # Distributed Pipeline Helpers (Private)
    # --------------------------------------------------------------------------

    def _get_dest_ranks(self, indices: torch.Tensor, table: LocalEmbeddingTable) -> torch.Tensor:
        if table.sharding_type == ShardingType.TABLE_WISE:
            owner = zlib.adler32(table.name.encode()) % self.world_size
            return torch.full_like(indices, owner, dtype=torch.int64)
        else:
            return ShardingUtils.get_dest_ranks_row_wise(indices, table.global_num_embeddings, self.world_size)

    def _sort_by_dest(self, indices: torch.Tensor, dest_ranks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        sort_idx = torch.argsort(dest_ranks)
        indices_sorted = indices[sort_idx].contiguous()
        splits_tensor = torch.bincount(dest_ranks, minlength=self.world_size)
        send_splits = splits_tensor.tolist()
        return indices_sorted, sort_idx, send_splits

    def _exchange_data(self, data: torch.Tensor, send_splits: List[int], output_splits_hint: Optional[List[int]] = None, dtype: torch.dtype = torch.float32) -> Tuple[torch.Tensor, List[int]]:
        if output_splits_hint is not None:
            recv_splits = output_splits_hint
        else:
            if self.world_size == 1:
                recv_splits = send_splits
            else:
                send_cuda = torch.tensor(send_splits, device=self.device, dtype=torch.int64)
                recv_cuda = torch.zeros_like(send_cuda)
                dist.all_to_all_single(recv_cuda, send_cuda)
                recv_splits = recv_cuda.tolist()

        total_recv = sum(recv_splits)
        recv_shape = [total_recv] + list(data.shape[1:])
        recv_data = torch.empty(recv_shape, dtype=dtype, device=self.device)

        dist.all_to_all_single(
            recv_data, data,
            output_split_sizes=recv_splits,
            input_split_sizes=send_splits
        )
        return recv_data, recv_splits

    def _restore_order(self, data_sorted: torch.Tensor, sort_idx: torch.Tensor) -> torch.Tensor:
        num_items = data_sorted.shape[0]
        if num_items == 0:
            return data_sorted
        inv_perm = torch.empty_like(sort_idx)
        inv_perm.index_copy_(0, sort_idx, torch.arange(num_items, device=self.device, dtype=sort_idx.dtype))
        return data_sorted.index_select(0, inv_perm)

    def _check_oob(self, indices: torch.Tensor, table: LocalEmbeddingTable):
        """发送端预检与自动修复"""
        if indices.numel() == 0:
            return
        
        # 为了性能，这里不做同步检查，只做简单的范围修正
        # 如果需要严格检查，可以像 _lookup_distributed 那样 .item() 判断
        if indices.min() < 0 or indices.max() >= table.global_num_embeddings:
            msg = f"[Rank {self.rank}] Table '{table.name}' OOB (Sender). Correcting via remainder."
            logger.info(msg)
            indices.remainder_(table.global_num_embeddings)