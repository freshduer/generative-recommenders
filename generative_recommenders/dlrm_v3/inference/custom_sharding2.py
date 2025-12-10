import os
import torch
import torch.distributed as dist
import logging
import zlib
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

# --- torchrec mock ---
try:
    from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
    from torchrec.distributed.planner.types import ParameterConstraints
    from torchrec.modules.embedding_configs import EmbeddingConfig, DataType
except ImportError:
    KeyedJaggedTensor = object
    ParameterConstraints = EmbeddingConfig = DataType = object

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | Rank %(process)d | %(message)s'
)
logger = logging.getLogger(__name__)

# --- Enums & Data Classes ---
class ShardingType(Enum):
    ROW_WISE = "row_wise"
    TABLE_WISE = "table_wise"
    REPLICATED = "replicated"
    CPU_OFFLOAD = "cpu_offload"

@dataclass
class LocalEmbeddingTable:
    name: str
    weight_cpu: torch.nn.Embedding
    weight_gpu: Optional[torch.nn.Embedding]
    sharding_type: ShardingType
    global_num_embeddings: int
    row_start: int
    row_end: int
    cache_size: int  # GPU cache 行数
    gpu_cache_rows: Optional[torch.Tensor] = None  # GPU cache 行号映射

# --- Sharding Utils ---
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
        return 0, total_rows

    @staticmethod
    def get_dest_ranks_row_wise(indices: torch.Tensor, total_rows: int, world_size: int) -> torch.Tensor:
        base = total_rows // world_size
        rem = total_rows % world_size
        cutoff = rem * (base + 1)
        owners = torch.empty_like(indices, dtype=torch.int64)
        mask_lower = indices < cutoff
        if mask_lower.any():
            owners[mask_lower] = indices[mask_lower] // (base + 1)
        mask_upper = ~mask_lower
        if mask_upper.any():
            owners[mask_upper] = rem + (indices[mask_upper] - cutoff) // base
        return owners

    @staticmethod
    def global_to_local(global_indices: torch.Tensor, table: LocalEmbeddingTable) -> torch.Tensor:
        local = global_indices - table.row_start
        return local

# --- Embedding Collection with CPU/GPU Cache ---
class CustomEmbeddingCollection(torch.nn.Module):
    def __init__(
        self,
        table_config: Dict[str, EmbeddingConfig],
        cache_ratio: float = 0.1,  # GPU cache占总表的比例
        device: torch.device = None,
        constraints: Optional[Dict[str, ParameterConstraints]] = None,
    ):
        super().__init__()
        if dist.is_available() and dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            print(f"Initializing CustomEmbeddingCollection with rank={self.rank}, world_size={self.world_size}")
        else:
            self.rank = 0
            self.world_size = 1
            print("Running in SINGLE CARD mode.")

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
        self.cache_ratio = cache_ratio

        print(f"Device: {self.device}, cache_ratio: {cache_ratio}, num_tables: {len(table_config)}")
        for idx, (name, cfg) in enumerate(table_config.items(), 1):
            print(f"Initializing table {idx}/{len(table_config)}: {name}")
            self._init_table(name, cfg)
        print("CustomEmbeddingCollection initialization complete")

    def _init_table(self, name: str, cfg: EmbeddingConfig):
        sharding_type = self._infer_sharding_type(name)
        num_embeddings = cfg.num_embeddings
        embedding_dim = cfg.embedding_dim
        print(f"  Table '{name}': sharding={sharding_type.value}, total_rows={num_embeddings}, dim={embedding_dim}")

        start, end = ShardingUtils.get_partition_bounds(
            sharding_type, num_embeddings, self.world_size, self.rank, name
        )
        local_rows = end - start
        print(f"  Partition: rows [{start}:{end}], local_rows={local_rows}")

        # CPU 主存
        print(f"  Allocating CPU embedding: {local_rows} x {embedding_dim}")
        weight_cpu = torch.nn.Embedding(local_rows, embedding_dim)
        torch.nn.init.uniform_(weight_cpu.weight, -0.01, 0.01)
        cpu_mem_mb = (local_rows * embedding_dim * 4) / (1024**2)  # assume fp32
        print(f"  CPU memory allocated: {cpu_mem_mb:.2f} MB")

        # GPU cache
        cache_rows = max(1, int(local_rows * self.cache_ratio))
        print(f"  GPU cache: cache_rows={cache_rows} (ratio={self.cache_ratio:.4f})")
        if cache_rows > 0:
            # 随机选 cache_rows 行放入 GPU
            print(f"  Creating GPU cache with {cache_rows} rows...")
            perm = torch.randperm(local_rows)[:cache_rows]
            weight_gpu = torch.nn.Embedding(cache_rows, embedding_dim, device=self.device)
            weight_gpu.weight.data.copy_(weight_cpu.weight.data[perm].to(self.device))
            gpu_rows_idx = perm.to(self.device)  # 移到 GPU
            gpu_mem_mb = (cache_rows * embedding_dim * 4) / (1024**2)
            print(f"  GPU cache created: {gpu_mem_mb:.2f} MB")
        else:
            print(f"  No GPU cache (cache_rows={cache_rows})")
            weight_gpu = None
            gpu_rows_idx = None

        self.tables[name] = LocalEmbeddingTable(
            name=name,
            weight_cpu=weight_cpu,
            weight_gpu=weight_gpu,
            sharding_type=sharding_type,
            global_num_embeddings=num_embeddings,
            row_start=start,
            row_end=end,
            cache_size=cache_rows,
            gpu_cache_rows=gpu_rows_idx
        )

        num_features = len(cfg.feature_names or [])
        for feat in (cfg.feature_names or []):
            self.feature_to_table[feat] = name
        print(f"  Mapped {num_features} features to table '{name}'")

    def _infer_sharding_type(self, table_name: str) -> ShardingType:
        if pc := self._constraints.get(table_name):
            if sts := getattr(pc, "sharding_types", None):
                return ShardingUtils.parse_type(sts[0])
        return ShardingType.ROW_WISE

    def forward(self, kjt: KeyedJaggedTensor) -> Dict[str, torch.Tensor]:
        results = {}
        feature_keys = kjt.keys() if isinstance(kjt.keys(), list) else list(kjt.keys())
        for key in feature_keys:
            table_name = self.feature_to_table.get(key)
            if not table_name:
                continue

            table = self.tables[table_name]
            indices = kjt[key].values()
            if indices.device != self.device:
                indices = indices.to(self.device, non_blocking=True)

            if table.weight_gpu is not None:
                # GPU cache lookup
                mask_gpu = torch.isin(indices, table.gpu_cache_rows)
                emb_out = torch.zeros((indices.shape[0], table.weight_cpu.embedding_dim), device=self.device)
                if mask_gpu.any():
                    # 查 GPU cache，需要映射到 GPU cache 索引
                    gpu_idx = torch.tensor([torch.where(table.gpu_cache_rows == i)[0][0] for i in indices[mask_gpu]], device=self.device)
                    emb_out[mask_gpu] = table.weight_gpu(gpu_idx)

                # CPU lookup
                mask_cpu = ~mask_gpu
                if mask_cpu.any():
                    idx_cpu = ShardingUtils.global_to_local(indices[mask_cpu], table)
                    idx_cpu = idx_cpu.to(table.weight_cpu.weight.device, non_blocking=True).long()
                    cpu_vals = table.weight_cpu(idx_cpu)
                    emb_out[mask_cpu] = cpu_vals.to(self.device, non_blocking=True)

                results[key] = emb_out
            else:
                # 全部从 CPU
                idx_cpu = ShardingUtils.global_to_local(indices, table)
                idx_cpu = idx_cpu.to(table.weight_cpu.weight.device, non_blocking=True).long()
                cpu_vals = table.weight_cpu(idx_cpu)
                results[key] = cpu_vals.to(self.device, non_blocking=True)

        return results
