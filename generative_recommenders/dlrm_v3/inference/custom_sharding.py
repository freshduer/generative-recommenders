import os
import torch
import torch.distributed as dist
import logging
import zlib
from collections import OrderedDict
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

# --- Mock Imports for Standalone Running ---
try:
    from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
    from torchrec.distributed.planner.types import ParameterConstraints
    from torchrec.modules.embedding_configs import EmbeddingConfig, DataType
except ImportError:
    # Mock classes so code runs without torchrec installed
    KeyedJaggedTensor = object 
    ParameterConstraints = object
    
    @dataclass
    class EmbeddingConfig:
        num_embeddings: int
        embedding_dim: int
        name: str
        data_type: str = "fp32"
        feature_names: List[str] = None
    
    class DataType:
        FP32 = "fp32"
        FP16 = "fp16"

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | Rank %(process)d | %(message)s'
)
logger = logging.getLogger(__name__)

# --- Enums ---

class ShardingType(Enum):
    ROW_WISE = "row_wise"
    TABLE_WISE = "table_wise"
    REPLICATED = "replicated"
    CPU_OFFLOAD = "cpu_offload"

# --- Vectorized Cache Module (Core Optimization) ---

class VectorizedEmbeddingCache(torch.nn.Module):
    """
    High-performance, purely vectorized GPU Cache for Embeddings.
    Replaces the slow Python-loop based dictionary approach.
    """
    def __init__(
        self, 
        table_name: str,
        total_num_embeddings: int, 
        embedding_dim: int, 
        max_cache_entries: int,
        device: torch.device,
        dtype: torch.dtype
    ):
        super().__init__()
        self.table_name = table_name
        self.num_embeddings = total_num_embeddings
        self.max_entries = max_cache_entries
        self.device = device
        
        # 1. Cache Storage (The actual cached vectors)
        # Shape: [Cache_Size, Dim]
        self.cache_data = torch.nn.Parameter(
            torch.empty((max_cache_entries, embedding_dim), device=device, dtype=dtype),
            requires_grad=False
        )
        torch.nn.init.uniform_(self.cache_data, -0.01, 0.01)

        # 2. Mapping Table (The Page Table)
        # Maps Global_ID -> Cache_Slot_Index. 
        # Value -1 means "Not in Cache".
        # Warning: Consumes (Num_Embeddings * 8) bytes memory.
        self.register_buffer(
            "mapping_table", 
            torch.full((total_num_embeddings,), -1, dtype=torch.long, device=device)
        )

        # 3. LRU Tracking
        # Stores the "timestamp" of last access for each slot
        self.register_buffer(
            "access_tick", 
            torch.zeros((max_cache_entries,), dtype=torch.long, device=device)
        )
        self.global_tick = 0
        self.filled = 0

        # Stats
        self.stats_hits = 0
        self.stats_misses = 0
        self.stats_evictions = 0

        # Track current owner of each slot so we can invalidate quickly
        self.register_buffer(
            "slot_to_id",
            torch.full((max_cache_entries,), -1, dtype=torch.long, device=device)
        )

    def forward(self, indices: torch.Tensor, cpu_weight: torch.nn.Embedding) -> torch.Tensor:
        """
        Args:
            indices: Local indices to lookup (GPU tensor)
            cpu_weight: Reference to the master CPU embedding layer
        """
        flat_indices = indices.view(-1)
        if flat_indices.numel() == 0:
            return torch.empty((0, self.cache_data.shape[1]), device=self.device, dtype=self.cache_data.dtype)

        # Update global clock
        self.global_tick += 1

        # 1. Check Cache Hits
        # Direct lookup in the mapping table (Fast!)
        slots = self.mapping_table[flat_indices]
        
        hit_mask = slots != -1
        miss_mask = ~hit_mask
        
        num_misses = miss_mask.sum().item()
        self.stats_hits += hit_mask.sum().item()
        self.stats_misses += num_misses

        # 2. Handle Misses (if any)
        if num_misses > 0:
            # Identify which IDs are missing
            miss_indices = flat_indices[miss_mask]
            
            # Deduplicate missing IDs to avoid fetching same ID twice in one batch
            unique_miss_indices, inverse_map = torch.unique(miss_indices, return_inverse=True)
            num_unique_miss = unique_miss_indices.numel()

            # Allocate free slots first (fast path for initial fills)
            free_capacity = max(self.max_entries - self.filled, 0)
            allocate_free = min(num_unique_miss, free_capacity)
            slots_for_unique = torch.empty(
                num_unique_miss, dtype=torch.long, device=self.device
            )
            if allocate_free > 0:
                free_range = torch.arange(
                    self.filled,
                    self.filled + allocate_free,
                    device=self.device,
                    dtype=torch.long,
                )
                slots_for_unique[:allocate_free] = free_range
                self.filled += allocate_free
            else:
                free_range = torch.empty(0, dtype=torch.long, device=self.device)

            remaining = num_unique_miss - allocate_free
            if remaining > 0:
                # Need to reuse existing slots via LRU
                _, evict_slots = torch.topk(self.access_tick, k=remaining, largest=False)
                slots_for_unique[allocate_free:] = evict_slots
                victim_slots = evict_slots
                self.stats_evictions += remaining
            else:
                victim_slots = torch.empty(0, dtype=torch.long, device=self.device)

            # --- CPU Fetch (The expensive part) ---
            # Move indices to CPU, fetch, move back. 
            # Pin_memory on cpu_weight helps here.
            idx_cpu = unique_miss_indices.to(cpu_weight.weight.device)
            fetched_data = cpu_weight(idx_cpu).to(self.device, non_blocking=True)
            
            # --- Update Cache State ---
            # 1. Invalidate old mapping (The ID that used to be in this slot is now gone)
            if victim_slots.numel() > 0:
                old_ids = self.slot_to_id[victim_slots]
                valid_old = old_ids != -1
                if valid_old.any():
                    check_ids = old_ids[valid_old]
                    check_slots = victim_slots[valid_old]
                    current_map = self.mapping_table[check_ids]
                    match = current_map == check_slots
                    if match.any():
                        self.mapping_table[check_ids[match]] = -1

            # 2. Update Cache Data
            self.cache_data[slots_for_unique] = fetched_data

            # 3. Update Mapping Table (The new IDs now point to these slots)
            self.mapping_table[unique_miss_indices] = slots_for_unique

            # 4. Update LRU Ticks for new slots
            self.access_tick[slots_for_unique] = self.global_tick

            # 5. Fix the 'slots' variable for the original request so we can gather
            slots[miss_mask] = slots_for_unique[inverse_map]

            # Register new owners
            self.slot_to_id[slots_for_unique] = unique_miss_indices

        # 3. Update LRU for hits (and just-filled misses)
        # Using scatter is faster than indexing for large updates
        self.access_tick[slots] = self.global_tick
        
        # 4. Gather Result
        output = self.cache_data[slots]
        
        return output.view(*indices.shape, -1)

    def stats(self) -> Dict[str, float]:
        total = self.stats_hits + self.stats_misses
        return {
            "hits": self.stats_hits,
            "misses": self.stats_misses,
            "evictions": self.stats_evictions,
            "ratio": self.stats_hits / total if total > 0 else 0.0
        }

# --- Sharding Utilities ---

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
    def global_to_local(global_indices: torch.Tensor, table_sharding: ShardingType, row_start: int, local_limit: int) -> torch.Tensor:
        if local_limit == 0: return global_indices
        if table_sharding == ShardingType.ROW_WISE:
            local = global_indices - row_start
            return torch.remainder(local, local_limit)
        elif table_sharding == ShardingType.TABLE_WISE:
            return torch.remainder(global_indices, local_limit)
        return global_indices

# --- Local Table Wrapper ---

@dataclass
class LocalEmbeddingTable:
    name: str
    weight: torch.nn.Embedding
    sharding_type: ShardingType
    global_num_embeddings: int
    row_start: int 
    row_end: int
    cache: Optional[VectorizedEmbeddingCache] = None 

# --- Main Module ---

class CustomEmbeddingCollection(torch.nn.Module):
    def __init__(
        self,
        table_config: Dict[str, EmbeddingConfig],
        constraints: Optional[Dict[str, ParameterConstraints]] = None,
        device: torch.device = None,
        embedding_budget_bytes: Optional[int] = None,
    ) -> None:
        super().__init__()
        
        # 1. Environment & Device
        if dist.is_available() and dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1
        
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
        
        # Calculate budget per table (Simple Strategy: Equal split or Proportional?)
        # For simplicity, we assign the full budget check logic loosely, 
        # or assuming the budget is GLOBAL. 
        # Here we split budget equally among tables enabled for caching for simplicity.
        self.global_budget_bytes = embedding_budget_bytes
        
        # 2. Init Tables
        self.cache_modules = torch.nn.ModuleDict() # To register caches properly
        for name, cfg in table_config.items():
            self._init_table(name, cfg)

    def _init_table(self, name: str, cfg: EmbeddingConfig):
        sharding_type = self._infer_sharding_type(name)
        start, end = ShardingUtils.get_partition_bounds(
            sharding_type, cfg.num_embeddings, self.world_size, self.rank, name
        )
        local_rows = end - start

        # Determine Device & Dtype
        is_offload = sharding_type == ShardingType.CPU_OFFLOAD
        # If cache is enabled (budget > 0), we store master weights on CPU
        use_cache = (self.global_budget_bytes is not None and local_rows > 0 and self.device.type == "cuda")
        
        # Primary storage device
        storage_device = torch.device("cpu") if (is_offload or use_cache) else self.device
        
        raw_dtype = getattr(cfg, "data_type", "fp32")
        if hasattr(raw_dtype, "value"):
            key_dtype = str(raw_dtype.value).lower()
        elif isinstance(raw_dtype, str):
            key_dtype = raw_dtype.lower()
        else:
            key_dtype = str(raw_dtype).lower()
        if "fp16" in key_dtype or "half" in key_dtype:
            target_dtype = torch.float16
        elif "bf16" in key_dtype:
            target_dtype = torch.bfloat16
        elif "fp64" in key_dtype or "double" in key_dtype:
            target_dtype = torch.float64
        else:
            target_dtype = torch.float32

        # Create Master Embedding
        # Enable pin_memory if on CPU for faster transfer
        emb_layer = torch.nn.Embedding(local_rows, cfg.embedding_dim, device=storage_device, dtype=target_dtype)
        if local_rows > 0:
            torch.nn.init.uniform_(emb_layer.weight, -0.01, 0.01)
            # Experimental: Pin memory for faster CPU->GPU copy
            if storage_device.type == "cpu" and torch.cuda.is_available():
                 # PyTorch nn.Embedding doesn't support .pin_memory() on initialization easily.
                 # We can manually pin the tensor data.
                 emb_layer.weight.data = emb_layer.weight.data.pin_memory()

        cache_module = None
        if use_cache:
            # Heuristic: Allocate budget proportional to something or fixed?
            # Let's allocate a fixed % of rows for cache to demonstrate.
            # E.g. 10% of rows or fit within budget.
            element_size = 2 if target_dtype == torch.float16 else 4
            row_bytes = cfg.embedding_dim * element_size
            
            # Simple Logic: If budget is provided, try to fit. 
            # If multiple tables, this simple logic might oversubscribe if we don't divide.
            # Let's assume budget is *Per Table* or we divide it.
            # Improved: Divide global budget by number of tables.
            per_table_budget = self.global_budget_bytes // len(self._constraints) if self._constraints else self.global_budget_bytes
            
            max_entries = per_table_budget // row_bytes
            # Clamp cache size
            max_entries = min(max_entries, local_rows)
            max_entries = max(max_entries, 1) # At least 1 slot

            if self.rank == 0:
                logger.info(f"Enable Cache for {name}: {max_entries} slots ({max_entries*row_bytes/1024**2:.2f} MB)")

            cache_module = VectorizedEmbeddingCache(
                name, local_rows, cfg.embedding_dim, max_entries, self.device, target_dtype
            )
            self.cache_modules[name] = cache_module

        self.tables[name] = LocalEmbeddingTable(
            name=name,
            weight=emb_layer,
            sharding_type=sharding_type,
            global_num_embeddings=cfg.num_embeddings,
            row_start=start,
            row_end=end,
            cache=cache_module
        )
        
        for feat in (cfg.feature_names or []):
            self.feature_to_table[feat] = name

    def _infer_sharding_type(self, table_name: str) -> ShardingType:
        if pc := self._constraints.get(table_name):
            if sts := getattr(pc, "sharding_types", None):
                return ShardingUtils.parse_type(sts[0])
        return ShardingType.ROW_WISE

    def forward(self, kjt: KeyedJaggedTensor) -> Dict[str, torch.Tensor]:
        results = {}
        # Handle list of keys or whatever KJT provides
        feature_keys = kjt.keys() 
        if not isinstance(feature_keys, list):
            feature_keys = list(feature_keys)

        for key in feature_keys:
            table_name = self.feature_to_table.get(key)
            if not table_name: continue
            
            table = self.tables[table_name]
            # Use .values() to get the 1D tensor of indices
            indices = kjt[key].values()

            # Ensure indices are on GPU for routing calculation
            if indices.device != self.device:
                indices = indices.to(self.device, non_blocking=True)

            # --- Dispatch Logic ---
            should_direct = (self.world_size == 1) or \
                            (table.sharding_type in (ShardingType.REPLICATED, ShardingType.CPU_OFFLOAD))
            
            if should_direct:
                if table.global_num_embeddings > 0:
                     indices = torch.remainder(indices, table.global_num_embeddings)
                embeddings = self._lookup_local_direct(indices, table)
            else:
                embeddings = self._lookup_distributed(indices, table)
            
            results[key] = embeddings
        return results

    def _lookup_local_direct(self, indices: torch.Tensor, table: LocalEmbeddingTable) -> torch.Tensor:
        """Single-card or Replicated lookup."""
        if table.cache:
            # Vectorized Cache Lookup
            # Indices here are global, but for single card/replicated, global==local usually.
            # But wait, if row_wise and world_size=1, global=local.
            # If cpu_offload, we verify constraints.
            
            # Safe guard: Convert to local range if needed (usually identity for single GPU)
            local_idx = ShardingUtils.global_to_local(indices, table.sharding_type, table.row_start, table.weight.num_embeddings)
            return table.cache(local_idx, table.weight)
        
        # No Cache: Move to where weight is
        weight_device = table.weight.weight.device
        if weight_device != indices.device:
            idx_remote = indices.to(weight_device, non_blocking=True)
            embs = table.weight(idx_remote)
            return embs.to(self.device, non_blocking=True)
        return table.weight(indices)

    def _lookup_distributed(self, global_indices: torch.Tensor, table: LocalEmbeddingTable) -> torch.Tensor:
        # 0. Safety Guard
        if table.global_num_embeddings > 0:
            global_indices = torch.remainder(global_indices, table.global_num_embeddings)

        # 1. Route
        dest_ranks = self._get_dest_ranks(global_indices, table)

        # 2. Shuffle Indices
        indices_sorted, sort_idx, send_splits = self._sort_by_dest(global_indices, dest_ranks)
        recv_indices, recv_splits = self._exchange_data(indices_sorted, send_splits, dtype=torch.int64)

        # 3. Local Lookup
        local_indices = ShardingUtils.global_to_local(
            recv_indices, table.sharding_type, table.row_start, table.weight.num_embeddings
        )

        if table.cache:
            local_embs = table.cache(local_indices, table.weight)
        else:
            # Fallback for distributed but no cache (e.g. pure model parallel on GPU)
            # If weight is on CPU (CPU offload distributed), this handles it too.
            weight_device = table.weight.weight.device
            if weight_device != self.device:
                idx_host = local_indices.to(weight_device, non_blocking=True)
                local_embs = table.weight(idx_host).to(self.device, non_blocking=True)
            else:
                local_embs = table.weight(local_indices)

        # 4. Shuffle Embeddings Back
        recv_embs_sorted, _ = self._exchange_data(
            local_embs, 
            send_splits=recv_splits, 
            output_splits_hint=send_splits,
            dtype=local_embs.dtype
        )

        # 5. Restore Order
        return self._restore_order(recv_embs_sorted, sort_idx)

    # --- Communication Helpers ---

    def _get_dest_ranks(self, indices: torch.Tensor, table: LocalEmbeddingTable) -> torch.Tensor:
        if table.sharding_type == ShardingType.TABLE_WISE:
            owner = zlib.adler32(table.name.encode()) % self.world_size
            return torch.full_like(indices, owner, dtype=torch.int64)
        return ShardingUtils.get_dest_ranks_row_wise(indices, table.global_num_embeddings, self.world_size)

    def _sort_by_dest(self, indices: torch.Tensor, dest_ranks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        sort_idx = torch.argsort(dest_ranks)
        indices_sorted = indices[sort_idx].contiguous()
        splits = torch.bincount(dest_ranks, minlength=self.world_size).tolist()
        return indices_sorted, sort_idx, splits

    def _exchange_data(self, data: torch.Tensor, send_splits: List[int], output_splits_hint: Optional[List[int]] = None, dtype: torch.dtype = torch.float32) -> Tuple[torch.Tensor, List[int]]:
        if output_splits_hint:
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
        
        if self.world_size > 1:
            dist.all_to_all_single(
                recv_data, data,
                output_split_sizes=recv_splits,
                input_split_sizes=send_splits
            )
        else:
            recv_data = data # No copy if single rank logic allows

        return recv_data, recv_splits

    def _restore_order(self, data_sorted: torch.Tensor, sort_idx: torch.Tensor) -> torch.Tensor:
        if data_sorted.shape[0] == 0: return data_sorted
        inv_perm = torch.empty_like(sort_idx)
        inv_perm.index_copy_(0, sort_idx, torch.arange(len(sort_idx), device=self.device, dtype=sort_idx.dtype))
        return data_sorted.index_select(0, inv_perm)

    def record_cache_usage(self):
        pass # Optional logging hook

    def cache_stats(self) -> Dict[str, float]:
        # Aggregate stats from all tables
        hits = misses = evictions = 0
        for mod in self.cache_modules.values():
            s = mod.stats()
            hits += s['hits']
            misses += s['misses']
            evictions += s['evictions']
        
        # Simple Mem calculation
        total_mem = 0
        for mod in self.cache_modules.values():
            total_mem += mod.cache_data.numel() * mod.cache_data.element_size()
            
        return {
            "avg_mb": total_mem / 1024**2,
            "max_mb": total_mem / 1024**2,
            "hits": hits, 
            "misses": misses,
            "evictions": evictions
        }

    def preload_hot_ids(self, mapping: Dict[str, torch.Tensor]) -> None:
        """Preload specific IDs into cache to warm it up."""
        with torch.no_grad():
            for name, ids in mapping.items():
                if name in self.tables and self.tables[name].cache:
                    table = self.tables[name]
                    # Convert to local IDs
                    ids = ids.to(self.device)
                    local_ids = ShardingUtils.global_to_local(ids, table.sharding_type, table.row_start, table.weight.num_embeddings)
                    # Force a lookup to trigger load
                    table.cache(local_ids, table.weight)
                    logger.info(f"Preloaded {len(ids)} IDs into table '{name}'")