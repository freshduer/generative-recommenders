#!/usr/bin/env python3
"""
Advanced HSTU / Transformer GPU Latency Simulator (with interconnect bandwidth model)

New features added:
- Explicit GPU interconnect bandwidth and latency modeling (PCIe vs NVLink).
- Dynamic HtoD transfer time computed from transferred bytes and interconnect bandwidth.
- Allgather / multi-GPU communication time computed from message size and interconnect bandwidth.
- CLI flags to override PCIe/NVLink bandwidth (GB/s) and interconnect latency (ms).

Assumptions / simplifications (documented so you can tweak):
- Embedding row stored as fp16 (2 bytes per element).
- Lookups-per-batch per table approximated by `batch_size * seq_len * table_access_prob`.
- Only misses trigger CPU->GPU transfers; hits are local GPU accesses.
- Allgather transfers approximate total bytes required to share missing embedding vectors across GPUs.
- This is a performance model for comparative analysis — tune bandwidth/latency parameters using measured data.

Usage: similar to previous script, with new CLI flags:
    --pcie-bandwidth  (GB/s), --nvlink-bandwidth (GB/s), --interconnect-latency-ms (ms)

"""

import argparse
import json
import csv
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import math
import sys
import os

# plotting only when available at runtime
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


@dataclass
class WorkloadConfig:
    batch_size: int
    seq_len: float
    tables: List[Dict]  # [{'rows': int, 'dim': int, 'access_prob': float (optional)}]
    hot_ratio: float = 0.1
    hot_access: float = 0.9


@dataclass
class SystemConfig:
    num_gpus: int = 1
    gpu_mem_total_gb: float = 40.0
    interconnect: str = 'pcie'  # 'pcie' or 'nvlink'
    pcie_bandwidth_gbs: float = 16.0  # GB/s, per-direction effective bandwidth (default conservative)
    nvlink_bandwidth_gbs: float = 150.0  # GB/s effective
    interconnect_latency_ms: float = 0.5  # ms per hop/round (approx)


@dataclass
class BaselineLatency:
    emb_cpu_lookup_ms: float = 19.5  # CPU lookup processing time (ms)
    emb_htod_ms_overhead: float = 1.0  # fixed overhead added to bandwidth-based HtoD transfer (ms)
    emb_gpu_ms: float = 1.36
    kv_prefill_ms: float = 67.0
    kv_delta_ms: float = 30.0
    allgather_ms_per_card: float = 1.0  # legacy fallback; now computed dynamically if interconnect BW provided

    @classmethod
    def from_dict(cls, d: Dict):
        return cls(**{k: d[k] for k in d if k in cls.__annotations__})


class Simulator:
    def __init__(self, wl: WorkloadConfig, sys_cfg: SystemConfig, baseline: BaselineLatency):
        self.wl = wl
        self.sys = sys_cfg
        self.base = baseline

    def _table_size_gb(self, table: Dict) -> float:
        rows = int(table['rows'])
        dim = int(table['dim'])
        # fp16 (2 bytes)
        return rows * dim * 2 / (1024**3)

    def _kv_requirement_gb(self) -> float:
        layers = 8
        heads = 4
        head_dim = 128
        kv_dtype_bytes = 2
        kv_req_bytes = self.wl.batch_size * self.wl.seq_len * 2 * layers * heads * head_dim * kv_dtype_bytes
        return kv_req_bytes / (1024**3)

    def _effective_gpu_latency(self) -> float:
        # per-table GPU latency including legacy per-card penalty (kept for compatibility)
        return self.base.emb_gpu_ms + self.base.allgather_ms_per_card * max(0, self.sys.num_gpus - 1)

    def _interconnect_bandwidth(self) -> float:
        if self.sys.interconnect == 'nvlink':
            return self.sys.nvlink_bandwidth_gbs
        return self.sys.pcie_bandwidth_gbs

    def _bytes_for_table_misses(self, table: Dict, miss_count: int) -> int:
        # number of bytes that must be transferred from CPU to GPU for misses for a table
        dim = int(table['dim'])
        row_bytes = dim * 2
        return miss_count * row_bytes

    def _estimate_lookups_per_table(self, table: Dict) -> int:
        # approximate number of lookups per batch that target this table
        access_prob = float(table.get('access_prob', 1.0))
        # assume each token in seq accesses an embedding (conservative)
        lookups = int(self.wl.batch_size * self.wl.seq_len * access_prob)
        return max(1, lookups)

    def simulate_embedding_tables(self, gpu_cache_per_table_gb: List[float]) -> Tuple[float, float, float]:
        """
        Simulate latency contribution from embeddings given per-table cache (GB) assigned to GPU cache.
        Returns (emb_lat_total_ms, avg_hit_rate, total_cache_allocated_gb)
        """
        emb_lat_total = 0.0
        hit_accumulator = 0.0
        total_cache = 0.0

        t_gpu = self._effective_gpu_latency()

        inter_bw_gbs = self._interconnect_bandwidth()
        inter_latency_ms = self.sys.interconnect_latency_ms

        for i, table in enumerate(self.wl.tables):
            rows = int(table['rows'])
            dim = int(table['dim'])
            row_bytes = dim * 2
            table_size_gb = self._table_size_gb(table)

            # compute how many rows fit into the provided cache (global across GPUs)
            cache_gb = gpu_cache_per_table_gb[i]
            cache_rows_total = int(cache_gb * self.sys.num_gpus * (1024**3) / row_bytes)

            # hot set and per-table access intensity
            hot_rows = int(rows * self.wl.hot_ratio)
            table_access_prob = float(table.get('access_prob', 1.0))

            if hot_rows <= 0:
                hit_rate = 0.0
            else:
                if cache_rows_total >= hot_rows:
                    cold_rows = rows - hot_rows
                    remaining = cache_rows_total - hot_rows
                    cold_hit = (remaining / cold_rows) if cold_rows > 0 else 1.0
                    cold_hit = min(1.0, cold_hit)
                    hit_rate = self.wl.hot_access * table_access_prob + cold_hit * (1 - self.wl.hot_access) * table_access_prob
                else:
                    frac = cache_rows_total / max(1, hot_rows)
                    hit_rate = frac * self.wl.hot_access * table_access_prob

            # estimate number of lookups for this table in a batch and misses
            lookups = self._estimate_lookups_per_table(table)
            miss_count = int((1 - hit_rate) * lookups)

            # bytes to transfer for misses (CPU->GPU); assume each missed row must be transferred once per batch
            bytes_to_transfer = self._bytes_for_table_misses(table, miss_count)

            # compute HtoD transfer time using interconnect bandwidth (GB/s)
            if bytes_to_transfer <= 0:
                t_cpu_htod = 0.0
            else:
                seconds = bytes_to_transfer / (inter_bw_gbs * (1024**3))
                t_bandwidth_ms = seconds * 1000.0
                # combine measured CPU lookup processing overhead and an HtoD overhead
                t_cpu_htod = self.base.emb_cpu_lookup_ms + self.base.emb_htod_ms_overhead + t_bandwidth_ms

            # estimate allgather cost for redistributing missing vectors across GPUs
            # approximate allgather message = bytes_to_transfer * (num_gpus - 1) / num_gpus (simplified)
            if self.sys.num_gpus > 1 and bytes_to_transfer > 0:
                msg_bytes = bytes_to_transfer * (self.sys.num_gpus - 1) / max(1, self.sys.num_gpus)
                sec_allgather = msg_bytes / (inter_bw_gbs * (1024**3))
                t_allgather_ms = sec_allgather * 1000.0 + inter_latency_ms * (self.sys.num_gpus - 1)
            else:
                t_allgather_ms = 0.0

            # final average latency per table = hit portion uses GPU; miss portion pays CPU+HtoD+allgather
            avg_lat = hit_rate * t_gpu + (1 - hit_rate) * (t_cpu_htod + t_allgather_ms)

            emb_lat_total += avg_lat
            hit_accumulator += hit_rate
            total_cache += cache_gb * self.sys.num_gpus

        avg_hit = hit_accumulator / len(self.wl.tables)
        return emb_lat_total, avg_hit, total_cache

    def simulate_kv_cache(self, kv_gpu_gb: float, kv_req_gb: float) -> Tuple[float, float, float]:
        kv_hit_rate = min(1.0, kv_gpu_gb * self.sys.num_gpus / max(1e-9, kv_req_gb))
        avg_kv_latency = kv_hit_rate * self.base.kv_delta_ms + (1 - kv_hit_rate) * self.base.kv_prefill_ms
        return avg_kv_latency, kv_hit_rate, kv_gpu_gb * self.sys.num_gpus

    def run_split(self, emb_gb_per_gpu: float, kv_gb_per_gpu: float, allocation_strategy: str = 'uniform', custom_table_alloc: Optional[List[float]] = None) -> Dict:
        # ensure within per-gpu memory
        if emb_gb_per_gpu + kv_gb_per_gpu > self.sys.gpu_mem_total_gb:
            scale = self.sys.gpu_mem_total_gb / (emb_gb_per_gpu + kv_gb_per_gpu)
            emb_gb_per_gpu *= scale
            kv_gb_per_gpu *= scale

        # allocate embedding cache per table
        if custom_table_alloc is not None:
            if len(custom_table_alloc) != len(self.wl.tables):
                raise ValueError("custom_table_alloc length must equal number of tables")
            emb_cache_per_table = custom_table_alloc
        elif allocation_strategy == 'uniform':
            emb_cache_per_table = [emb_gb_per_gpu / len(self.wl.tables)] * len(self.wl.tables)
        elif allocation_strategy == 'proportional':
            sizes = [self._table_size_gb(t) for t in self.wl.tables]
            total = sum(sizes)
            if total <= 0:
                emb_cache_per_table = [emb_gb_per_gpu / len(self.wl.tables)] * len(self.wl.tables)
            else:
                emb_cache_per_table = [emb_gb_per_gpu * (s / total) for s in sizes]
        else:
            raise ValueError(f"Unknown allocation_strategy: {allocation_strategy}")

        emb_lat, emb_hit, emb_alloc_total = self.simulate_embedding_tables(emb_cache_per_table)
        kv_req = self._kv_requirement_gb()
        kv_lat, kv_hit, kv_alloc_total = self.simulate_kv_cache(kv_gb_per_gpu, kv_req)

        total = emb_lat + kv_lat
        return {
            'emb_gb_per_gpu': emb_gb_per_gpu,
            'kv_gb_per_gpu': kv_gb_per_gpu,
            'emb_lat_ms': emb_lat,
            'emb_hit': emb_hit,
            'emb_alloc_total_gb': emb_alloc_total,
            'kv_lat_ms': kv_lat,
            'kv_hit': kv_hit,
            'kv_alloc_total_gb': kv_alloc_total,
            'total_ms': total,
            'emb_ratio': emb_lat / max(1e-9, total),
            'kv_ratio': kv_lat / max(1e-9, total),
            'kv_req_gb': kv_req
        }


# ----------------- Utilities -----------------

def parse_splits(splits_arg: List[str]) -> List[Tuple[float, float]]:
    parsed = []
    for s in splits_arg:
        if isinstance(s, str) and ':' in s:
            a, b = s.split(':')
            try:
                parsed.append((float(a), float(b)))
            except:
                continue
    return parsed


def save_csv(path: str, rows: List[Dict]):
    keys = list(rows[0].keys()) if rows else []
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def plot_results(rows: List[Dict], out_path: str):
    if plt is None:
        print("matplotlib not available; skipping plotting")
        return
    x = [r['emb_gb_per_gpu'] for r in rows]
    y = [r['total_ms'] for r in rows]
    emb_hits = [r['emb_hit'] for r in rows]

    fig, ax1 = plt.subplots()
    ax1.plot(x, y, marker='o')
    ax1.set_xlabel('Embedding GB per GPU')
    ax1.set_ylabel('Total Latency (ms)')
    ax1.set_title('Latency vs Embedding Allocation (per GPU)')

    ax2 = ax1.twinx()
    ax2.plot(x, emb_hits, marker='x')
    ax2.set_ylabel('Avg Emb Hit Rate')

    fig.tight_layout()
    fig.savefig(out_path)
    print(f"Saved plot to {out_path}")


# ----------------- CLI -----------------

def main():
    parser = argparse.ArgumentParser(description='Advanced GPU Latency Simulator (with interconnect model)')
    parser.add_argument('--workload', type=str, required=False, help='JSON file describing workload (see README in header)')
    parser.add_argument('--baseline', type=str, required=False, help='JSON file providing measured baseline latencies (optional)')
    parser.add_argument('--num-gpus', type=int, default=1)
    parser.add_argument('--gpu-mem', type=float, default=40.0)
    parser.add_argument('--interconnect', type=str, choices=['pcie', 'nvlink'], default='pcie')
    parser.add_argument('--pcie-bandwidth', type=float, default=16.0, help='PCIe effective bandwidth in GB/s')
    parser.add_argument('--nvlink-bandwidth', type=float, default=150.0, help='NVLink effective bandwidth in GB/s')
    parser.add_argument('--interconnect-latency-ms', type=float, default=0.5, help='Interconnect latency per hop in ms')
    parser.add_argument('--sweep', type=str, default=None, help='Comma-separated splits OR path to CSV with emb,kv columns. Example: "5:35,10:30,20:20"')
    parser.add_argument('--alloc-strategy', type=str, choices=['uniform', 'proportional', 'custom'], default='uniform')
    parser.add_argument('--custom-alloc', type=str, default=None, help='Comma-separated per-table GB allocations (only if --alloc-strategy custom)')
    parser.add_argument('--out', type=str, default='sim_results.csv')
    parser.add_argument('--plot', type=str, default='sim_results.png')
    args = parser.parse_args()

    # load workload
    if args.workload:
        with open(args.workload, 'r') as f:
            wjson = json.load(f)
        wl = WorkloadConfig(
            batch_size = int(wjson.get('batch_size', 16)),
            seq_len = float(wjson.get('seq_len', 11713)),
            tables = wjson.get('tables', []),
            hot_ratio = float(wjson.get('hot_ratio', 0.1)),
            hot_access = float(wjson.get('hot_access', 0.9))
        )
    else:
        # default small workload if none provided
        tables = [{'rows': 250_000_000, 'dim': 256} for _ in range(24)]
        wl = WorkloadConfig(batch_size=16, seq_len=11713, tables=tables)

    # baseline
    baseline = BaselineLatency()
    if args.baseline:
        with open(args.baseline, 'r') as f:
            bjson = json.load(f)
        baseline = BaselineLatency.from_dict(bjson)

    sys_cfg = SystemConfig(num_gpus=args.num_gpus, gpu_mem_total_gb=args.gpu_mem, interconnect=args.interconnect,
                           pcie_bandwidth_gbs=args.pcie_bandwidth, nvlink_bandwidth_gbs=args.nvlink_bandwidth,
                           interconnect_latency_ms=args.interconnect_latency_ms)

    sim = Simulator(wl, sys_cfg, baseline)

    # parse sweep
    splits = []
    if args.sweep:
        if os.path.exists(args.sweep):
            with open(args.sweep, 'r') as f:
                reader = csv.DictReader(f)
                for r in reader:
                    emb = float(r.get('emb', r.get('emb_gb_per_gpu', 0)))
                    kv = float(r.get('kv', r.get('kv_gb_per_gpu', 0)))
                    splits.append((emb, kv))
        else:
            items = [it.strip() for it in args.sweep.split(',') if it.strip()]
            for it in items:
                if ':' in it:
                    a, b = it.split(':')
                    try:
                        splits.append((float(a), float(b)))
                    except:
                        pass

    if not splits:
        splits = [(5,35),(10,30),(20,20),(25,15),(30,10),(35,5)]

    custom_alloc = None
    if args.alloc_strategy == 'custom':
        if not args.custom_alloc:
            raise ValueError('custom allocation strategy requires --custom-alloc')
        parts = [float(x) for x in args.custom_alloc.split(',')]
        custom_alloc = parts

    rows = []
    for emb_gb, kv_gb in splits:
        row = sim.run_split(emb_gb, kv_gb, allocation_strategy=args.alloc_strategy, custom_table_alloc=custom_alloc)
        row['split'] = f"{emb_gb}:{kv_gb}"
        rows.append(row)

    save_csv(args.out, rows)
    print(f"Saved results to {args.out}")

    if plt is not None:
        try:
            plot_results(rows, args.plot)
        except Exception as e:
            print('Plotting failed:', e)

    # print table summary (compact)
    print('Summary:')
    print(f"{'split':<10} {'emb_gb':>7} {'kv_gb':>7} {'emb_hit%':>8} {'kv_hit%':>7} {'emb_lat':>9} {'kv_lat':>8} {'total':>8}")
    for r in rows:
        print(f"{r['split']:<10} {r['emb_gb_per_gpu']:7.1f} {r['kv_gb_per_gpu']:7.1f} {r['emb_hit']*100:8.1f} {r['kv_hit']*100:7.1f} {r['emb_lat_ms']:9.2f} {r['kv_lat_ms']:8.2f} {r['total_ms']:8.2f}")


if __name__ == '__main__':
    main()
