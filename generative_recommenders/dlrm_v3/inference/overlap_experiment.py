# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
H2D and Compute Overlap Experiment Module.

This module implements an experiment to measure how H2D (Host-to-Device) transfers
can run in parallel with GPU compute without interfering with compute performance.

Key Concepts:
- Compute runs on HIGH-priority CUDA stream (gets GPU SM resources first)
- H2D runs on LOW-priority CUDA stream (uses DMA engine, shouldn't interfere)
- Both streams run simultaneously (true parallelism)

Metrics Explained:
- Interference: (overlap_compute - baseline_compute) / baseline_compute
  - Should be < 5% for successful parallel execution
- Overlap Ratio: How much H2D data can be transferred during compute time
  - = min(1, compute_time / h2d_time) * 100%
  - If compute takes longer than H2D, overlap can be 100%
  - If H2D takes longer, overlap = compute_time / h2d_time

Plots:
1. Compute Time Comparison: Check if H2D interferes with compute (should be similar)
2. H2D Overlap Ratio: How much of 10GB H2D finished during compute
3. H2D Bandwidth Benchmark: Transfer times for different data sizes
4. Compute Time Series: Per-batch compute times
5. Timeline: Visual comparison of sequential vs parallel execution
6. H2D Data Transferred: Actual GB transferred during compute
"""

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class OverlapMetrics:
    """Metrics for a single batch in the overlap experiment."""
    batch_idx: int
    overlap_enabled: bool
    compute_time_ms: float
    h2d_time_ms: float
    h2d_overlapped_time_ms: float  # Time of H2D that overlapped with compute
    bubble_utilization: float  # h2d_time / compute_time * 100 (bubble利用率)
    total_time_ms: float


@dataclass
class OverlapExperimentStats:
    """Statistics for the overlap experiment."""
    # Non-overlap batches (baseline)
    no_overlap_compute_times: List[float] = field(default_factory=list)
    no_overlap_h2d_times: List[float] = field(default_factory=list)
    no_overlap_total_times: List[float] = field(default_factory=list)
    
    # Overlap batches
    overlap_compute_times: List[float] = field(default_factory=list)
    overlap_h2d_times: List[float] = field(default_factory=list)
    overlap_h2d_overlapped_times: List[float] = field(default_factory=list)
    bubble_utilizations: List[float] = field(default_factory=list)  # bubble利用率
    overlap_total_times: List[float] = field(default_factory=list)
    
    # H2D transfer stats
    h2d_data_sizes_mb: List[float] = field(default_factory=list)
    h2d_bandwidths_gbps: List[float] = field(default_factory=list)
    chunks_completed_list: List[int] = field(default_factory=list)
    total_chunks_list: List[int] = field(default_factory=list)
    
    # Per-batch H2D transferred (for accumulation tracking)
    h2d_transferred_per_batch_gb: List[float] = field(default_factory=list)
    
    def clear(self):
        """Clear all statistics."""
        self.no_overlap_compute_times.clear()
        self.no_overlap_h2d_times.clear()
        self.no_overlap_total_times.clear()
        self.overlap_compute_times.clear()
        self.overlap_h2d_times.clear()
        self.overlap_h2d_overlapped_times.clear()
        self.bubble_utilizations.clear()
        self.overlap_total_times.clear()
        self.h2d_data_sizes_mb.clear()
        self.h2d_bandwidths_gbps.clear()
        self.chunks_completed_list.clear()
        self.total_chunks_list.clear()
        self.h2d_transferred_per_batch_gb.clear()


class OverlapExperiment:
    """
    Experiment to measure H2D and compute overlap.
    
    This class manages:
    - Two CUDA streams with different priorities (high for compute, low for H2D)
    - Chunked H2D transfers with event-based monitoring
    - Metrics collection and visualization
    """
    
    def __init__(
        self,
        device: torch.device,
        output_dir: str = "overlap_experiment_results",
        chunk_size_mb: float = 16.0,  # Size of each H2D chunk in MB
        switch_interval: int = 50,  # Switch between overlap/no-overlap every N batches
    ):
        """
        Initialize the overlap experiment.
        
        Args:
            device: CUDA device to use
            output_dir: Directory to save plots
            chunk_size_mb: Size of each H2D transfer chunk in MB
            switch_interval: Number of batches before switching overlap mode
        """
        self.device = device
        self.output_dir = output_dir
        self.chunk_size_bytes = int(chunk_size_mb * 1024 * 1024)
        self.switch_interval = switch_interval
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Create streams with different priorities
        # Lower number = higher priority
        high_priority, low_priority = torch.cuda.Stream.priority_range()
        self.compute_stream = torch.cuda.Stream(device=device, priority=high_priority)
        self.h2d_stream = torch.cuda.Stream(device=device, priority=low_priority)
        
        # Experiment state
        self.batch_count = 0
        self.stats = OverlapExperimentStats()
        self.all_metrics: List[OverlapMetrics] = []
        self.plot_count = 0
        
        # Pre-allocate dummy tensor for simulated H2D overlap
        self._dummy_src: Optional[torch.Tensor] = None
        self._dummy_dst: Optional[torch.Tensor] = None
        
        # Cumulative H2D tracking for 10GB target
        self.target_h2d_gb = 10.0  # Total data to transfer (10GB)
        self.cumulative_h2d_gb = 0.0  # Data transferred so far
        self.cumulative_h2d_batches = 0  # Batches used for this 10GB
        self.cumulative_h2d_time_ms = 0.0  # Total H2D time (should be ~0 if hidden in compute)
        self.cumulative_compute_time_ms = 0.0  # Total compute time during H2D
        self.h2d_transfer_records: List[Dict] = []  # Record each 10GB completion
        
        # Default H2D bandwidth estimation (will be updated after benchmark)
        self.h2d_bytes_per_ms = 20 * 1024 * 1024  # Default 20 GB/s = 20MB/ms
        
        # Estimated compute time (will be updated with actual)
        self.estimated_compute_time_ms = 80.0  # Initial estimate
        self.compute_time_history: List[float] = []
        
        logger.info(f"[OverlapExperiment] Initialized with chunk_size={chunk_size_mb}MB, "
                   f"switch_interval={switch_interval}, output_dir={output_dir}")
        logger.info(f"[OverlapExperiment] Stream priorities: compute={high_priority}, h2d={low_priority}")
        
        # Run H2D bandwidth benchmark
        self.h2d_benchmark_results = {}  # Initialize before benchmark
        self._run_h2d_bandwidth_benchmark()
    
    def _run_h2d_bandwidth_benchmark(self):
        """Run H2D bandwidth benchmark for different data sizes."""
        logger.info("[OverlapExperiment] Running H2D bandwidth benchmark...")
        
        test_sizes_gb = [0.5, 1.0, 2.0, 5.0, 10.0]
        self.h2d_benchmark_results = {}
        
        for size_gb in test_sizes_gb:
            size_bytes = int(size_gb * 1024 * 1024 * 1024)
            num_elements = size_bytes // 4  # float32
            
            # Allocate pinned memory on CPU and GPU memory
            try:
                src = torch.randn(num_elements, dtype=torch.float32, pin_memory=True)
                dst = torch.empty(num_elements, dtype=torch.float32, device=self.device)
            except RuntimeError as e:
                logger.warning(f"[OverlapExperiment] Cannot allocate {size_gb}GB: {e}")
                continue
            
            # Warmup
            dst.copy_(src, non_blocking=False)
            torch.cuda.synchronize()
            
            # Measure transfer time
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            dst.copy_(src, non_blocking=True)
            end_event.record()
            torch.cuda.synchronize()
            
            transfer_time_ms = start_event.elapsed_time(end_event)
            bandwidth_gbps = (size_gb * 8) / (transfer_time_ms / 1000)  # Gbps
            
            # Calculate chunks needed
            num_chunks = size_bytes // self.chunk_size_bytes
            
            self.h2d_benchmark_results[size_gb] = {
                'time_ms': transfer_time_ms,
                'bandwidth_gbps': bandwidth_gbps,
                'num_chunks': num_chunks,
            }
            
            logger.info(f"[OverlapExperiment] H2D {size_gb}GB: {transfer_time_ms:.2f}ms, "
                       f"{bandwidth_gbps:.2f}Gbps, {num_chunks} chunks")
            
            # Free memory
            del src, dst
            torch.cuda.empty_cache()
        
        # Update H2D bandwidth estimate from benchmark results
        if 1.0 in self.h2d_benchmark_results:
            bw_gbps = self.h2d_benchmark_results[1.0]['bandwidth_gbps']
            self.h2d_bytes_per_ms = bw_gbps * 1e9 / 8 / 1000  # Convert Gbps to bytes/ms
            logger.info(f"[OverlapExperiment] H2D bandwidth: {bw_gbps:.2f}Gbps = {self.h2d_bytes_per_ms/1024/1024:.2f}MB/ms")
        
        logger.info("[OverlapExperiment] H2D bandwidth benchmark complete")
    
    def should_overlap(self) -> bool:
        """Determine if current batch should use overlap mode."""
        # Every switch_interval batches, alternate between overlap/no-overlap
        cycle_position = self.batch_count % (2 * self.switch_interval)
        return cycle_position >= self.switch_interval
    
    def _ensure_dummy_tensors(self, size_bytes: int):
        """Ensure dummy tensors are allocated for simulated H2D."""
        num_elements = size_bytes // 4  # float32
        if self._dummy_src is None or self._dummy_src.numel() < num_elements:
            self._dummy_src = torch.randn(num_elements, dtype=torch.float32, pin_memory=True)
            self._dummy_dst = torch.empty(num_elements, dtype=torch.float32, device=self.device)
    
    def run_overlapped_h2d_with_compute(
        self,
        compute_fn,
        compute_args: dict,
        h2d_data_size: int,
    ) -> Tuple[Any, float, float, float, float, int, int]:
        """
        Run compute with H2D transfer in parallel - H2D STOPS when compute finishes.
        
        Key design:
        - Compute runs on HIGH priority stream (async)
        - H2D runs on LOW priority stream, one chunk at a time
        - After EACH chunk completes (sync), check if compute is done
        - If compute is done, STOP submitting more H2D
        - H2D is perfectly bounded by compute bubble
        
        This approach:
        - Each H2D chunk actually completes before checking compute status
        - No async queue flooding - strict control over H2D submission
        - H2D truly limited to what fits in compute time
        
        Args:
            compute_fn: Function to run for compute
            compute_args: Arguments for compute function
            h2d_data_size: Max H2D data size (will stop when compute finishes)
            
        Returns:
            Tuple of (compute_result, compute_time_ms, h2d_time_ms, overlapped_time_ms, 
                     overlap_ratio, chunks_completed, total_chunks)
        """
        # Prepare large enough buffer for potential H2D
        self._ensure_dummy_tensors(h2d_data_size)
        
        # Calculate max number of chunks
        max_chunks = max(1, h2d_data_size // self.chunk_size_bytes)
        chunk_elements = self.chunk_size_bytes // 4
        
        # Create events for timing
        compute_start_event = torch.cuda.Event(enable_timing=True)
        compute_end_event = torch.cuda.Event(enable_timing=True)
        h2d_start_event = torch.cuda.Event(enable_timing=True)
        h2d_end_event = torch.cuda.Event(enable_timing=True)
        chunk_event = torch.cuda.Event()  # For synchronizing each chunk
        
        # Launch compute on HIGH-priority stream (non-blocking, async)
        with torch.cuda.stream(self.compute_stream):
            compute_start_event.record()
            result = compute_fn(**compute_args)
            compute_end_event.record()
        
        # Submit H2D chunks one by one, checking compute status after EACH chunk completes
        chunks_completed = 0
        h2d_start_event.record(self.h2d_stream)
        
        # Check if compute is already done before starting H2D (shouldn't be)
        compute_already_done = compute_end_event.query()
        
        for i in range(max_chunks):
            # First check if compute is done BEFORE submitting next chunk
            if compute_end_event.query():
                # Compute is done, stop H2D to avoid overflow into next batch
                break
            
            # Submit one H2D chunk on h2d_stream
            with torch.cuda.stream(self.h2d_stream):
                start_idx = i * chunk_elements
                end_idx = min((i + 1) * chunk_elements, self._dummy_src.numel())
                if start_idx < end_idx:
                    self._dummy_dst[start_idx:end_idx].copy_(
                        self._dummy_src[start_idx:end_idx], non_blocking=True
                    )
                chunk_event.record()
            
            # CRITICAL: Wait for this chunk to actually complete on GPU
            # This ensures we don't flood the queue with async requests
            self.h2d_stream.synchronize()
            chunks_completed += 1
        
        h2d_end_event.record(self.h2d_stream)
        
        # Wait for compute to finish (may already be done)
        self.compute_stream.synchronize()
        
        # Calculate times using CUDA events
        compute_time = compute_start_event.elapsed_time(compute_end_event)  # ms
        
        if chunks_completed > 0:
            h2d_time = h2d_start_event.elapsed_time(h2d_end_event)
        else:
            h2d_time = 0.0
        
        # H2D is bounded by compute time (by design)
        overlapped_time = min(h2d_time, compute_time)
        
        # Calculate actual data transferred
        data_transferred = chunks_completed * self.chunk_size_bytes
        data_transferred_gb = data_transferred / (1024 ** 3)
        
        # Update cumulative H2D tracking
        self.cumulative_h2d_gb += data_transferred_gb
        self.cumulative_h2d_batches += 1
        self.cumulative_compute_time_ms += compute_time
        self.cumulative_h2d_time_ms += overlapped_time
        
        # Check if we've completed 10GB
        if self.cumulative_h2d_gb >= self.target_h2d_gb:
            self.h2d_transfer_records.append({
                'batches': self.cumulative_h2d_batches,
                'total_gb': self.cumulative_h2d_gb,
                'total_compute_ms': self.cumulative_compute_time_ms,
                'total_h2d_ms': self.cumulative_h2d_time_ms,
                'avg_per_batch_gb': self.cumulative_h2d_gb / self.cumulative_h2d_batches,
            })
            logger.info(f"[OverlapExperiment] Completed {self.target_h2d_gb}GB H2D in "
                       f"{self.cumulative_h2d_batches} batches, "
                       f"total compute time: {self.cumulative_compute_time_ms:.1f}ms, "
                       f"avg per batch: {self.cumulative_h2d_gb/self.cumulative_h2d_batches:.2f}GB")
            # Reset for next 10GB
            self.cumulative_h2d_gb = 0.0
            self.cumulative_h2d_batches = 0
            self.cumulative_compute_time_ms = 0.0
            self.cumulative_h2d_time_ms = 0.0
        
        # bubble_utilization = what % of compute time was used for H2D (bubble利用率)
        bubble_utilization = (h2d_time / compute_time * 100) if compute_time > 0 else 0
        
        # Debug log for every batch
        if self.batch_count % 10 == 0:
            logger.info(f"[OverlapExperiment] Batch {self.batch_count}: "
                       f"chunks={chunks_completed}/{max_chunks}, "
                       f"compute={compute_time:.1f}ms, h2d={h2d_time:.1f}ms, "
                       f"bubble_util={bubble_utilization:.1f}%, "
                       f"data={data_transferred_gb:.2f}GB")
        
        return result, compute_time, h2d_time, overlapped_time, bubble_utilization, chunks_completed, max_chunks
    
    def run_sequential_h2d_with_compute(
        self,
        compute_fn,
        compute_args: dict,
        h2d_data_size: int,
    ) -> Tuple[Any, float, float]:
        """
        Run compute without H2D overlap (baseline).
        
        Args:
            compute_fn: Function to run for compute
            compute_args: Arguments for compute function
            h2d_data_size: Total size of H2D data to simulate in bytes
            
        Returns:
            Tuple of (compute_result, compute_time_ms, h2d_time_ms)
        """
        self._ensure_dummy_tensors(h2d_data_size)
        
        # Run H2D first (sequential)
        t0_h2d = time.time()
        self._dummy_dst.copy_(self._dummy_src[:self._dummy_dst.numel()], non_blocking=False)
        torch.cuda.synchronize(self.device)
        h2d_time = (time.time() - t0_h2d) * 1000
        
        # Then run compute
        t0_compute = time.time()
        with torch.cuda.stream(self.compute_stream):
            result = compute_fn(**compute_args)
        self.compute_stream.synchronize()
        compute_time = (time.time() - t0_compute) * 1000
        
        return result, compute_time, h2d_time
    
    def record_metrics(
        self,
        overlap_enabled: bool,
        compute_time_ms: float,
        h2d_time_ms: float,
        h2d_overlapped_time_ms: float = 0.0,
        bubble_utilization: float = 0.0,  # h2d_time / compute_time * 100
        total_time_ms: float = 0.0,
        h2d_data_size_bytes: int = 0,
        chunks_completed: int = 0,
        total_chunks: int = 0,
    ):
        """Record metrics for the current batch."""
        metrics = OverlapMetrics(
            batch_idx=self.batch_count,
            overlap_enabled=overlap_enabled,
            compute_time_ms=compute_time_ms,
            h2d_time_ms=h2d_time_ms,
            h2d_overlapped_time_ms=h2d_overlapped_time_ms,
            bubble_utilization=bubble_utilization,
            total_time_ms=total_time_ms,
        )
        self.all_metrics.append(metrics)
        
        # Calculate H2D bandwidth
        h2d_data_size_mb = h2d_data_size_bytes / (1024 * 1024)
        h2d_bandwidth_gbps = (h2d_data_size_mb * 8 / 1024) / (h2d_time_ms / 1000) if h2d_time_ms > 0 else 0
        
        if overlap_enabled:
            self.stats.overlap_compute_times.append(compute_time_ms)
            self.stats.overlap_h2d_times.append(h2d_time_ms)
            self.stats.overlap_h2d_overlapped_times.append(h2d_overlapped_time_ms)
            self.stats.bubble_utilizations.append(bubble_utilization)
            self.stats.overlap_total_times.append(total_time_ms)
        else:
            self.stats.no_overlap_compute_times.append(compute_time_ms)
            self.stats.no_overlap_h2d_times.append(h2d_time_ms)
            self.stats.no_overlap_total_times.append(total_time_ms)
        
        # Record H2D stats
        self.stats.h2d_data_sizes_mb.append(h2d_data_size_mb)
        self.stats.h2d_bandwidths_gbps.append(h2d_bandwidth_gbps)
        self.stats.chunks_completed_list.append(chunks_completed)
        self.stats.total_chunks_list.append(total_chunks)
        
        # Record per-batch H2D transferred (in GB)
        h2d_transferred_gb = chunks_completed * self.chunk_size_bytes / (1024 ** 3)
        self.stats.h2d_transferred_per_batch_gb.append(h2d_transferred_gb)
        
        self.batch_count += 1
    
    def should_plot(self) -> bool:
        """Check if we should generate plots (every 100 batches)."""
        return self.batch_count > 0 and self.batch_count % 100 == 0
    
    def generate_plots(self):
        """Generate comparison plots for the last 100 batches."""
        if len(self.stats.no_overlap_compute_times) == 0 or len(self.stats.overlap_compute_times) == 0:
            logger.warning("[OverlapExperiment] Not enough data for plotting")
            return
        
        self.plot_count += 1
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'H2D-Compute Overlap Experiment (Batches {self.batch_count - 100} - {self.batch_count})', 
                     fontsize=14, fontweight='bold')
        
        # Plot 1: Compute Time Comparison
        ax1 = axes[0, 0]
        no_overlap_compute = self.stats.no_overlap_compute_times
        overlap_compute = self.stats.overlap_compute_times
        
        x = np.arange(2)
        width = 0.35
        means = [np.mean(no_overlap_compute), np.mean(overlap_compute)]
        stds = [np.std(no_overlap_compute), np.std(overlap_compute)]
        
        bars = ax1.bar(x, means, width, yerr=stds, capsize=5, 
                       color=['#2ecc71', '#3498db'], edgecolor='black')
        ax1.set_ylabel('Compute Time (ms)', fontsize=11)
        ax1.set_title('Compute Time: No Overlap vs Overlap', fontsize=12)
        ax1.set_xticks(x)
        ax1.set_xticklabels(['No Overlap\n(Baseline)', 'With Overlap'])
        ax1.axhline(y=means[0], color='#2ecc71', linestyle='--', alpha=0.5, label='Baseline mean')
        
        # Add value labels on bars
        for bar, mean, std in zip(bars, means, stds):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.5,
                    f'{mean:.2f}±{std:.2f}', ha='center', va='bottom', fontsize=10)
        
        # Calculate interference percentage
        if means[0] > 0:
            interference = (means[1] - means[0]) / means[0] * 100
            ax1.text(0.5, 0.95, f'Interference: {interference:+.2f}%', 
                    transform=ax1.transAxes, ha='center', va='top',
                    fontsize=11, color='red' if interference > 5 else 'green',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Plot 2: Bubble Utilization Time Series (only for overlap batches)
        ax2 = axes[0, 1]
        if self.stats.bubble_utilizations:
            # Get bubble utilization for the last 100 batches (only overlap-enabled ones)
            overlap_batch_indices = []
            chunks_completed_plot = []
            total_chunks_plot = []
            
            # Calculate bubble utilization (h2d_time / compute_time * 100)
            bubble_utils = []
            for i, m in enumerate(self.all_metrics[-100:]):
                if m.overlap_enabled:
                    overlap_batch_indices.append(i)
                    # Calculate bubble utilization for this batch
                    bubble_util = (m.h2d_time_ms / m.compute_time_ms * 100) if m.compute_time_ms > 0 else 0
                    bubble_utils.append(bubble_util)
            
            if bubble_utils:
                ax2.plot(range(len(bubble_utils)), bubble_utils, 
                        color='#9b59b6', linewidth=1.5, marker='o', markersize=3, alpha=0.7)
                ax2.axhline(y=np.mean(bubble_utils), color='red', 
                           linestyle='--', linewidth=2, label=f'Mean: {np.mean(bubble_utils):.1f}%')
                ax2.fill_between(range(len(bubble_utils)), bubble_utils, alpha=0.3, color='#9b59b6')
                ax2.set_ylim(0, 110)
                
                # Add reference line at 100%
                ax2.axhline(y=100, color='green', linestyle=':', linewidth=1, alpha=0.5, label='100% (ideal)')
            
            ax2.set_xlabel('Overlap Batch Index', fontsize=11)
            ax2.set_ylabel('Bubble Utilization (%)', fontsize=11)
            ax2.set_title('Bubble Utilization (H2D Time / Compute Time)', fontsize=12)
            ax2.legend(loc='upper right')
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: H2D Bandwidth Benchmark (from init)
        ax3 = axes[0, 2]
        if hasattr(self, 'h2d_benchmark_results') and self.h2d_benchmark_results:
            sizes = list(self.h2d_benchmark_results.keys())
            times = [self.h2d_benchmark_results[s]['time_ms'] for s in sizes]
            bandwidths = [self.h2d_benchmark_results[s]['bandwidth_gbps'] for s in sizes]
            chunks = [self.h2d_benchmark_results[s]['num_chunks'] for s in sizes]
            
            # Create bar chart for transfer times
            colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(sizes)))
            bars = ax3.bar(range(len(sizes)), times, color=colors, edgecolor='black')
            ax3.set_xticks(range(len(sizes)))
            ax3.set_xticklabels([f'{s}GB' for s in sizes])
            ax3.set_xlabel('Data Size', fontsize=11)
            ax3.set_ylabel('Transfer Time (ms)', fontsize=11)
            ax3.set_title('H2D Transfer Time by Data Size', fontsize=12)
            
            # Add bandwidth and chunks annotation on bars
            for i, (bar, bw, ch, t) in enumerate(zip(bars, bandwidths, chunks, times)):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                        f'{t:.0f}ms\n{bw:.1f}Gbps\n{ch} chunks', 
                        ha='center', va='bottom', fontsize=8)
        else:
            ax3.text(0.5, 0.5, 'No benchmark data', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('H2D Transfer Time by Data Size', fontsize=12)
        
        # Plot 4: Time Series of Compute Times
        ax4 = axes[1, 0]
        all_compute = []
        all_labels = []
        for m in self.all_metrics[-100:]:
            all_compute.append(m.compute_time_ms)
            all_labels.append('overlap' if m.overlap_enabled else 'no_overlap')
        
        colors = ['#3498db' if l == 'overlap' else '#2ecc71' for l in all_labels]
        ax4.scatter(range(len(all_compute)), all_compute, c=colors, alpha=0.6, s=30)
        ax4.axhline(y=np.mean(no_overlap_compute), color='#2ecc71', linestyle='--', 
                   alpha=0.8, label=f'No Overlap Mean: {np.mean(no_overlap_compute):.2f}ms')
        ax4.axhline(y=np.mean(overlap_compute), color='#3498db', linestyle='--', 
                   alpha=0.8, label=f'Overlap Mean: {np.mean(overlap_compute):.2f}ms')
        ax4.set_xlabel('Batch Index (within window)', fontsize=11)
        ax4.set_ylabel('Compute Time (ms)', fontsize=11)
        ax4.set_title('Compute Time Per Batch', fontsize=12)
        ax4.legend(loc='upper right', fontsize=9)
        
        # Add shaded regions for overlap/no-overlap
        for i in range(0, len(all_labels), self.switch_interval):
            end_i = min(i + self.switch_interval, len(all_labels))
            if i // self.switch_interval % 2 == 0:
                ax4.axvspan(i, end_i, alpha=0.1, color='green')
            else:
                ax4.axvspan(i, end_i, alpha=0.1, color='blue')
        
        # Plot 5: Timeline comparison - Compute N batches + transfer 10GB
        # Sequential: N batches compute THEN 10GB H2D
        # Parallel: N batches compute with H2D hidden inside (no extra time)
        ax5 = axes[1, 1]
        
        # Get 10GB benchmark time (sequential H2D)
        h2d_10gb_time = self.h2d_benchmark_results.get(10.0, {}).get('time_ms', 423) if hasattr(self, 'h2d_benchmark_results') else 423
        
        # Get average compute time per batch
        avg_compute_time = np.mean(overlap_compute) if overlap_compute else 80
        
        # Calculate how many batches needed for 10GB (from records)
        if self.h2d_transfer_records:
            avg_batches_for_10gb = np.mean([r['batches'] for r in self.h2d_transfer_records])
        else:
            avg_per_batch = np.mean(self.stats.h2d_transferred_per_batch_gb) if self.stats.h2d_transferred_per_batch_gb else 1.5
            avg_batches_for_10gb = self.target_h2d_gb / avg_per_batch if avg_per_batch > 0 else 7
        
        # Total compute time for N batches
        total_compute_n_batches = avg_batches_for_10gb * avg_compute_time
        
        # Draw timeline bars (Gantt-chart style)
        bar_height = 0.35
        
        # Sequential: N batches Compute + 10GB H2D (serial)
        # Must do compute first, then transfer H2D
        seq_compute = total_compute_n_batches  # N batches compute
        seq_h2d = h2d_10gb_time  # Full 10GB transfer time after compute
        seq_total = seq_compute + seq_h2d
        
        ax5.barh(y=2, width=seq_compute, left=0, height=bar_height, 
                 color='#e74c3c', label=f'Compute ({avg_batches_for_10gb:.0f} batches)', edgecolor='black')
        ax5.barh(y=2, width=seq_h2d, left=seq_compute, height=bar_height,
                 color='#f39c12', label='H2D 10GB', edgecolor='black')
        
        # Parallel: N batches Compute with H2D hidden inside
        # H2D runs during compute, so no extra time needed!
        par_compute = total_compute_n_batches  # Same N batches compute
        par_total = par_compute  # H2D is FREE (hidden in compute bubble)
        
        ax5.barh(y=1, width=par_compute, left=0, height=bar_height, 
                 color='#e74c3c', edgecolor='black')
        # Show H2D as overlay inside compute (overlapping, hidden)
        ax5.barh(y=0.65, width=par_compute, left=0, height=bar_height * 0.8,
                 color='#f39c12', edgecolor='black', alpha=0.6)
        
        # Add text "H2D hidden inside" 
        ax5.text(par_compute / 2, 0.65, 'H2D 10GB (hidden)', ha='center', va='center', 
                fontsize=9, color='black', fontweight='bold')
        
        # Add labels with details
        ax5.text(seq_total + 15, 2, f'Total: {seq_total:.0f}ms', va='center', fontsize=10, fontweight='bold')
        ax5.text(par_total + 15, 0.85, f'Total: {par_total:.0f}ms', va='center', fontsize=10, fontweight='bold')
        
        # Add mode labels
        ax5.text(-30, 2, 'Sequential', ha='right', va='center', fontsize=10, fontweight='bold')
        ax5.text(-30, 0.85, 'Parallel\n(Overlap)', ha='right', va='center', fontsize=10, fontweight='bold')
        
        ax5.set_xlabel('Time (ms)', fontsize=11)
        ax5.set_title(f'Process {avg_batches_for_10gb:.0f} Batches + Transfer 10GB H2D', fontsize=12)
        ax5.set_yticks([])
        ax5.set_xlim(-80, seq_total * 1.15)
        ax5.set_ylim(0.3, 2.5)
        ax5.legend(loc='upper right', fontsize=9)
        
        # Calculate and display savings
        saved_time = seq_total - par_total  # Should be ~423ms (the H2D time)
        speedup = seq_total / par_total if par_total > 0 else 1
        ax5.text(0.5, 0.02, f'H2D 10GB is FREE! Saved {saved_time:.0f}ms ({speedup:.2f}x faster)', 
                transform=ax5.transAxes, ha='center', va='bottom',
                fontsize=11, color='green', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        # Plot 6: H2D Transfer Progress & Statistics
        ax6 = axes[1, 2]
        
        # Show per-batch H2D transferred and cumulative progress
        if self.stats.h2d_transferred_per_batch_gb:
            h2d_per_batch = self.stats.h2d_transferred_per_batch_gb[-100:]
            
            # Calculate cumulative sum for progress visualization
            cumulative = np.cumsum(h2d_per_batch)
            
            # Create bar chart for per-batch transfer
            x = np.arange(len(h2d_per_batch))
            ax6.bar(x, h2d_per_batch, color='#27ae60', alpha=0.8, label='Per Batch (GB)')
            
            # Add cumulative line on secondary axis
            ax6_twin = ax6.twinx()
            ax6_twin.plot(x, cumulative, color='#e74c3c', linewidth=2, marker='', label='Cumulative (GB)')
            ax6_twin.axhline(y=self.target_h2d_gb, color='#e74c3c', linestyle='--', alpha=0.5, label=f'Target: {self.target_h2d_gb}GB')
            ax6_twin.set_ylabel('Cumulative (GB)', fontsize=11, color='#e74c3c')
            ax6_twin.tick_params(axis='y', labelcolor='#e74c3c')
            
            ax6.set_xlabel('Batch Index', fontsize=11)
            ax6.set_ylabel('Per Batch (GB)', fontsize=11, color='#27ae60')
            ax6.tick_params(axis='y', labelcolor='#27ae60')
            ax6.set_title('H2D Transfer Progress (Hidden in Compute)', fontsize=12)
            
            # Calculate statistics
            avg_per_batch = np.mean(h2d_per_batch)
            total_transferred = np.sum(h2d_per_batch)
            batches_for_10gb = int(np.ceil(self.target_h2d_gb / avg_per_batch)) if avg_per_batch > 0 else 0
            
            # Show 10GB completion records if any
            if self.h2d_transfer_records:
                last_record = self.h2d_transfer_records[-1]
                stats_text = (f'Avg/Batch: {avg_per_batch:.2f}GB\n'
                             f'10GB needs: ~{batches_for_10gb} batches\n'
                             f'Last 10GB: {last_record["batches"]} batches\n'
                             f'Compute: {last_record["total_compute_ms"]:.0f}ms')
            else:
                stats_text = (f'Avg/Batch: {avg_per_batch:.2f}GB\n'
                             f'10GB needs: ~{batches_for_10gb} batches\n'
                             f'Progress: {self.cumulative_h2d_gb:.2f}/{self.target_h2d_gb}GB')
            
            ax6.text(0.02, 0.98, stats_text, transform=ax6.transAxes, ha='left', va='top',
                    fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            # Show fewer x-ticks for readability
            if len(x) > 20:
                tick_indices = np.linspace(0, len(x)-1, 10, dtype=int)
                ax6.set_xticks(tick_indices)
        else:
            ax6.text(0.5, 0.5, 'No H2D data recorded', ha='center', va='center', transform=ax6.transAxes)
            ax6.set_title('H2D Transfer Progress', fontsize=12)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.output_dir, f'overlap_experiment_batch_{self.batch_count}.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"[OverlapExperiment] Saved plot to {plot_path}")
        
        # Log summary statistics
        logger.info(f"[OverlapExperiment] === Summary (Batches {self.batch_count - 100} - {self.batch_count}) ===")
        logger.info(f"[OverlapExperiment] No Overlap - Compute: {np.mean(no_overlap_compute):.2f}±{np.std(no_overlap_compute):.2f}ms")
        logger.info(f"[OverlapExperiment] Overlap    - Compute: {np.mean(overlap_compute):.2f}±{np.std(overlap_compute):.2f}ms")
        
        # Log H2D transfer stats
        if self.stats.h2d_transferred_per_batch_gb:
            avg_per_batch = np.mean(self.stats.h2d_transferred_per_batch_gb)
            batches_for_10gb = int(np.ceil(self.target_h2d_gb / avg_per_batch)) if avg_per_batch > 0 else 0
            logger.info(f"[OverlapExperiment] H2D Per Batch: {avg_per_batch:.2f}GB (10GB needs ~{batches_for_10gb} batches)")
            logger.info(f"[OverlapExperiment] Current Progress: {self.cumulative_h2d_gb:.2f}/{self.target_h2d_gb}GB")
        
        # Log 10GB completion records
        if self.h2d_transfer_records:
            last = self.h2d_transfer_records[-1]
            logger.info(f"[OverlapExperiment] Last 10GB completion: {last['batches']} batches, "
                       f"{last['total_compute_ms']:.0f}ms compute time")
        
        # Clear stats for next window
        self.stats.clear()
    
    def cleanup(self):
        """Clean up resources."""
        self._dummy_src = None
        self._dummy_dst = None
        torch.cuda.empty_cache()

