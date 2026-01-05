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

# pyre-strict
"""
model_family for dlrm_v3.
"""

import copy
import functools
import logging
import os
import time
import uuid
from threading import Event
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.multiprocessing as mp
import torchrec
from generative_recommenders.dlrm_v3.checkpoint import (
    load_nonsparse_checkpoint,
    load_sparse_checkpoint,
)
from generative_recommenders.dlrm_v3.configs import HASH_SIZE_1B
from generative_recommenders.dlrm_v3.datasets.dataset import Samples
from generative_recommenders.dlrm_v3.inference.inference_modules import (
    get_hstu_model,
    HSTUSparseInferenceModule,
    move_sparse_output_to_device,
    set_is_inference,
)
from generative_recommenders.dlrm_v3.utils import Profiler
from generative_recommenders.modules.dlrm_hstu import DlrmHSTUConfig, SequenceEmbedding
from pyre_extensions import none_throws
from torch import quantization as quant
from torchrec.distributed.quant_embedding import QuantEmbeddingCollection
from torchrec.modules.embedding_configs import EmbeddingConfig, QuantConfig
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor
from torchrec.sparse.tensor_dict import maybe_td_to_kjt
from torchrec.test_utils import get_free_port

logger: logging.Logger = logging.getLogger(__name__)


class HSTUModelFamily:
    """
    High-level interface for the HSTU model family.

    Manages both sparse (embedding) and dense (transformer) components of the
    HSTU model, supporting distributed inference across multiple GPUs.

    Args:
        hstu_config: Configuration object for the HSTU model.
        table_config: Dictionary of embedding table configurations.
        output_trace: Whether to enable profiling trace output.
        sparse_quant: Whether to quantize sparse embeddings.
        compute_eval: Whether to compute evaluation metrics (includes labels).
        enable_overlap_experiment: Whether to enable H2D-compute overlap experiment.
    """

    def __init__(
        self,
        hstu_config: DlrmHSTUConfig,
        table_config: Dict[str, EmbeddingConfig],
        output_trace: bool = False,
        sparse_quant: bool = False,
        compute_eval: bool = False,
        enable_overlap_experiment: bool = False,
    ) -> None:
        self.hstu_config = hstu_config
        self.table_config = table_config
        self.sparse: ModelFamilySparseDist = ModelFamilySparseDist(
            hstu_config=hstu_config,
            table_config=table_config,
            quant=sparse_quant,
        )

        assert torch.cuda.is_available(), "CUDA is required for this benchmark."
        ngpus = torch.cuda.device_count()
        self.world_size = int(os.environ.get("WORLD_SIZE", str(ngpus)))
        logger.warning(f"Using {self.world_size} GPU(s)...")
        dense_model_family_clazz = (
            ModelFamilyDenseDist
            if self.world_size > 1
            else ModelFamilyDenseSingleWorker
        )
        
        # Prepare kwargs for dense model
        dense_kwargs = {
            "hstu_config": hstu_config,
            "table_config": table_config,
            "output_trace": output_trace,
            "compute_eval": compute_eval,
        }
        # Only pass overlap experiment flag to single worker (not distributed)
        if self.world_size == 1 and enable_overlap_experiment:
            dense_kwargs["enable_overlap_experiment"] = enable_overlap_experiment
            
        self.dense: Union[ModelFamilyDenseDist, ModelFamilyDenseSingleWorker] = (
            dense_model_family_clazz(**dense_kwargs)
        )

    def version(self) -> str:
        """Return the PyTorch version string."""
        return torch.__version__

    def name(self) -> str:
        """Return the model family name identifier."""
        return "model-family-hstu"

    def load(self, model_path: str) -> None:
        """
        Load model checkpoints from disk.

        Args:
            model_path: Base path to the model checkpoint directory.
        """
        self.sparse.load(model_path=model_path)
        self.dense.load(model_path=model_path)

    def predict(
        self, samples: Optional[Samples]
    ) -> Optional[
        Tuple[
            torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], float, float, float, float, int, int, float, float
        ]
    ]:
        """
        Run inference on a batch of samples.

        Processes samples through sparse embeddings, then dense forward pass.

        Args:
            samples: Input samples containing features. If None, signals shutdown.

        Returns:
            Tuple of (predictions, labels, weights, sparse_time, dense_time) or None.
        """
        with torch.no_grad():
            if samples is None:
                self.dense.predict(None, None, 0, None, 0, None)
                return None
            # Debug: log input samples info
            logger.debug(f"[ModelFamily.predict] Starting prediction")
            logger.debug(f"[ModelFamily.predict] uih_features_kjt keys: {samples.uih_features_kjt.keys()}")
            logger.debug(f"[ModelFamily.predict] uih_features_kjt lengths: {samples.uih_features_kjt.lengths()}")
            logger.debug(f"[ModelFamily.predict] candidates_features_kjt keys: {samples.candidates_features_kjt.keys()}")
            logger.debug(f"[ModelFamily.predict] candidates_features_kjt lengths: {samples.candidates_features_kjt.lengths()}")
            
            logger.debug(f"[ModelFamily.predict] Running sparse.predict...")
            (
                seq_embeddings,
                payload_features,
                max_uih_len,
                uih_seq_lengths,
                max_num_candidates,
                num_candidates,
                dt_sparse,
            ) = self.sparse.predict(samples)
            
            # Debug: log sparse output info
            logger.debug(f"[ModelFamily.predict] Sparse done. max_uih_len={max_uih_len}, max_num_candidates={max_num_candidates}")
            logger.debug(f"[ModelFamily.predict] uih_seq_lengths shape: {uih_seq_lengths.shape}, values: {uih_seq_lengths}")
            logger.debug(f"[ModelFamily.predict] num_candidates shape: {num_candidates.shape}, values: {num_candidates}")
            for k, v in seq_embeddings.items():
                logger.debug(f"[ModelFamily.predict] seq_embeddings[{k}]: embedding shape={v.embedding.shape}, lengths shape={v.lengths.shape}")
            for k, v in payload_features.items():
                logger.debug(f"[ModelFamily.predict] payload_features[{k}]: shape={v.shape}, dtype={v.dtype}")
            
            logger.debug(f"[ModelFamily.predict] Running dense.predict...")
            out = self.dense.predict(
                seq_embeddings,
                payload_features,
                max_uih_len,
                uih_seq_lengths,
                max_num_candidates,
                num_candidates,
            )
            if len(out) == 10:
                (  # pyre-ignore [23]
                    mt_target_preds,
                    mt_target_labels,
                    mt_target_weights,
                    dt_dense,
                    dt_h2d,
                    dt_d2h,
                    h2d_size,
                    d2h_size,
                    dt_compute,
                    dt_bubble,
                ) = out
            elif len(out) == 6:
                # Backward compatibility
                (  # pyre-ignore [23]
                    mt_target_preds,
                    mt_target_labels,
                    mt_target_weights,
                    dt_dense,
                    dt_h2d,
                    dt_d2h,
                ) = out
                h2d_size = 0
                d2h_size = 0
                dt_compute = 0.0
                dt_bubble = 0.0
            else:
                # Backward compatibility
                (  # pyre-ignore [23]
                    mt_target_preds,
                    mt_target_labels,
                    mt_target_weights,
                    dt_dense,
                ) = out
                dt_h2d = 0.0
                dt_d2h = 0.0
                h2d_size = 0
                d2h_size = 0
                dt_compute = 0.0
                dt_bubble = 0.0
            return (
                mt_target_preds,
                mt_target_labels,
                mt_target_weights,
                dt_sparse,
                dt_dense,
                dt_h2d,
                dt_d2h,
                h2d_size,
                d2h_size,
                dt_compute,
                dt_bubble,
            )


def ec_patched_forward_wo_embedding_copy(
    ec_module: torchrec.EmbeddingCollection,
    features: KeyedJaggedTensor,  # can also take TensorDict as input
) -> Dict[str, JaggedTensor]:
    """
    Run the EmbeddingBagCollection forward pass. This method takes in a `KeyedJaggedTensor`
    and returns a `Dict[str, JaggedTensor]`, which is the result of the individual embeddings for each feature.

    Args:
        features (KeyedJaggedTensor): KJT of form [F X B X L].

    Returns:
        Dict[str, JaggedTensor]
    """
    features = maybe_td_to_kjt(features, None)
    feature_embeddings: Dict[str, JaggedTensor] = {}
    jt_dict: Dict[str, JaggedTensor] = features.to_dict()
    for i, emb_module in enumerate(ec_module.embeddings.values()):
        feature_names = ec_module._feature_names[i]
        embedding_names = ec_module._embedding_names_by_table[i]
        for j, embedding_name in enumerate(embedding_names):
            feature_name = feature_names[j]
            f = jt_dict[feature_name]
            indices = torch.clamp(f.values(), min=0, max=HASH_SIZE_1B - 1)
            lookup = emb_module(
                input=indices
            )  # remove the dtype cast at https://github.com/meta-pytorch/torchrec/blob/0a2cebd5472a7edc5072b3c912ad8aaa4179b9d9/torchrec/modules/embedding_modules.py#L486
            feature_embeddings[embedding_name] = JaggedTensor(
                values=lookup,
                lengths=f.lengths(),
                weights=f.values() if ec_module._need_indices else None,
            )
    return feature_embeddings


class ModelFamilySparseDist:
    """
    Sparse Arch module manager.

    Handles loading and inference of sparse embedding lookups, optionally
    with quantization for memory efficiency.

    Args:
        hstu_config: HSTU model configuration.
        table_config: Embedding table configurations.
        quant: Whether to apply dynamic quantization to embeddings.
    """

    def __init__(
        self,
        hstu_config: DlrmHSTUConfig,
        table_config: Dict[str, EmbeddingConfig],
        quant: bool = False,
    ) -> None:
        super(ModelFamilySparseDist, self).__init__()
        self.hstu_config = hstu_config
        self.table_config = table_config
        self.module: Optional[torch.nn.Module] = None
        self.quant: bool = quant

    def load(self, model_path: str) -> None:
        """
        Load sparse model checkpoint and optionally apply quantization.

        Args:
            model_path: Path to the model checkpoint directory.
        """
        logger.warning(f"Loading sparse module from {model_path}")

        sparse_arch: HSTUSparseInferenceModule = HSTUSparseInferenceModule(
            table_config=self.table_config,
            hstu_config=self.hstu_config,
        )
        load_sparse_checkpoint(model=sparse_arch._hstu_model, path=model_path)
        sparse_arch.eval()
        if self.quant:
            self.module = quant.quantize_dynamic(
                sparse_arch,
                qconfig_spec={
                    torchrec.EmbeddingCollection: QuantConfig(
                        activation=quant.PlaceholderObserver.with_args(
                            dtype=torch.float
                        ),
                        weight=quant.PlaceholderObserver.with_args(dtype=torch.int8),
                    ),
                },
                mapping={
                    torchrec.EmbeddingCollection: QuantEmbeddingCollection,
                },
                inplace=False,
            )
        else:
            sparse_arch._hstu_model._embedding_collection.forward = (  # pyre-ignore[8]
                functools.partial(
                    ec_patched_forward_wo_embedding_copy,
                    sparse_arch._hstu_model._embedding_collection,
                )
            )
            self.module = sparse_arch
        logger.warning(f"sparse module is {self.module}")

    def predict(
        self, samples: Samples
    ) -> Tuple[
        Dict[str, SequenceEmbedding],
        Dict[str, torch.Tensor],
        int,
        torch.Tensor,
        int,
        torch.Tensor,
        float,
    ]:
        """
        Run sparse forward pass (embedding lookups).

        Args:
            samples: Input samples with feature tensors.

        Returns:
            Tuple of (seq_embeddings, payload_features, max_uih_len, uih_seq_lengths,
            max_num_candidates, num_candidates, elapsed_time).
        """
        with torch.profiler.record_function("sparse forward"):
            module: torch.nn.Module = none_throws(self.module)
            assert self.module is not None
            uih_features = samples.uih_features_kjt
            candidates_features = samples.candidates_features_kjt
            t0: float = time.time()
            (
                seq_embeddings,
                payload_features,
                max_uih_len,
                uih_seq_lengths,
                max_num_candidates,
                num_candidates,
            ) = module(
                uih_features=uih_features,
                candidates_features=candidates_features,
            )
            dt_sparse: float = time.time() - t0
            return (
                seq_embeddings,
                payload_features,
                max_uih_len,
                uih_seq_lengths,
                max_num_candidates,
                num_candidates,
                dt_sparse,
            )


class ModelFamilyDenseDist:
    """
    Distributed dense module manager for multi-GPU inference.

    Spawns worker processes for each GPU to run dense forward passes in parallel,
    with samples distributed via inter-process queues.

    Args:
        hstu_config: HSTU model configuration.
        table_config: Embedding table configurations.
        output_trace: Whether to enable profiling traces.
        compute_eval: Whether to compute evaluation metrics.
    """

    def __init__(
        self,
        hstu_config: DlrmHSTUConfig,
        table_config: Dict[str, EmbeddingConfig],
        output_trace: bool = False,
        compute_eval: bool = False,
    ) -> None:
        super(ModelFamilyDenseDist, self).__init__()
        self.hstu_config = hstu_config
        self.table_config = table_config
        self.output_trace = output_trace
        self.compute_eval = compute_eval

        ngpus = torch.cuda.device_count()
        self.world_size = int(os.environ.get("WORLD_SIZE", str(ngpus)))
        self.rank = 0
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(get_free_port())
        self.dist_backend = "nccl"

        ctx = mp.get_context("spawn")
        self.samples_q: List[mp.Queue] = [ctx.Queue() for _ in range(self.world_size)]
        self.result_q: List[mp.Queue] = [ctx.Queue() for _ in range(self.world_size)]

    def load(self, model_path: str) -> None:
        """
        Load dense model and spawn worker processes for distributed inference.

        Args:
            model_path: Path to the model checkpoint directory.
        """
        logger.warning(f"Loading dense module from {model_path}")

        ctx = mp.get_context("spawn")
        processes = []
        for rank in range(self.world_size):
            p = ctx.Process(
                target=self.distributed_setup,
                args=(
                    rank,
                    self.world_size,
                    model_path,
                ),
            )
            p.start()
            processes.append(p)

    def distributed_setup(self, rank: int, world_size: int, model_path: str) -> None:
        """
        Initialize and run a dense worker process.

        Each worker loads the model, processes samples from its queue, and
        returns results.

        Args:
            rank: Process rank (GPU index).
            world_size: Total number of worker processes.
            model_path: Path to model checkpoint.
        """
        nprocs_per_rank = 16
        start_core: int = nprocs_per_rank * rank
        cores: set[int] = set([start_core + i for i in range(nprocs_per_rank)])
        os.sched_setaffinity(0, cores)
        set_is_inference(is_inference=not self.compute_eval)
        model = get_hstu_model(
            table_config=self.table_config,
            hstu_config=self.hstu_config,
            table_device="cpu",
            max_hash_size=100,
            is_dense=True,
        ).to(torch.bfloat16)
        model.set_training_dtype(torch.bfloat16)
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(f"cuda:{rank}")
        load_nonsparse_checkpoint(
            model=model, device=device, optimizer=None, path=model_path
        )
        model = model.to(device)
        model.eval()
        profiler = Profiler(rank) if self.output_trace else None
        
        # Create separate CUDA streams for compute and transfer to enable pipeline overlap
        # compute_stream: for model forward pass (default stream)
        # transfer_stream: for H2D/D2H transfers to overlap with compute
        transfer_stream = torch.cuda.Stream(device=device)
        next_item = None
        next_item_ready = False

        with torch.no_grad():
            while True:
                # Pipeline: while computing current batch, prepare next batch transfer
                current_item = next_item if next_item_ready else self.samples_q[rank].get()
                # If -1 is received terminate all subprocesses
                if current_item == -1:
                    break
                
                # Start async transfer of next batch while computing current batch
                if not next_item_ready:
                    # Try to get next item for pipeline overlap
                    try:
                        next_item = self.samples_q[rank].get_nowait()
                        if next_item != -1:
                            next_item_ready = True
                            # Start async H2D transfer for next batch on transfer stream
                            with torch.cuda.stream(transfer_stream):
                                with torch.profiler.record_function("pipeline H2D transfer (next batch)"):
                                    (
                                        next_id,
                                        next_seq_embeddings,
                                        next_payload_features,
                                        next_max_uih_len,
                                        next_uih_seq_lengths,
                                        next_max_num_candidates,
                                        next_num_candidates,
                                    ) = next_item
                                    # Transfer next batch data asynchronously (non-blocking)
                                    next_num_candidates = next_num_candidates.to(device, non_blocking=True)
                                    next_uih_seq_lengths = next_uih_seq_lengths.to(device, non_blocking=True)
                                    next_seq_embeddings = {
                                        k: SequenceEmbedding(
                                            lengths=next_seq_embeddings[k].lengths.to(device, non_blocking=True),
                                            embedding=next_seq_embeddings[k].embedding.to(device, non_blocking=True).to(torch.bfloat16),
                                        )
                                        for k in next_seq_embeddings.keys()
                                    }
                                    for k, v in next_payload_features.items():
                                        next_payload_features[k] = v.to(device, non_blocking=True)
                    except:
                        # No next item available yet
                        next_item_ready = False
                        next_item = None
                
                if self.output_trace:
                    assert profiler is not None
                    profiler.step()
                
                with torch.profiler.record_function("get_item_from_queue"):
                    # Copy here to release data in the producer to avoid invalid cuda caching allocator release.
                    item = copy.deepcopy(current_item)
                    (
                        id,
                        seq_embeddings,
                        payload_features,
                        max_uih_len,
                        uih_seq_lengths,
                        max_num_candidates,
                        num_candidates,
                    ) = item
                    assert seq_embeddings is not None
                
                # Wait for transfer stream to complete if we're using pre-transferred data
                if next_item_ready and current_item == next_item:
                    transfer_stream.synchronize()
                
                # Calculate D2H transfer size and perform compute + async D2H
                t0_compute = time.time()
                with torch.profiler.record_function("dense forward"):
                    (
                        _,
                        _,
                        _,
                        mt_target_preds,
                        mt_target_labels,
                        mt_target_weights,
                    ) = model.main_forward(
                        seq_embeddings=seq_embeddings,
                        payload_features=payload_features,
                        max_uih_len=max_uih_len,
                        uih_seq_lengths=uih_seq_lengths,
                        max_num_candidates=max_num_candidates,
                        num_candidates=num_candidates,
                    )
                    assert mt_target_preds is not None
                    
                    # Calculate D2H transfer size
                    d2h_size = mt_target_preds.numel() * mt_target_preds.element_size()
                    if mt_target_labels is not None:
                        d2h_size += mt_target_labels.numel() * mt_target_labels.element_size()
                    if mt_target_weights is not None:
                        d2h_size += mt_target_weights.numel() * mt_target_weights.element_size()
                    
                    # Use async D2H transfer on transfer stream for pipeline overlap
                    # This can overlap with next batch's H2D on the same stream
                    t0_d2h = time.time()
                    with torch.profiler.record_function("D2H transfer"):
                        with torch.cuda.stream(transfer_stream):
                            mt_target_preds = mt_target_preds.detach().to(device="cpu", non_blocking=True)
                            if mt_target_labels is not None:
                                mt_target_labels = mt_target_labels.detach().to(device="cpu", non_blocking=True)
                            if mt_target_weights is not None:
                                mt_target_weights = mt_target_weights.detach().to(device="cpu", non_blocking=True)
                
                # Synchronize compute stream to ensure forward pass is complete
                torch.cuda.current_stream(device).synchronize()
                dt_compute = time.time() - t0_compute
                
                # Synchronize transfer stream to ensure D2H is complete
                transfer_stream.synchronize()
                dt_d2h = time.time() - t0_d2h
                
                # Calculate bubble: time when GPU compute is idle
                # Bubble = compute_time - overlapped_transfer_time
                # If D2H overlaps with compute, bubble is reduced
                dt_bubble = max(0.0, dt_compute - dt_d2h) if dt_d2h < dt_compute else 0.0
                
                self.result_q[rank].put(
                    (id, mt_target_preds, mt_target_labels, mt_target_weights, dt_d2h, d2h_size, dt_compute, dt_bubble)
                )
                
                # Update for next iteration
                if next_item_ready:
                    current_item = next_item
                    next_item_ready = False

    def capture_output(
        self, id: uuid.UUID, rank: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], float, int, float, float]:
        """
        Retrieve inference results from a worker process.

        Args:
            id: Unique identifier for the request.
            rank: Worker rank to retrieve from.

        Returns:
            Tuple of (predictions, labels, weights, D2H transfer time, D2H size, compute time, bubble time).
        """
        while True:
            result = self.result_q[rank].get()
            if len(result) == 8:
                recv_id, preds, labels, weights, dt_d2h, d2h_size, dt_compute, dt_bubble = result
            elif len(result) == 5:
                recv_id, preds, labels, weights, dt_d2h = result
                d2h_size = 0
                dt_compute = 0.0
                dt_bubble = 0.0
            else:
                # Backward compatibility
                recv_id, preds, labels, weights = result
                dt_d2h = 0.0
                d2h_size = 0
                dt_compute = 0.0
                dt_bubble = 0.0
            assert recv_id == id
            return preds, labels, weights, dt_d2h, d2h_size, dt_compute, dt_bubble

    def get_rank(self) -> int:
        """
        Get the next worker rank for load balancing.

        Returns:
            Rank index, cycling through available workers.
        """
        rank = self.rank
        self.rank = (self.rank + 1) % self.world_size
        return rank

    def predict(
        self,
        seq_embeddings: Optional[Dict[str, SequenceEmbedding]],
        payload_features: Optional[Dict[str, torch.Tensor]],
        max_uih_len: int,
        uih_seq_lengths: Optional[torch.Tensor],
        max_num_candidates: int,
        num_candidates: Optional[torch.Tensor],
    ) -> Optional[
        Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], float, float, float, int, int, float, float]
    ]:
        """
        Run distributed dense forward pass.

        Dispatches work to a worker process and collects results.

        Args:
            seq_embeddings: Sequence embeddings from sparse module.
            payload_features: Additional feature tensors.
            max_uih_len: Maximum UIH sequence length.
            uih_seq_lengths: Per-sample UIH lengths.
            max_num_candidates: Maximum candidates per sample.
            num_candidates: Per-sample candidate counts.

        Returns:
            Tuple of (predictions, labels, weights, dense_time, h2d_time, d2h_time, h2d_size, d2h_size, compute_time, bubble_time) or None if shutdown.
        """
        id = uuid.uuid4()
        # If none is received terminate all subprocesses
        if seq_embeddings is None:
            for rank in range(self.world_size):
                self.samples_q[rank].put(-1)
            return None
        rank = self.get_rank()
        device = torch.device(f"cuda:{rank}")
        assert (
            payload_features is not None
            and num_candidates is not None
            and uih_seq_lengths is not None
        )
        t0: float = time.time()
        # Use async H2D transfer with CUDA stream for pipeline overlap
        stream = torch.cuda.Stream(device=device)
        seq_embeddings, payload_features, uih_seq_lengths, num_candidates, dt_h2d, h2d_size = (
            move_sparse_output_to_device(
                seq_embeddings=seq_embeddings,
                payload_features=payload_features,
                uih_seq_lengths=uih_seq_lengths,
                num_candidates=num_candidates,
                device=device,
                stream=stream,
            )
        )
        # Wait for H2D to complete before putting in queue
        stream.synchronize()
        self.samples_q[rank].put(
            (
                id,
                seq_embeddings,
                payload_features,
                max_uih_len,
                uih_seq_lengths,
                max_num_candidates,
                num_candidates,
            )
        )
        (mt_target_preds, mt_target_labels, mt_target_weights, dt_d2h, d2h_size, dt_compute, dt_bubble) = self.capture_output(
            id, rank
        )
        dt_dense = time.time() - t0
        return (
            mt_target_preds,
            mt_target_labels,
            mt_target_weights,
            dt_dense,
            dt_h2d,
            dt_d2h,
            h2d_size,
            d2h_size,
            dt_compute,
            dt_bubble,
        )


class ModelFamilyDenseSingleWorker:
    """
    Single-worker dense module manager for single-GPU inference.

    Simpler alternative to ModelFamilyDenseDist for single-GPU setups.

    Args:
        hstu_config: HSTU model configuration.
        table_config: Embedding table configurations.
        output_trace: Whether to enable profiling traces.
        compute_eval: Whether to compute evaluation metrics.
        enable_overlap_experiment: Whether to enable H2D-compute overlap experiment.
    """

    def __init__(
        self,
        hstu_config: DlrmHSTUConfig,
        table_config: Dict[str, EmbeddingConfig],
        output_trace: bool = False,
        compute_eval: bool = False,
        enable_overlap_experiment: bool = False,
    ) -> None:
        self.model: Optional[torch.nn.Module] = None
        self.hstu_config = hstu_config
        self.table_config = table_config
        self.output_trace = output_trace
        self.device: torch.device = torch.device("cuda:0")
        torch.cuda.set_device(self.device)
        self.profiler: Optional[Profiler] = (
            Profiler(rank=0) if self.output_trace else None
        )
        
        # Overlap experiment
        self.enable_overlap_experiment = enable_overlap_experiment
        self.overlap_experiment = None
        if enable_overlap_experiment:
            from generative_recommenders.dlrm_v3.inference.overlap_experiment import OverlapExperiment
            # Use absolute path to ensure files are saved correctly
            import os
            output_dir = os.path.join(os.path.dirname(__file__), "logs-bubble")
            self.overlap_experiment = OverlapExperiment(
                device=self.device,
                output_dir=output_dir,
                chunk_size_mb=128.0,  # Larger chunks = fewer syncs = less overhead
                switch_interval=50,
            )

    def load(self, model_path: str) -> None:
        """
        Load dense model for single-GPU inference.

        Args:
            model_path: Path to the model checkpoint directory.
        """
        logger.warning(f"Loading dense module from {model_path}")
        self.model = (
            get_hstu_model(
                table_config=self.table_config,
                hstu_config=self.hstu_config,
                table_device="cpu",
                is_dense=True,
            )
            .to(self.device)
            .to(torch.bfloat16)
        )
        self.model.set_training_dtype(torch.bfloat16)
        load_nonsparse_checkpoint(
            model=self.model, device=self.device, optimizer=None, path=model_path
        )
        assert self.model is not None
        self.model.eval()

    def predict(
        self,
        seq_embeddings: Optional[Dict[str, SequenceEmbedding]],
        payload_features: Optional[Dict[str, torch.Tensor]],
        max_uih_len: int,
        uih_seq_lengths: Optional[torch.Tensor],
        max_num_candidates: int,
        num_candidates: Optional[torch.Tensor],
    ) -> Optional[
        Tuple[
            torch.Tensor,
            Optional[torch.Tensor],
            Optional[torch.Tensor],
            float,
            float,
            float,
            int,
            int,
            float,
            float,
        ]
    ]:
        """
        Run dense forward pass on single GPU.

        Args:
            seq_embeddings: Sequence embeddings from sparse module.
            payload_features: Additional feature tensors.
            max_uih_len: Maximum UIH sequence length.
            uih_seq_lengths: Per-sample UIH lengths.
            max_num_candidates: Maximum candidates per sample.
            num_candidates: Per-sample candidate counts.

        Returns:
            Tuple of (predictions, labels, weights, dense_time, h2d_time, d2h_time, h2d_size, d2h_size, compute_time, bubble_time).
        """
        if self.output_trace:
            assert self.profiler is not None
            self.profiler.step()
        assert (
            payload_features is not None
            and uih_seq_lengths is not None
            and num_candidates is not None
            and seq_embeddings is not None
        )
        t0: float = time.time()
        with torch.profiler.record_function("dense forward"):
            # Debug: log before H2D transfer
            logger.debug(f"[DenseSingleWorker.predict] Before H2D transfer")
            logger.debug(f"[DenseSingleWorker.predict] uih_seq_lengths: {uih_seq_lengths}, device: {uih_seq_lengths.device}")
            logger.debug(f"[DenseSingleWorker.predict] num_candidates: {num_candidates}, device: {num_candidates.device}")
            for k, v in seq_embeddings.items():
                logger.debug(f"[DenseSingleWorker.predict] seq_embeddings[{k}] before H2D: embedding shape={v.embedding.shape}, device={v.embedding.device}")
            
            # Use async H2D transfer with CUDA stream for pipeline overlap
            h2d_stream = torch.cuda.Stream(device=self.device)
            seq_embeddings, payload_features, uih_seq_lengths, num_candidates, dt_h2d, h2d_size = (
                move_sparse_output_to_device(
                    seq_embeddings=seq_embeddings,
                    payload_features=payload_features,
                    uih_seq_lengths=uih_seq_lengths,
                    num_candidates=num_candidates,
                    device=self.device,
                    stream=h2d_stream,
                )
            )
            # Wait for H2D to complete before forward pass
            h2d_stream.synchronize()
            
            # Debug: log after H2D transfer
            logger.debug(f"[DenseSingleWorker.predict] After H2D transfer")
            logger.debug(f"[DenseSingleWorker.predict] uih_seq_lengths: {uih_seq_lengths}, device: {uih_seq_lengths.device}")
            logger.debug(f"[DenseSingleWorker.predict] num_candidates: {num_candidates}, device: {num_candidates.device}")
            for k, v in seq_embeddings.items():
                logger.debug(f"[DenseSingleWorker.predict] seq_embeddings[{k}] after H2D: embedding shape={v.embedding.shape}, device={v.embedding.device}")
            for k, v in payload_features.items():
                logger.debug(f"[DenseSingleWorker.predict] payload_features[{k}] after H2D: shape={v.shape}, device={v.device}")
            
            t0_compute = time.time()
            assert self.model is not None
            
            # Debug: log before model forward
            logger.debug(f"[DenseSingleWorker.predict] Before model.main_forward")
            logger.debug(f"[DenseSingleWorker.predict] max_uih_len={max_uih_len}, max_num_candidates={max_num_candidates}")
            
            # Overlap experiment: run compute with optional H2D overlap simulation
            h2d_overlapped_time = 0.0
            bubble_utilization = 0.0
            
            if self.overlap_experiment is not None:
                overlap_enabled = self.overlap_experiment.should_overlap()
                
                # Define compute function
                def compute_fn():
                    return self.model.main_forward(
                        seq_embeddings=seq_embeddings,
                        payload_features=payload_features,
                        max_uih_len=max_uih_len,
                        uih_seq_lengths=uih_seq_lengths,
                        max_num_candidates=max_num_candidates,
                        num_candidates=num_candidates,
                    )
                
                # Simulate H2D data size - use 10GB to test overlap capacity
                # 10GB = 10 * 1024 * 1024 * 1024 bytes
                # With ~12 GB/s bandwidth and ~155ms compute, only ~18% can be transferred
                simulated_h2d_size = 10 * 1024 * 1024 * 1024  # 10GB
                
                if overlap_enabled:
                    # Run with overlap
                    result, compute_time_ms, sim_h2d_time_ms, h2d_overlapped_time, bubble_utilization, chunks_completed, total_chunks = (
                        self.overlap_experiment.run_overlapped_h2d_with_compute(
                            compute_fn=compute_fn,
                            compute_args={},
                            h2d_data_size=simulated_h2d_size,
                        )
                    )
                    dt_compute = compute_time_ms / 1000.0
                else:
                    # Run without overlap (baseline)
                    result, compute_time_ms, sim_h2d_time_ms = (
                        self.overlap_experiment.run_sequential_h2d_with_compute(
                            compute_fn=compute_fn,
                            compute_args={},
                            h2d_data_size=simulated_h2d_size,
                        )
                    )
                    dt_compute = compute_time_ms / 1000.0
                    chunks_completed = 0
                    total_chunks = 0
                
                _, _, _, mt_target_preds, mt_target_labels, mt_target_weights = result
                
                # Record metrics
                total_time_ms = (time.time() - t0) * 1000
                self.overlap_experiment.record_metrics(
                    overlap_enabled=overlap_enabled,
                    compute_time_ms=compute_time_ms,
                    h2d_time_ms=sim_h2d_time_ms,
                    h2d_overlapped_time_ms=h2d_overlapped_time,
                    bubble_utilization=bubble_utilization,
                    total_time_ms=total_time_ms,
                    h2d_data_size_bytes=simulated_h2d_size,
                    chunks_completed=chunks_completed,
                    total_chunks=total_chunks,
                )
                
                # Generate plots if needed
                if self.overlap_experiment.should_plot():
                    self.overlap_experiment.generate_plots()
            else:
                # Normal execution without overlap experiment
                (
                    _,
                    _,
                    _,
                    mt_target_preds,
                    mt_target_labels,
                    mt_target_weights,
                ) = self.model.main_forward(  # pyre-ignore [29]
                    seq_embeddings=seq_embeddings,
                    payload_features=payload_features,
                    max_uih_len=max_uih_len,
                    uih_seq_lengths=uih_seq_lengths,
                    max_num_candidates=max_num_candidates,
                    num_candidates=num_candidates,
                )
                dt_compute = time.time() - t0_compute
            
            # Debug: log after model forward
            logger.debug(f"[DenseSingleWorker.predict] After model.main_forward")
            logger.debug(f"[DenseSingleWorker.predict] mt_target_preds shape: {mt_target_preds.shape if mt_target_preds is not None else None}")
            
            assert mt_target_preds is not None
            
            # Calculate D2H transfer size
            d2h_size = mt_target_preds.numel() * mt_target_preds.element_size()
            if mt_target_labels is not None:
                d2h_size += mt_target_labels.numel() * mt_target_labels.element_size()
            if mt_target_weights is not None:
                d2h_size += mt_target_weights.numel() * mt_target_weights.element_size()
            
            # Use async D2H transfer with CUDA stream for pipeline overlap
            t0_d2h = time.time()
            with torch.profiler.record_function("D2H transfer"):
                d2h_stream = torch.cuda.Stream(device=self.device)
                with torch.cuda.stream(d2h_stream):
                    mt_target_preds = mt_target_preds.detach().to(device="cpu", non_blocking=True)
                    if mt_target_labels is not None:
                        mt_target_labels = mt_target_labels.detach().to(device="cpu", non_blocking=True)
                    if mt_target_weights is not None:
                        mt_target_weights = mt_target_weights.detach().to(device="cpu", non_blocking=True)
                d2h_stream.synchronize()
            dt_d2h = time.time() - t0_d2h
            
            # Calculate bubble: time when GPU compute is idle
            dt_bubble = max(0.0, dt_compute - dt_d2h) if dt_d2h < dt_compute else 0.0
            
            dt_dense: float = time.time() - t0
            return mt_target_preds, mt_target_labels, mt_target_weights, dt_dense, dt_h2d, dt_d2h, h2d_size, d2h_size, dt_compute, dt_bubble
