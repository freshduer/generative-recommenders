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

import os
import time
import uuid
from threading import Event
from typing import Dict, List, Optional, Tuple, Union
from collections import Counter

import torch
import torch.multiprocessing as mp
import torchrec
from generative_recommenders.dlrm_v3.checkpoint import (
    load_nonsparse_checkpoint,
    load_sparse_checkpoint,
)
from generative_recommenders.dlrm_v3.datasets.dataset import Samples
from generative_recommenders.dlrm_v3.inference.inference_modules import (
    get_hstu_model,
    HSTUSparseInferenceModule,
    move_sparse_output_to_device,
)
from generative_recommenders.dlrm_v3.utils import Profiler
from generative_recommenders.modules.dlrm_hstu import DlrmHSTUConfig, SequenceEmbedding
from torch import quantization as quant
from torchrec.distributed.quant_embedding import QuantEmbeddingCollection
from torchrec.modules.embedding_configs import EmbeddingConfig, QuantConfig
from torchrec.test_utils import get_free_port
import torch.distributed as dist
from torchrec.distributed.model_parallel import DistributedModelParallel, get_default_sharders
from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
from torchrec.distributed.planner.types import ParameterConstraints
from torchrec.distributed.comm import get_local_size
from torchrec.distributed.types import (
    ModuleSharder,
    ShardingEnv,
    ShardingPlan,
    ShardingType,
)
from torchrec.modules.embedding_modules import EmbeddingCollection

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def init_distributed(rank: int, world_size: int, backend: str = "nccl"):
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    return dist.group.WORLD

class HSTUModelFamily:
    def __init__(
        self,
        hstu_config: DlrmHSTUConfig,
        table_config: Dict[str, EmbeddingConfig],
        output_trace: bool = False,
        constraints: Dict[str, ParameterConstraints] = {},
        backend: str = "nccl",
    ) -> None:
        self.hstu_config = hstu_config
        self.table_config = table_config
        self.sparse: ModelFamilySparseDist = ModelFamilySparseDist(
            hstu_config=hstu_config,
            table_config=table_config,
            constraints=constraints,
        )

        assert torch.cuda.is_available(), "CUDA is required for this benchmark."
        ngpus = torch.cuda.device_count()
        self.world_size = int(os.environ.get("WORLD_SIZE", str(ngpus)))
        print(f"Using {self.world_size} GPU(s)...")

        # dense_model_family_clazz = (
        #     ModelFamilyDenseDist
        #     if self.world_size > 1
        #     else ModelFamilyDenseSingleWorker
        # )
        dense_model_family_clazz = ModelFamilyDenseSingleWorker

        self.dense: Union[ModelFamilyDenseDist, ModelFamilyDenseSingleWorker] = (
            dense_model_family_clazz(
                hstu_config=hstu_config,
                table_config=table_config,
                output_trace=output_trace,
            )
        )

    def version(self) -> str:
        return torch.__version__

    def name(self) -> str:
        return "model-family-hstu"

    def load(self, model_path: str) -> None:
        self.sparse.load(model_path=model_path)
        self.dense.load(model_path=model_path)

    def predict(
        self, samples: Optional[Samples]
    ) -> Optional[Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]]:
        if samples is None:
            return self.dense.predict(None, None, 0, None, 0, None)
        (
            seq_embeddings,
            payload_features,
            max_uih_len,
            uih_seq_lengths,
            max_num_candidates,
            num_candidates,
        ) = self.sparse.predict(samples)
        return self.dense.predict(
            seq_embeddings,
            payload_features,
            max_uih_len,
            uih_seq_lengths,
            max_num_candidates,
            num_candidates,
        )


class ModelFamilySparseDist:
    def __init__(
        self,
        hstu_config: DlrmHSTUConfig,
        table_config: Dict[str, EmbeddingConfig],
        constraints: Dict[str, ParameterConstraints],
    ) -> None:
        super(ModelFamilySparseDist, self).__init__()
        self.hstu_config = hstu_config
        self.table_config = table_config
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.device = torch.device(f"cuda:{self.rank}")
        torch.cuda.set_device(f"cuda:{self.rank}")
        logger.info(f"[rank {self.rank}] local_size: {get_local_size()}")
        self.constraints = constraints
        self.module = None
    
    def _load_single_sparse(self,model_path):
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
            self.module = sparse_arch
        print(f"sparse module is {self.module}")

    def _load_distributed_sparse(self, 
        constraints: Dict[str, ParameterConstraints],
    ) -> None:
        # 环境变量控制是否启用自定义分片
        # if os.environ.get("CUSTOM_SHARDING", "0") == "1":
        from generative_recommenders.dlrm_v3.inference.custom_sharding import CustomEmbeddingCollection
        logger.info(f"[rank {self.rank}] Using CustomEmbeddingCollection manual sharding")
        custom_ec = CustomEmbeddingCollection(
            table_config={cfg.name: cfg for cfg in self.table_config.values()},
            constraints=constraints,
            device=torch.device("cuda")
        )
        # 注意: 如果需要从原始 checkpoint 中加载嵌入，请在外部调用 custom_ec.set_weights_for_key(key, full_weight)
        self.module = HSTUSparseInferenceModule(
            table_config=self.table_config,
            hstu_config=self.hstu_config,
            embedding_collection=custom_ec,
        ).to(self.device)
        return
    
        ebc = EmbeddingCollection(
            tables=list(self.table_config.values()),
            need_indices=False,
            device=torch.device("meta"),
        )

        # planner + sharders
        topology = Topology(world_size=self.world_size, compute_device="cuda")
        planner = EmbeddingShardingPlanner(
            topology=topology,
            constraints=constraints,
        )
        sharders = get_default_sharders()
        logger.info(f"dist.get_rank():{dist.get_rank()}")
        torch.cuda.set_device(dist.get_rank())
        plan: ShardingPlan = planner.collective_plan(module=ebc, sharders=sharders, pg=dist.group.WORLD)
        logger.info(f"[rank {self.rank}] sharding plan _serialize:{plan._serialize()}.")
        # DMP 封装
        sharded_model= DistributedModelParallel(
            module=ebc,
            env=ShardingEnv.from_process_group(dist.group.WORLD),
            plan=plan,
            sharders=sharders,
            device=self.device,
        )

        self.module = HSTUSparseInferenceModule(
            table_config=self.table_config,
            hstu_config=self.hstu_config,
            embedding_collection=sharded_model,
        ).to(self.device)
        

    def load(self, model_path: str) -> None:
        logger.info(f"Loading sparse module from {model_path}")
        if self.world_size > 1:
            # 多卡 DMP 逻辑
            logger.info(f"[rank {self.rank}] Loading sparse module (DMP) from {model_path}")
            self._load_distributed_sparse(constraints=self.constraints)
            self.module.eval()
        else:
            # 单卡逻辑
            logger.info(f"rank:{self.rank} Loading sparse module (single GPU) from {model_path}")
            self._load_single_sparse(model_path)

    def predict(
        self, samples: Samples
    ) -> Tuple[
        Dict[str, SequenceEmbedding],
        Dict[str, torch.Tensor],
        int,
        torch.Tensor,
        int,
        torch.Tensor,
    ]:
        with torch.profiler.record_function("sparse forward"):
            assert self.module is not None
            # 自动获取模型所在 device
            logger.info(f"device:{self.device}")
            # 把 KeyedJaggedTensor 移到 GPU
            uih_features = samples.uih_features_kjt.to(self.device)
            candidates_features = samples.candidates_features_kjt.to(self.device)
            (
                seq_embeddings,
                payload_features,
                max_uih_len,
                uih_seq_lengths,
                max_num_candidates,
                num_candidates,
            ) = self.module(
                uih_features=uih_features,
                candidates_features=candidates_features,
            )
            return (
                seq_embeddings,
                payload_features,
                max_uih_len,
                uih_seq_lengths,
                max_num_candidates,
                num_candidates,
            )


class ModelFamilyDenseDist:
    def __init__(
        self,
        hstu_config: DlrmHSTUConfig,
        table_config: Dict[str, EmbeddingConfig],
        output_trace: bool = False,
    ) -> None:
        super(ModelFamilyDenseDist, self).__init__()
        self.hstu_config = hstu_config
        self.table_config = table_config
        self.output_trace = output_trace

        ngpus = torch.cuda.device_count()
        self.world_size = int(os.environ.get("WORLD_SIZE", str(ngpus)))
        self.rank = 0
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(get_free_port())
        self.dist_backend = "nccl"

        ctx = mp.get_context("spawn")
        self.samples_q: List[mp.Queue] = [ctx.Queue() for _ in range(self.world_size)]
        self.predictions_cache = [  # pyre-ignore[4]
            mp.Manager().dict() for _ in range(self.world_size)
        ]
        self.main_lock: Event = ctx.Event()

    def load(self, model_path: str) -> None:
        print(f"Loading dense module from {model_path}")

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
        self.main_lock.wait()

    def distributed_setup(self, rank: int, world_size: int, model_path: str) -> None:
        model = get_hstu_model(
            table_config=self.table_config,
            hstu_config=self.hstu_config,
            table_device="cpu",
            max_hash_size=100,
            is_dense=True,
        ).to(torch.bfloat16)
        device = torch.device(f"cuda:{rank}")
        load_nonsparse_checkpoint(
            model=model, device=device, optimizer=None, path=model_path
        )
        self.main_lock.set()
        model = model.to(device)
        model.eval()
        profiler = Profiler(rank) if self.output_trace else None

        with torch.no_grad():
            while True:
                if self.samples_q[rank].empty():
                    time.sleep(0.001)
                    continue
                item = self.samples_q[rank].get()
                # If -1 is received terminate all subprocesses
                if item == -1:
                    break
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
                if self.output_trace:
                    assert profiler is not None
                    profiler.step()
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
                    mt_target_preds = mt_target_preds.detach().to(
                        device="cpu", non_blocking=True
                    )
                    if mt_target_labels is not None:
                        mt_target_labels = mt_target_labels.detach().to(
                            device="cpu", non_blocking=True
                        )
                    if mt_target_weights is not None:
                        mt_target_weights = mt_target_weights.detach().to(
                            device="cpu", non_blocking=True
                        )
                    self.predictions_cache[rank][id] = (
                        mt_target_preds,
                        mt_target_labels,
                        mt_target_weights,
                    )

    def capture_output(
        self, id: uuid.UUID, rank: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        out = None
        while out is None:
            time.sleep(0.001)
            out = self.predictions_cache[rank].get(id, None)
        self.predictions_cache[rank].pop(id)
        return out

    def get_rank(self) -> int:
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
    ) -> Optional[Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]]:
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
        seq_embeddings, payload_features, uih_seq_lengths, num_candidates = (
            move_sparse_output_to_device(
                seq_embeddings=seq_embeddings,
                payload_features=payload_features,
                uih_seq_lengths=uih_seq_lengths,
                num_candidates=num_candidates,
                device=device,
            )
        )
        self.main_lock.wait()
        self.main_lock.clear()
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
        out = self.capture_output(id, rank)
        self.main_lock.set()
        return out


class ModelFamilyDenseSingleWorker:
    def __init__(
        self,
        hstu_config: DlrmHSTUConfig,
        table_config: Dict[str, EmbeddingConfig],
        output_trace: bool = False,
    ) -> None:
        self.model: Optional[torch.nn.Module] = None
        self.hstu_config = hstu_config
        self.table_config = table_config
        self.output_trace = output_trace
        self.rank = dist.get_rank()       # 当前进程的 rank
        self.world_size = dist.get_world_size()  # 总进程数
        gpu_count = torch.cuda.device_count()  # 当前节点 GPU 数量
        # 将 rank 映射到 GPU
        self.device = torch.device(f"cuda:{self.rank % gpu_count}")
        torch.cuda.set_device(self.device)
        self.profiler: Optional[Profiler] = (
            Profiler(rank=0) if self.output_trace else None
        )

    def load(self, model_path: str) -> None:
        print(f"Loading dense module from {model_path}")
        self.model = (
            get_hstu_model(
                table_config=self.table_config,
                hstu_config=self.hstu_config,
                table_device=self.device,
                is_dense=True,
            )
            .to(self.device)
            .to(torch.bfloat16)
        )
        load_nonsparse_checkpoint(model=self.model, optimizer=None, path=model_path)
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
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.output_trace:
            assert self.profiler is not None
            self.profiler.step()
        assert (
            payload_features is not None
            and uih_seq_lengths is not None
            and num_candidates is not None
            and seq_embeddings is not None
        )
        with torch.profiler.record_function("dense forward"):
            t0 = time.perf_counter()
            logger.info(f"rank:{self.rank} before move_sparse_output_to_device")
            seq_embeddings, payload_features, uih_seq_lengths, num_candidates = (
                move_sparse_output_to_device(
                    seq_embeddings=seq_embeddings,
                    payload_features=payload_features,
                    uih_seq_lengths=uih_seq_lengths,
                    num_candidates=num_candidates,
                    device=self.device,
                )
            )
            logger.info(f"rank:{self.rank} main_forward start")
            assert self.model is not None
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
            dt_ms = (time.perf_counter() - t0) * 1000.0
            logger.info(f"[dense]rank:{self.rank} forward took {dt_ms:.3f} ms")
            assert mt_target_preds is not None
            mt_target_preds = mt_target_preds.detach().to(
                device="cpu", non_blocking=True
            )
            if mt_target_labels is not None:
                mt_target_labels = mt_target_labels.detach().to(
                    device="cpu", non_blocking=True
                )
            if mt_target_weights is not None:
                mt_target_weights = mt_target_weights.detach().to(
                    device="cpu", non_blocking=True
                )
            return mt_target_preds, mt_target_labels, mt_target_weights
