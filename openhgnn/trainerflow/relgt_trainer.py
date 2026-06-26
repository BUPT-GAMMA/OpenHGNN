"""
RelGT Trainer Flow for OpenHGNN.

Ported from relgt/main_node_ddp.py.
Training logic, loss functions, and evaluation loop are identical
to the original implementation.
"""
import copy
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, L1Loss
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from ..models import build_model
from . import BaseFlow, register_flow

try:
    from relbench.base import TaskType
    from relbench.datasets import get_dataset
    from relbench.modeling.graph import make_pkey_fkey_graph
    from relbench.modeling.utils import get_stype_proposal
    from relbench.tasks import get_task
    HAS_RELBENCH = True
except ImportError:
    HAS_RELBENCH = False

try:
    from torch_frame import stype
    from torch_frame.config.text_embedder import TextEmbedderConfig
    HAS_TORCH_FRAME = True
except ImportError:
    HAS_TORCH_FRAME = False

try:
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data.distributed import DistributedSampler
    HAS_DIST = True
except ImportError:
    HAS_DIST = False

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    def tqdm(x, **kwargs):  # type: ignore[misc]
        return x


@register_flow("RelGT_trainer")
class RelGTTrainer(BaseFlow):
    """
    Trainer for the Relational Graph Transformer (RelGT).

    Manages the full training lifecycle for RelBench tasks:
    dataset loading, neighbour-token precomputation (HDF5 cache),
    supervised fine-tuning, and evaluation.

    Does NOT call super().__init__(args) — it fully self-manages the
    data pipeline (same pattern as AdapropTTrainer / IngramTrainer).

    Expected args attributes
    ------------------------
    relbench_dataset : str    e.g. "rel-f1"
    relbench_task    : str    e.g. "driver-top3"
    cache_dir        : str    root cache directory
    channels         : int    hidden width (default 128)
    num_layers       : int    local transformer layers (default 1)
    num_heads        : int    attention heads (default 4)
    gt_conv_type     : str    "full" | "local" | "global"
    num_neighbors    : int    K tokens per sample (default 100)
    num_centroids    : int    VQ codebook size (default 4096)
    ff_dropout       : float
    attn_dropout     : float
    lr               : float  (default 0.0001)
    weight_decay     : float  (default 1e-5)
    max_epoch        : int    (default 10)
    batch_size       : int    (default 512)
    seed             : int    (default 42)
    num_workers      : int    DataLoader workers (default 2)
    max_steps_per_epoch : int cap on batches per epoch (default 3000)
    use_wandb        : bool   log metrics to wandb (default False)
    """

    def __init__(self, args):
        # Bypass BaseFlow.__init__ — no DGL graph loading needed
        self.args   = args
        self.logger = getattr(args, "logger", None)
        self.device = getattr(args, "device", torch.device("cpu"))
        self.max_epoch = getattr(args, "max_epoch", 10)
        self.use_distributed = getattr(args, "use_distributed", False)

        if not HAS_RELBENCH:
            raise ImportError(
                "relbench is required for RelGTTrainer. "
                "Install with: pip install relbench"
            )
        if not HAS_TORCH_FRAME:
            raise ImportError(
                "pytorch-frame is required for RelGTTrainer. "
                "Install with: pip install pytorch-frame"
            )

        from ..dataset.RelGTDataset import RelGTTokens, GloveTextEmbedding

        relbench_dataset = getattr(args, "relbench_dataset", "rel-f1")
        relbench_task    = getattr(args, "relbench_task",    "driver-top3")
        cache_dir        = getattr(args, "cache_dir",
                                   os.path.expanduser("~/.cache/relbench_examples"))

        # ----------------------------------------------------------------
        # 1. Load RelBench dataset and build HeteroData
        # ----------------------------------------------------------------
        dataset = get_dataset(relbench_dataset, download=True)
        task    = get_task(relbench_dataset, relbench_task, download=True)

        stypes_cache_path = Path(f"{cache_dir}/{relbench_dataset}/stypes.json")
        try:
            with open(stypes_cache_path) as f:
                col_to_stype_dict = json.load(f)
            for table, col_to_stype in col_to_stype_dict.items():
                for col, stype_str in col_to_stype.items():
                    col_to_stype[col] = stype(stype_str)
        except FileNotFoundError:
            col_to_stype_dict = get_stype_proposal(dataset.get_db())
            stypes_cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(stypes_cache_path, "w") as f:
                json.dump(col_to_stype_dict, f, indent=2, default=str)

        data, col_stats_dict = make_pkey_fkey_graph(
            dataset.get_db(),
            col_to_stype_dict=col_to_stype_dict,
            text_embedder_cfg=TextEmbedderConfig(
                text_embedder=GloveTextEmbedding(device=self.device),
                batch_size=256,
            ),
            cache_dir=f"{cache_dir}/{relbench_dataset}/materialized",
        )

        # ----------------------------------------------------------------
        # 2. Build RelGTTokens for each split
        # ----------------------------------------------------------------
        precomputed_dir = getattr(
            args, "precomputed_dir",
            f"{cache_dir}/precomputed/{relbench_dataset}/{relbench_task}",
        )
        num_workers = getattr(args, "num_workers", 2)
        K           = getattr(args, "num_neighbors", 100)

        self.splits: Dict[str, RelGTTokens] = {
            split: RelGTTokens(
                data=data,
                task=task,
                K=K,
                split=split,
                undirected=True,
                precompute=getattr(args, "precompute", True),
                precomputed_dir=precomputed_dir,
                num_workers=num_workers,
                train_stage=getattr(args, "train_stage", "finetune"),
            )
            for split in ["train", "val", "test"]
        }

        # ----------------------------------------------------------------
        # 3. DataLoaders
        # ----------------------------------------------------------------
        batch_size = getattr(args, "batch_size", 512)

        if self.use_distributed and HAS_DIST:
            train_sampler = DistributedSampler(
                self.splits["train"], shuffle=True,
                seed=getattr(args, "seed", 42))
            val_sampler  = DistributedSampler(
                self.splits["val"], shuffle=False, drop_last=False,
                seed=getattr(args, "seed", 42))
            test_sampler = DistributedSampler(
                self.splits["test"], shuffle=False, drop_last=False,
                seed=getattr(args, "seed", 42))
        else:
            train_sampler = val_sampler = test_sampler = None

        make_loader = lambda split, sampler, shuffle: DataLoader(
            self.splits[split],
            batch_size=batch_size,
            sampler=sampler,
            shuffle=(shuffle and sampler is None),
            collate_fn=self.splits[split].collate,
            num_workers=num_workers,
            persistent_workers=(num_workers > 0),
            pin_memory=True,
        )

        self.loaders: Dict[str, DataLoader] = {
            "train": make_loader("train", train_sampler, True),
            "val":   make_loader("val",   val_sampler,   False),
            "test":  make_loader("test",  test_sampler,  False),
        }
        self._train_sampler = train_sampler

        # ----------------------------------------------------------------
        # 4. Task settings
        # ----------------------------------------------------------------
        self.task_obj  = task
        self.clamp_min = self.clamp_max = None

        if task.task_type == TaskType.BINARY_CLASSIFICATION:
            out_channels           = 1
            self.loss_fn           = BCEWithLogitsLoss()
            self.tune_metric       = "roc_auc"
            self.higher_is_better  = True
        elif task.task_type == TaskType.REGRESSION:
            out_channels           = 1
            self.loss_fn           = L1Loss()
            self.tune_metric       = "mae"
            self.higher_is_better  = False
            train_table = task.get_table("train")
            self.clamp_min, self.clamp_max = np.percentile(
                train_table.df[task.target_col].to_numpy(), [2, 98])
        elif task.task_type == TaskType.MULTILABEL_CLASSIFICATION:
            out_channels           = task.num_labels
            self.loss_fn           = BCEWithLogitsLoss()
            self.tune_metric       = "multilabel_auprc_macro"
            self.higher_is_better  = True
        else:
            raise ValueError(f"Unsupported task type: {task.task_type}")

        # ----------------------------------------------------------------
        # 5. Model
        # ----------------------------------------------------------------
        args.out_channels = out_channels
        data_info = {
            "num_nodes":        self.splits["train"].data.num_nodes,
            "max_neighbor_hop": self.splits["train"].max_neighbor_hop,
            "node_type_map":    self.splits["train"].node_type_to_index,
            "col_names_dict": {
                nt: data[nt].tf.col_names_dict for nt in data.node_types
            },
            "col_stats_dict": col_stats_dict,
        }
        self.model = build_model("RelGT").build_model_from_args(
            args, data_info).to(self.device)

        # Fix int16 buffers before DDP
        for name, buf in self.model.named_buffers():
            if buf.dtype == torch.int16:
                buf.data = buf.data.to(torch.int64)

        if self.use_distributed and HAS_DIST:
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = DDP(self.model,
                             device_ids=[self.device.index],
                             find_unused_parameters=True)

        # ----------------------------------------------------------------
        # 6. Optimiser
        # ----------------------------------------------------------------
        world_size = dist.get_world_size() if (self.use_distributed and HAS_DIST) else 1
        base_lr    = getattr(args, "lr", 0.0001) * world_size
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=base_lr,
            weight_decay=getattr(args, "weight_decay", 1e-5),
        )

        # ----------------------------------------------------------------
        # 7. Output path and logging
        # ----------------------------------------------------------------
        output_dir = getattr(args, "output_dir", "results")
        self.output_path = os.path.join(
            output_dir, relbench_dataset, relbench_task)
        os.makedirs(self.output_path, exist_ok=True)

        self.use_wandb = getattr(args, "use_wandb", False) and HAS_WANDB
        self.max_steps_per_epoch = getattr(args, "max_steps_per_epoch", 3000)
        self._global_step = 0

    # ----------------------------------------------------------------
    # Training loop (faithful port of main_node_ddp.py §7)
    # ----------------------------------------------------------------

    def _log(self, msg: str):
        if self.logger is not None:
            self.logger.info(msg)
        else:
            print(msg)

    def _forward_batch(self, batch: dict) -> torch.Tensor:
        neighbor_types = batch["neighbor_types"].to(self.device)
        node_indices   = batch["node_indices"].to(self.device)
        neighbor_hops  = batch["neighbor_hops"].to(self.device)
        neighbor_times = batch["neighbor_times"].to(self.device)
        edge_index     = batch["edge_index"].to(self.device)
        batch_vec      = batch["batch"].to(self.device)
        grouped_tf_dict = {
            "grouped_tfs":    batch["grouped_tfs"],
            "grouped_indices": batch["grouped_indices"],
            "flat_batch_idx":  batch["flat_batch_idx"],
            "flat_nbr_idx":    batch["flat_nbr_idx"],
        }
        return self.model(
            neighbor_types, node_indices, neighbor_hops, neighbor_times,
            grouped_tf_dict, edge_index=edge_index, batch=batch_vec,
        )

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        loss_accum = count_accum = 0
        total_steps = min(len(self.loaders["train"]), self.max_steps_per_epoch)

        if self._train_sampler is not None:
            self._train_sampler.set_epoch(epoch)

        for step, batch in enumerate(tqdm(
                self.loaders["train"], total=total_steps, desc="Train"), start=1):
            labels = batch["labels"].to(self.device)
            self.optimizer.zero_grad()
            pred = self._forward_batch(batch)
            pred = pred.view(-1) if pred.size(1) == 1 else pred
            loss = self.loss_fn(pred.float(), labels)
            loss.backward()
            clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            loss_val = loss.detach().item()
            if self.use_wandb:
                wandb.log({"train_loss": loss_val,
                           "global_step": self._global_step,
                           "lr": self.optimizer.param_groups[0]["lr"]})

            loss_accum  += loss_val * pred.size(0)
            count_accum += pred.size(0)
            self._global_step += 1
            if step >= self.max_steps_per_epoch:
                break

        return loss_accum / count_accum if count_accum > 0 else float("inf")

    @torch.no_grad()
    def _evaluate(self, split: str, epoch: int, eval_model) -> np.ndarray:
        eval_model.eval()
        pred_list: List[np.ndarray] = []
        idx_list:  List[np.ndarray] = []
        loader = self.loaders[split]

        if hasattr(loader.sampler, "set_epoch"):
            loader.sampler.set_epoch(epoch)

        for batch in tqdm(loader, desc=split.capitalize()):
            pred = self._forward_batch(batch)
            if self.task_obj.task_type == TaskType.REGRESSION:
                pred = torch.clamp(pred, self.clamp_min, self.clamp_max)
            if self.task_obj.task_type in (
                    TaskType.BINARY_CLASSIFICATION,
                    TaskType.MULTILABEL_CLASSIFICATION):
                pred = torch.sigmoid(pred)
            pred = pred.view(-1) if pred.size(1) == 1 else pred
            pred_list.append(pred.detach().cpu().numpy())
            idx_list.append(batch["global_idx"].cpu().numpy())

        all_preds = np.concatenate(pred_list) if pred_list else np.array([])
        return all_preds

    # ----------------------------------------------------------------
    # BaseFlow interface
    # ----------------------------------------------------------------

    def train(self):
        """Run the supervised fine-tuning stage."""
        best_val_metric = -math.inf if self.higher_is_better else math.inf
        best_state_dict = None

        eval_model = (self.model.module
                      if isinstance(self.model, DDP)
                      else self.model)

        for epoch in range(1, self.max_epoch + 1):
            train_loss = self._train_epoch(epoch)

            val_preds   = self._evaluate("val", epoch, eval_model)
            val_metrics = self.task_obj.evaluate(
                val_preds, self.task_obj.get_table("val"))

            self._log(
                f"Epoch {epoch:02d} | train_loss={train_loss:.4f} "
                f"| val {self.tune_metric}={val_metrics[self.tune_metric]:.4f}"
            )

            if self.use_wandb:
                wandb.log({"epoch": epoch, "epoch_train_loss": train_loss,
                           **{f"val_{k}": v for k, v in val_metrics.items()}})

            improved = (
                (self.higher_is_better
                 and val_metrics[self.tune_metric] >= best_val_metric)
                or (not self.higher_is_better
                    and val_metrics[self.tune_metric] <= best_val_metric)
            )
            if improved:
                best_val_metric = val_metrics[self.tune_metric]
                best_state_dict = copy.deepcopy(eval_model.state_dict())
                torch.save(best_state_dict,
                           os.path.join(self.output_path, "best_model.pt"))

        # Final evaluation with best model
        if best_state_dict is not None:
            eval_model.load_state_dict(best_state_dict)

        final_val_preds  = self._evaluate("val",  0, eval_model)
        final_test_preds = self._evaluate("test", 0, eval_model)

        val_metrics  = self.task_obj.evaluate(
            final_val_preds, self.task_obj.get_table("val"))
        test_metrics = self.task_obj.evaluate(final_test_preds)

        self._log(f"Best Val  metrics: {val_metrics}")
        self._log(f"Best Test metrics: {test_metrics}")

        results = {"val_metrics": val_metrics, "test_metrics": test_metrics}
        with open(os.path.join(self.output_path,
                               f"{getattr(self.args, 'seed', 0)}.json"), "w") as f:
            json.dump(results, f, indent=4)

        return results
