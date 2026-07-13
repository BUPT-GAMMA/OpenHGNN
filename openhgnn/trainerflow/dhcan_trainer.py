import copy
import time

import torch
from torch.utils.data import DataLoader

from ..models import build_model
from . import BaseFlow, register_flow


@register_flow("dhcan_trainer")
class DHCANTrainer(BaseFlow):
    r"""Target-node mini-batch trainer for decoupled HCAN.

    Equations 13-15 are precomputed once on CPU. Only the cached channels for
    the classification node type are kept on GPU during iterative training.
    """

    def __init__(self, args):
        if args.task != "node_classification":
            raise ValueError(
                "The D-HCAN trainer currently supports node classification only."
            )
        args.use_uva = True
        super(DHCANTrainer, self).__init__(args)

        self.category = self.task.dataset.category
        self.num_classes = self.task.dataset.num_classes
        self.train_idx, self.val_idx, self.test_idx = self.task.get_split()
        self.labels = self.task.get_labels().long().cpu()

        h_dict = self.hg.ndata.get("h", {})
        if not isinstance(h_dict, dict) or set(h_dict) != set(self.hg.ntypes):
            raise ValueError(
                "D-HCAN requires an input feature tensor for every node type."
            )
        feature_dims = {features.shape[1] for features in h_dict.values()}
        if len(feature_dims) != 1:
            raise ValueError(
                "D-HCAN attention requires equal feature dimensions across node types."
            )
        self.h_dict = {
            ntype: features.detach().cpu() for ntype, features in h_dict.items()
        }

        args.dhcan_input_dim = feature_dims.pop()
        args.out_dim = self.num_classes
        self.model = build_model(self.model).build_model_from_args(
            args, self.hg
        ).to(self.device)
        self.optimizer = self.candidate_optimizer[args.optimizer](
            self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        self.loss_fn = self.task.get_loss_fn()
        self.batch_size = getattr(args, "batch_size", 65536)
        self.cache_target_on_gpu = getattr(args, "cache_target_on_gpu", True)

    def _loader(self, node_idx, shuffle=False):
        return DataLoader(
            node_idx.cpu(),
            batch_size=self.batch_size,
            shuffle=shuffle,
            pin_memory=torch.cuda.is_available(),
        )

    def _train_epoch(self):
        self.model.train()
        total_loss = 0.0
        total_nodes = 0
        for node_idx in self._loader(self.train_idx, shuffle=True):
            logits = self.model.forward_cached(self.category, node_idx)
            labels = self.labels[node_idx].to(self.device, non_blocking=True)
            loss = self.loss_fn(logits, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * node_idx.numel()
            total_nodes += node_idx.numel()
        return total_loss / total_nodes

    @torch.no_grad()
    def _evaluate(self, node_idx):
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_nodes = 0
        for batch_idx in self._loader(node_idx):
            logits = self.model.forward_cached(self.category, batch_idx)
            labels = self.labels[batch_idx].to(self.device, non_blocking=True)
            total_loss += self.loss_fn(logits, labels).item() * batch_idx.numel()
            total_correct += (logits.argmax(dim=-1) == labels).sum().item()
            total_nodes += batch_idx.numel()
        return total_loss / total_nodes, total_correct / total_nodes

    def train(self):
        precompute_start = time.perf_counter()
        self.logger.info("[D-HCAN] Precomputing parameter-free channels on CPU.")
        self.model.precompute(
            self.hg, self.h_dict, target_ntype=self.category
        )
        if self.cache_target_on_gpu:
            self.model.move_cached_type(self.category, self.device)
        precompute_seconds = time.perf_counter() - precompute_start
        self.logger.info(
            "[D-HCAN] Precomputation finished in {:.2f}s.".format(
                precompute_seconds
            )
        )

        best_epoch = -1
        best_val_accuracy = float("-inf")
        best_state = None
        stale_epochs = 0
        epoch_times = []

        for epoch in range(self.max_epoch):
            epoch_start = time.perf_counter()
            train_loss = self._train_epoch()
            epoch_seconds = time.perf_counter() - epoch_start
            epoch_times.append(epoch_seconds)
            val_loss, val_accuracy = self._evaluate(self.val_idx)
            self.logger.info(
                "Epoch: {}, Train loss: {:.4f}, Valid loss: {:.4f}, "
                "Valid accuracy: {:.4f}, Time: {:.2f}s".format(
                    epoch,
                    train_loss,
                    val_loss,
                    val_accuracy,
                    epoch_seconds,
                )
            )

            if val_accuracy >= best_val_accuracy:
                best_epoch = epoch
                best_val_accuracy = val_accuracy
                best_state = copy.deepcopy(self.model.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= self.patience:
                    break

        self.model.load_state_dict(best_state)
        val_loss, val_accuracy = self._evaluate(self.val_idx)
        metric = {"valid": {"Accuracy": val_accuracy}}
        if self.args.test_flag:
            test_loss, test_accuracy = self._evaluate(self.test_idx)
            metric["test"] = {"Accuracy": test_accuracy}
            self.logger.info(
                "[Test Info] Valid accuracy: {:.4f}, Test accuracy: {:.4f}".format(
                    val_accuracy, test_accuracy
                )
            )

        return {
            "metric": metric,
            "epoch": best_epoch,
            "precompute_seconds": precompute_seconds,
            "mean_epoch_seconds": sum(epoch_times) / len(epoch_times),
        }
