import argparse
import copy
import dgl
import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F
from openhgnn.models import build_model

from . import BaseTask, register_task
from ..utils import build_dataset, get_nodes_dict
from ..utils import cal_acc


@register_task("node_classification")
class NodeClassification(BaseTask):
    """Node classification tasks."""

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        """Add tasks-specific arguments to the parser."""
        # fmt: off
        parser.add_argument("--missing-rate", type=int, default=0, help="missing rate, from 0 to 100")
        # fmt: on

    def __init__(self, args):
        super(NodeClassification, self).__init__(args)

        self.args = args
        self.model_name = args.model
        self.device = args.device

        self.hg, self.category, num_classes = build_dataset(args.model, args.dataset)
        self.hg = self.hg.to(self.device)
        self.g = dgl.to_homogeneous(self.hg)


        self.args.num_classes = num_classes

        self.model = build_model(self.model_name).build_model_from_args(self.args, self.hg)
        self.model.set_device(self.device)


        self.set_loss_fn(F.cross_entropy)
        self.evaluator = cal_acc

        self.trainer = self.get_trainer(self.model, self.args)
        if not self.trainer:
            self.optimizer = (
                torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
                if not hasattr(self.model, "get_optimizer")
                else self.model.get_optimizer(args)
            )

            self.model = self.model.to(self.device)
            self.patience = args.patience
            self.max_epoch = args.max_epoch

    def preprocess(self):
        from openhgnn.utils import get_idx
        train_idx, self.test_idx, self.labels = get_idx(self.hg, self.g, self.category)
        if self.args.validation:
            self.val_idx = train_idx[:len(train_idx) // 10]
            self.train_idx = train_idx[len(train_idx) // 10:]
        else:
            self.val_idx = train_idx
            self.train_idx = train_idx

    def train(self):
        if self.trainer:
            result = self.trainer.fit(self.model, self.dataset)
            if issubclass(type(result), torch.nn.Module):
                self.model = result
                self.model.to(self.data.x.device)
            else:
                return result
        else:
            self.preprocess()
            epoch_iter = tqdm(range(self.max_epoch))
            patience = 0
            best_score = 0
            best_loss = np.inf
            max_score = 0
            min_loss = np.inf
            best_model = copy.deepcopy(self.model)
            for epoch in tqdm(range(self.max_epoch)):
                self._train_step()
                acc, losses = self._test_step()
                train_acc = acc["train"]
                val_acc = acc["val"]
                val_loss = losses["val"]
                epoch_iter.set_description(
                    f"Epoch: {epoch:03d}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, ValLoss:{val_loss: .4f}"
                )
                if val_loss <= min_loss or val_acc >= max_score:
                    if val_loss <= best_loss:  # and val_acc >= best_score:
                        best_loss = val_loss
                        best_score = val_acc
                        best_model = copy.deepcopy(self.model)
                    min_loss = np.min((min_loss, val_loss.cpu()))
                    max_score = np.max((max_score, val_acc))
                    patience = 0
                else:
                    patience += 1
                    if patience == self.patience:
                        epoch_iter.close()
                        break
            print(f"Valid accurracy = {best_score: .4f}")
            self.model = best_model
        test_acc, _ = self._test_step(split="test")
        val_acc, _ = self._test_step(split="val")
        print(f"Test accuracy = {test_acc:.4f}")
        return dict(Acc=test_acc, ValAcc=val_acc)

    def _train_step(self):
        self.model.train()
        self.optimizer.zero_grad()
        self.model.node_classification_loss(self.hg, self.labels, self.train_idx, self.category).backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
        self.optimizer.step()

    def _test_step(self, split=None, logits=None):

        self.model.eval()
        with torch.no_grad():
            logits = logits if logits else self.model.predict(self.hg)[self.category]
        if split == "train":
            mask = self.train_idx
        elif split == "val":
            mask = self.val_idx
        elif split == "test":
            mask = self.test_idx
        else:
            mask = None

        if mask is not None:
            loss = self.loss_fn(logits[mask], self.labels[mask])
            metric = self.evaluator(logits[mask], self.labels[mask])
            return metric, loss
        else:

            masks = {'train': self.train_idx, 'val': self.val_idx, 'test': self.test_idx}
            metrics = {key: self.evaluator(logits[mask], self.labels[mask]) for key, mask in masks.items()}
            losses = {key: self.loss_fn(logits[mask], self.labels[mask]) for key, mask in masks.items()}
            return metrics, losses
