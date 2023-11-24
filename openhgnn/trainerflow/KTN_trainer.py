import dgl
import torch
from tqdm import tqdm
from ..models import build_model
from . import BaseFlow, register_flow
from ..utils import EarlyStopping, to_hetero_idx, to_homo_feature, to_homo_idx
from ..tasks import build_task
import warnings
import torch.nn as nn
import dgl.sparse as dglsp
from torch.utils.tensorboard import SummaryWriter


def process_category(labels: torch.tensor, num_classes: int) -> torch.tensor:
    device = labels.device

    if labels.shape[1] != num_classes:
        processed_labels = torch.zeros(
            (labels.shape[0], num_classes), dtype=torch.float, device=device
        )
        valid_labels = (labels >= 0).to(device)
        num_indices = valid_labels.sum(dim=1)
        weights = (1.0 / num_indices.clamp(min=1).unsqueeze(1)).float().to(device)
        indices = (labels * valid_labels).to(torch.int64).to(device)
        processed_labels.scatter_add_(1, indices, weights * valid_labels)
    else:
        processed_labels = labels.float().to(device)

    return processed_labels


@register_flow("KTN_trainer")
class KTNTrainer(BaseFlow):
    r"""
    Knowledge Transfer Learning flow.
    `Link to the paper <https://arxiv.org/abs/2203.02018>`__
    """

    def __init__(self, args):
        r"""
        Attributes:
        ----------
        source_type : str
            Node type of source domain.
        target_type : str
            Node type of target domain.
        task_type : str
            Task type of dataset.
        use_matching_loss : bool
            Whether to use matching loss.
        matching_w : nn.ModuleDict
            Trainable transformation matrix for each meta path.
        matching_path : int
            Meta path for matching loss.

        """
        if not hasattr(args, "use_matching_loss"):
            args.use_matching_loss = True
        if not hasattr(args, "feature_name"):
            args.feature_name = "feat"
        if not hasattr(args, "task_type"):
            args.task_type = "L1"
        if not hasattr(args, "source_type"):
            args.source_type = "paper"
        if not hasattr(args, "target_type"):
            args.target_type = "author"
        if not hasattr(args, "batch_size"):
            args.batch_size = 3072
        if not hasattr(args, "source_train_batch"):
            args.source_train_batch = 200
        if not hasattr(args, "source_test_batch"):
            args.source_test_batch = 50
        if not hasattr(args, "target_test_batch"):
            args.target_test_batch = 50
        if not hasattr(args, "patience"):
            args.patience = 10
            self.patience = 10
        if not hasattr(args, "matching_coeff"):
            args.matching_coeff = 1
        if not hasattr(args, "mini_batch_flag"):
            args.mini_batch_flag = False
        if not hasattr(args, "task_type"):
            args.task_type = "L1"
        super(KTNTrainer, self).__init__(args)
        self.args = args
        self.max_epoch = args.max_epoch
        self.device = args.device
        self.hg = self.task.get_graph().to(self.device)
        self.source_type = args.source_type
        self.target_type = args.target_type
        self.source_type = args.source_type
        self.target_type = args.target_type
        self.dataset = self.task.dataset
        self.dataset.to(self.device)
        self.use_matching_loss = args.use_matching_loss
        self.classifier = self.task.classifier.to(self.device)
        self.task_type = args.task_type
        self.num_layers = args.num_layers
        self.g = self.dataset.g.to(self.device)
        self.batch_size = args.batch_size
        self.source_train_batch = args.source_train_batch
        self.source_test_batch = args.source_test_batch
        self.target_test_batch = args.target_test_batch
        self.feature = self.dataset.get_feature()
        (
            self.source_train_idx,
            self.source_val_idx,
            self.source_test_idx,
        ) = self.task.get_split(self.source_type, self.device)
        (
            self.target_train_idx,
            self.target_val_idx,
            self.target_test_idx,
        ) = self.task.get_split(self.target_type, self.device)
        self.source_labels = self.dataset.get_labels(self.task_type, self.source_type)
        self.target_labels = self.dataset.get_labels(self.task_type, self.target_type)
        self.matching_coeff = args.matching_coeff
        self.args.in_dim = self.feature[self.source_type].shape[1]
        if not hasattr(self.args, "out_dim"):
            self.args.out_dim = self.args.hid_dim
        self.model_name = args.model
        self.model = (
            build_model(self.model)
            .build_model_from_args(self.args, self.hg)
            .to(self.device)
        )
        matching_w = {}
        self.label_dim = self.dataset.dims[self.task_type].item()
        abbrev = self.target_type[0].upper() + "-" + self.source_type[0].upper()
        self.matching_path = self.dataset.meta_paths_dict[abbrev]
        for matching_id, relation in enumerate(self.matching_path):
            matching_w[str(matching_id) + relation[1]] = nn.Linear(
                self.args.out_dim, self.args.out_dim
            ).to(self.device)
        self.matching_w = nn.ModuleDict(matching_w)
        for matching_id in self.matching_w.keys():
            nn.init.xavier_uniform_(self.matching_w[matching_id].weight)
        self.matching_loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.model.parameters()},
                {"params": self.matching_w.parameters()},
                {"params": self.classifier.parameters()},
            ],
            lr=self.args.lr,
        )
        self.writer = SummaryWriter(f"./openhgnn/output/{self.model_name}/")

    def train(self):
        self.preprocess()
        stopper = EarlyStopping(self.patience)
        epoch_iter = tqdm(range(self.max_epoch))
        for epoch in epoch_iter:
            if self.args.mini_batch_flag:
                train_loss, matching_loss = self._mini_train_step()
            else:
                train_loss, matching_loss = self._full_train_step()
            loss = train_loss + self.matching_coeff * matching_loss
            if epoch % self.evaluate_interval == 0:
                if self.args.mini_batch_flag:
                    acc = self._mini_test_step()
                else:
                    acc = self._full_test_step()
                self.writer.add_scalar("source_acc", acc[0], epoch)
                self.writer.add_scalar("target_acc", acc[1], epoch)
                self.writer.add_scalar("ktn_acc", acc[2], epoch)
                self.logger.train_info(
                    "Epoch {:d} | Train Loss {:.4f} | Matching Loss {:.4f} | Source Acc {:.4f} | Target Acc {:.4f} | KTN Acc {:.4f}".format(
                        epoch, train_loss, matching_loss, acc[0], acc[1], acc[2]
                    )
                )
                early_stop = stopper.loss_step(loss, self.model)
                if early_stop:
                    self.logger.train_info("Early Stop!\tEpoch:" + str(epoch))
                    break
        stopper.load_model(self.model)
        self.writer.close()

    def _full_train_step(self):
        self.model.train()
        self.matching_w.train()
        self.classifier.train()
        self.g = self.g.to(self.device)
        h_dict = self.feature
        h_dict = self.model(self.g, h_dict)
        logits = h_dict[self.source_type][self.source_train_idx]
        pred_y = self.classifier(logits)
        training_loss = self.loss_fn(pred_y, self.train_labels)
        h_S = h_dict[self.source_type]
        h_T = h_dict[self.target_type]
        matching_loss = self.get_matching_loss(h_S, h_T, self.g)
        loss = training_loss + self.matching_coeff * matching_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item(), matching_loss.item()

    def _mini_train_step(
        self,
    ):
        self.model.train()
        self.matching_w.train()
        self.classifier.train()
        loader_tqdm = tqdm(self.source_train_loader, ncols=120)
        batch_count = len(self.source_train_loader)
        if batch_count > self.source_train_batch:
            batch_count = self.source_train_batch
        all_loss = 0.0
        all_matching_loss = 0.0
        for i, (input_nodes, seeds, blocks) in enumerate(loader_tqdm):
            if i == self.source_train_batch:
                break
            sg = dgl.node_subgraph(self.g, input_nodes).to(self.device)
            h = {}
            for ntype in input_nodes.keys():
                h[ntype] = self.feature[ntype][input_nodes[ntype]].to(self.device)
            lbl = self.source_labels[input_nodes[self.source_type]].to(self.device)
            lbl = process_category(lbl, self.label_dim).to(self.device)
            h = self.model(sg, h)
            logits = self.classifier(h[self.source_type])
            loss = self.loss_fn(logits, lbl)
            h_S = h[self.source_type]
            h_T = h[self.target_type]
            matching_loss = self.get_matching_loss(h_S, h_T, sg)
            all_loss += loss.item()
            all_matching_loss += loss.item()
            loss = loss + self.matching_coeff * matching_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        all_loss = all_loss / batch_count
        all_matching_loss = all_matching_loss / batch_count
        return all_loss, all_matching_loss

    def get_matching_loss(self, h_S, h_T, hg):
        loss = torch.tensor([0.0], requires_grad=True).to(self.device)
        if self.use_matching_loss == False:
            return loss
        h_Z = h_T
        for matching_id, edge in enumerate(self.matching_path):
            h_Z = self.matching_w[str(matching_id) + edge[1]](h_Z)
            adj = hg.adj(etype=edge).transpose()
            h_Z = dglsp.spmm(adj, h_Z)
        loss = loss + self.matching_loss(h_S, h_Z)
        return loss

    def preprocess(self):
        if self.args.mini_batch_flag:
            sampler = dgl.dataloading.NeighborSampler(self.num_layers * [1])
            self.source_train_loader = dgl.dataloading.DataLoader(
                self.g,
                {self.source_type: self.source_train_idx},
                sampler,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=False,
            )
            self.source_test_loader = dgl.dataloading.DataLoader(
                self.g,
                {self.source_type: self.source_test_idx},
                sampler,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=False,
            )
            self.target_test_loader = dgl.dataloading.DataLoader(
                self.g,
                {self.target_type: self.target_test_idx},
                sampler,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=False,
            )
        else:
            self.train_labels = process_category(
                self.source_labels[self.source_train_idx], self.label_dim
            )
            self.source_test_labels = process_category(
                self.source_labels[self.source_test_idx], self.label_dim
            )
            self.target_test_labels = process_category(
                self.target_labels[self.target_test_idx], self.label_dim
            )
        return

    def _full_test_step(self):
        self.model.eval()
        self.classifier.eval()
        self.matching_w.eval()
        with torch.no_grad():
            h_dict = self.feature
            logits = self.model(self.g, h_dict)
            source_y = self.classifier(logits[self.source_type])[self.source_test_idx]
            origin_target_y = self.classifier(logits[self.target_type])[
                self.target_test_idx
            ]
            ktn_logits = logits[self.target_type]
            for matching_id, edge in enumerate(self.matching_path):
                ktn_logits = self.matching_w[str(matching_id) + edge[1]](ktn_logits)
            ktn_target_y = self.classifier(ktn_logits)[self.target_test_idx]
            source_acc = self.task.evaluate(source_y, self.source_test_labels)
            target_acc = self.task.evaluate(origin_target_y, self.target_test_labels)
            ktn_acc = self.task.evaluate(ktn_target_y, self.target_test_labels)
            return source_acc, target_acc, ktn_acc

    def _mini_test_step(self):
        self.model.eval()
        self.classifier.eval()
        self.matching_w.eval()
        with torch.no_grad():
            source_loader_tqdm = tqdm(self.source_test_loader, ncols=120)
            source_acc = 0
            source_batch_count = len(self.source_test_loader)
            if source_batch_count > self.source_test_batch:
                source_batch_count = self.source_test_batch
            for i, (input_nodes, seeds, blocks) in enumerate(source_loader_tqdm):
                if i == self.source_test_batch:
                    break
                h = {}
                for ntype in input_nodes.keys():
                    h[ntype] = self.feature[ntype][input_nodes[ntype]].to(self.device)
                lbl = self.source_labels[seeds[self.source_type]].to(self.device)
                lbl = process_category(lbl, self.label_dim).to(self.device)
                h = self.model(blocks, h)
                logits = self.classifier(h[self.source_type])
                acc = self.task.evaluate(logits, lbl)
                source_acc += acc
            source_acc /= source_batch_count
            target_loader_tqdm = tqdm(self.target_test_loader, ncols=120)
            target_batch_count = len(self.target_test_loader)
            if target_batch_count > self.target_test_batch:
                target_batch_count = self.target_test_batch
            target_acc = 0
            ktn_acc = 0
            for i, (input_nodes, seeds, blocks) in enumerate(target_loader_tqdm):
                if i == self.target_test_batch:
                    break
                h = {}
                for ntype in input_nodes.keys():
                    h[ntype] = self.feature[ntype][input_nodes[ntype]].to(self.device)
                lbl = self.target_labels[seeds[self.target_type]].to(self.device)
                lbl = process_category(lbl, self.label_dim)
                h = self.model(blocks, h)
                logits = self.classifier(h[self.target_type])
                origin_acc = self.task.evaluate(logits, lbl)
                target_acc += origin_acc
                target_h = h[self.target_type]
                for matching_id, edge in enumerate(self.matching_path):
                    target_h = self.matching_w[str(matching_id) + edge[1]](target_h)
                ktn_logits = self.classifier(target_h)
                _ktn_acc = self.task.evaluate(ktn_logits, lbl)
                ktn_acc += _ktn_acc
            target_acc /= target_batch_count
            ktn_acc /= target_batch_count
            return source_acc, target_acc, ktn_acc
