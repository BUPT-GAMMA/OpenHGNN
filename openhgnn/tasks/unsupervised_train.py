import argparse
import copy
import dgl
import numpy as np
import torch as th
from tqdm import tqdm
import torch.nn.functional as F
from openhgnn.models import build_model

from . import BaseTask, register_task
from ..utils import build_dataset, get_idx, cal_acc, load_link_pred


def compute_loss(pos_score, neg_score):
    # an example hinge loss
    loss = []
    for i in pos_score:
        loss.append(F.logsigmoid(pos_score[i]))
        loss.append(F.logsigmoid(-neg_score[i]))
    loss = th.cat(loss)
    return -loss.mean()

@register_task("unsupervised_train")
class UnsupervisedTrain(BaseTask):
    def __init__(self, args):
        super(UnsupervisedTrain, self).__init__(args)

        self.args = args
        self.model_name = args.model
        self.device = args.device

        self.hg, self.category, num_classes = build_dataset(args.model, args.dataset)
        self.hg = self.hg.to(self.device)
        self.g = dgl.to_homogeneous(self.hg)

        self.model = build_model(self.model_name).build_model_from_args(self.args, self.hg)
        self.model.set_device(self.device)
        self.model = self.model.to(self.device)


        self.set_loss_fn(compute_loss)
        self.evaluator = cal_acc

        if hasattr(self.model, "split_dataset"):
            pass
        else:
            pass
        self.evaluate_interval = 1
        self.max_epoch = args.max_epoch
        self.patience = args.patience
        self.grad_norm = 1.5
        self.optimizer = th.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.mini_batch_flag = True
        if self.mini_batch_flag == True:
            self.dataloader_it = self.model.get_dataloader(self.hg, self.args)

    def preprocess(self):
        self.train_batch = load_link_pred('./openhgnn/dataset/a_a_list_train.txt')
        self.test_batch = load_link_pred('./openhgnn/dataset/a_a_list_test.txt')
        self.train_idx, self.test_idx, self.labels = get_idx(self.hg, self.category)

    def train(self):
        self.preprocess()
        best_model = None
        best_score = 0
        patience = 0
        auc_score = 0
        epoch_iter = tqdm(range(self.max_epoch))
        for epoch in epoch_iter:
            train_loss = self._train_step()
            print('Epoch {:05d} |Train - Loss: {:.4f}'.format(epoch, train_loss))
            if (epoch + 1) % self.evaluate_interval == 0:
                self.link_preddiction()
                self.node_classification()
                # if auc_score > best_score:
                #     best_score = auc_score
                #     best_model = copy.deepcopy(self.model)
                #     patience = 0
                # else:
                #     patience += 1
                #     if patience == self.patience:
                #         break
            epoch_iter.set_description(f"Epoch {epoch: 3d}: TrainLoss: {train_loss: .4f}, AUC: {auc_score: .4f}")
        self.model = best_model
        test_score = self._test_step(split="test")
        val_score = self._test_step(split="val")
        print(f"Val: {val_score: .4f}, Test: {test_score: .4f}")
        return dict(AUC=test_score)

    def _train_step(self):
        self.model.train()
        self.optimizer.zero_grad()
        if self.mini_batch_flag == True:
            for batch_id in tqdm(range(self.args.batches_per_epoch)):
                positive_graph, negative_graph, blocks = next(self.dataloader_it)
                blocks = [b.to(self.device) for b in blocks]
                positive_graph = positive_graph.to(self.device)
                negative_graph = negative_graph.to(self.device)
                # we need extract multi-feature
                input_features = self.model.extract_feature(blocks[0], self.hg.ntypes)

                x = self.model(blocks[0], input_features)
                loss = self.loss_fn(self.model.pred(positive_graph, x), self.model.pred(negative_graph, x))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            return loss.item()
        else:
            pass

    def get_embedding(self):
        self.model.eval()
        input_features = self.model.extract_feature(self.hg, self.hg.ntypes)
        x = self.model(self.model.preprocess(self.hg, self.args).to(self.args.device), input_features)
        return x

    def link_preddiction(self):
        x = self.get_embedding()
        self.model.lp_evaluator(x[self.category].to('cpu').detach(), self.train_batch, self.test_batch)


    def node_classification(self):
        x = self.get_embedding()
        self.model.nc_evaluator(x[self.category].to('cpu').detach(), self.labels, self.train_idx, self.test_idx)
    def _test_step(self, split="val"):
        self.model.eval()
        if split == "val":
            pos_edges = self.data.val_edges
            neg_edges = self.data.val_neg_edges
        elif split == "test":
            pos_edges = self.data.test_edges
            neg_edges = self.data.test_neg_edges
        else:
            raise ValueError
        train_edges = self.data.train_edges
        edges = torch.cat([pos_edges, neg_edges], dim=1)
        labels = self.get_link_labels(pos_edges.shape[1], neg_edges.shape[1], self.device).long()
        with self.data.local_graph():
            self.data.edge_index = train_edges
            with torch.no_grad():
                emb = self.model(self.data)
                pred = (emb[edges[0]] * emb[edges[1]]).sum(-1)
        pred = torch.sigmoid(pred)

        auc_score = roc_auc_score(labels.cpu().numpy(), pred.cpu().numpy())
        return auc_score

    def _train_test_edge_split(self):
        num_nodes = self.data.x.shape[0]
        (
            (train_edges, val_edges, test_edges),
            (val_false_edges, test_false_edges),
        ) = self.train_test_edge_split(self.data.edge_index, num_nodes)
        self.data.train_edges = train_edges
        self.data.val_edges = val_edges
        self.data.test_edges = test_edges
        self.data.val_neg_edges = val_false_edges
        self.data.test_neg_edges = test_false_edges

    @staticmethod
    def train_test_edge_split(edge_index, num_nodes, val_ratio=0.1, test_ratio=0.2):
        row, col = edge_index
        mask = row > col
        row, col = row[mask], col[mask]
        num_edges = row.size(0)

        perm = torch.randperm(num_edges)
        row, col = row[perm], col[perm]

        num_val = int(num_edges * val_ratio)
        num_test = int(num_edges * test_ratio)

        index = [[0, num_val], [num_val, num_val + num_test], [num_val + num_test, -1]]
        sampled_rows = [row[l:r] for l, r in index]  # noqa E741
        sampled_cols = [col[l:r] for l, r in index]  # noqa E741

        # sample false edges
        num_false = num_val + num_test
        row_false = np.random.randint(0, num_nodes, num_edges * 5)
        col_false = np.random.randint(0, num_nodes, num_edges * 5)

        indices_false = row_false * num_nodes + col_false
        indices_true = row.cpu().numpy() * num_nodes + col.cpu().numpy()
        indices_false = list(set(indices_false).difference(indices_true))
        indices_false = np.array(indices_false)
        row_false = indices_false // num_nodes
        col_false = indices_false % num_nodes

        mask = row_false > col_false
        row_false = row_false[mask]
        col_false = col_false[mask]

        edge_index_false = np.stack([row_false, col_false])
        if edge_index.shape[1] < num_false:
            ratio = edge_index_false.shape[1] / num_false
            num_val = int(ratio * num_val)
            num_test = int(ratio * num_test)
        val_false_edges = torch.from_numpy(edge_index_false[:, 0:num_val])
        test_fal_edges = torch.from_numpy(edge_index_false[:, num_val : num_test + num_val])

        def to_undirected(_row, _col):
            _edge_index = torch.stack([_row, _col], dim=0)
            _r_edge_index = torch.stack([_col, _row], dim=0)
            return torch.cat([_edge_index, _r_edge_index], dim=1)

        train_edges = to_undirected(sampled_rows[2], sampled_cols[2])
        val_edges = torch.stack([sampled_rows[0], sampled_cols[0]])
        test_edges = torch.stack([sampled_rows[1], sampled_cols[1]])
        return (train_edges, val_edges, test_edges), (val_false_edges, test_fal_edges)

    @staticmethod
    def get_link_labels(num_pos, num_neg, device=None):
        labels = torch.zeros(num_pos + num_neg)
        labels[:num_pos] = 1
        if device is not None:
            labels = labels.to(device)
        return labels.float()