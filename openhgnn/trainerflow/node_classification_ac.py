import argparse
import copy
import dgl
import numpy as np
import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from ..models import build_model
from . import BaseFlow, register_flow
from ..utils import EarlyStopping
from ..utils.logger import printInfo, printMetric


@register_flow("node_classification_ac")
class NodeClassificationAC(BaseFlow):
    """
    Node classification with attribute completion flows.

    Supported Model: MAGNN_AC

    Supported Dataset：IMDB

    The task is to classify the nodes of HIN(Heterogeneous Information Network).

    Note: If the output dim is not equal the number of classes, a MLP will follow the gnn model.
    """

    def __init__(self, args):
        super(NodeClassificationAC, self).__init__(args)
        
        if hasattr(args, 'metric'):
            self.metric = args.metric
        else:
            self.metric = 'f1'
        self.model_name = args.HIN

        self.num_classes = self.task.dataset.num_classes
        if hasattr(self.task.dataset, 'in_dim'):
            self.args.in_dim = self.task.dataset.in_dim
        if not hasattr(self.task.dataset, 'out_dim') or args.out_dim != self.num_classes:
            print('Modify the out_dim with num_classes')
            args.out_dim = self.num_classes

        self.args.category = self.task.dataset.category
        self.category = self.args.category
        self.model = build_model(
            self.model_name).build_model_from_args(self.args, self.hg).to(self.device)
        self.hgnn_ac = build_model(
            "HGNN_AC").build_model_from_args(self.args, self.hg).to(self.device)
        self.optimizer = torch.optim.Adam([{'params': self.model.parameters()},
                                           {'params': self.hgnn_ac.parameters()}],
                                          lr=args.lr, weight_decay=args.weight_decay)

        # for ntype in self.model.in_feats.keys():
        #     in_dims.append(self.hg.nodes[ntype].data['feat'].shape[1])

        # self.input_projection = torch.nn.ModuleDict()
        # for ntype in self.model.in_feats.keys():
        #     self.input_projection[ntype] = torch.nn.Linear(in_features=self.model.in_feats[ntype],
        #                                                    out_features=self.model.h_feats * self.model.num_heads)
        # for layer in self.input_projection.values():
        #     torch.nn.init.xavier_normal_(layer.weight, gain=1.414)
        # self.feat_drop = torch.nn.Dropout(p = args.dropout)
        self.evaluator = self.task.get_evaluator('f1')

        self.train_idx, self.valid_idx, self.test_idx = self.task.get_split()
        self.labels = self.task.get_labels().to(self.device)
        if self.args.mini_batch_flag:
            # sampler = dgl.dataloading.MultiLayerNeighborSampler([self.args.fanout] * self.args.n_layers)
            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(
                self.args.n_layers)
            self.loader = dgl.dataloading.NodeDataLoader(
                self.hg.to('cpu'), {
                    self.category: self.train_idx.to('cpu')}, sampler,
                batch_size=self.args.batch_size, device=self.device, shuffle=True, num_workers=0)
        return

    def preprocess(self):
        r'''
        Parameters
        ----------
        ntypes: list
            node types of the dataset
        in_dim: list
            feature dimension of each type of node
        adj: matrix
            adjacency matrix related to the source node type
        feat_keep_idx: list
            nodes that reserve feature
        feat_drop_idx: list
            nodes that drop feature
        '''
        self.ntypes = self.hg.ntypes
        self.in_dim = []
        self.adj = {}
        for ntype in self.ntypes:
            self.in_dim.append(self.hg.nodes[ntype].data['h'].shape[0])        
        
        self.feat_keep_idx, self.feat_drop_idx = train_test_split(np.arange(self.in_dim[self.args.src_node_type]),
                                                                  test_size=self.args.feats_drop_rate)
        src = self.ntypes[self.args.src_node_type]

        for ntype in self.ntypes:
            dst = ntype
            edge = src + '-' + dst
            matrix = torch.zeros(
                  (self.hg.num_nodes(ntype=src), self.hg.num_nodes(ntype=dst)))
            if edge in self.hg.etypes:
                for i, j in enumerate(self.hg.edges(etype=edge)[0]):
                    i = self.hg.edges(etype=edge)[1][i]
                    matrix[j][i] = 1
            matrix = matrix.to(self.device)
            self.adj[dst] = matrix
        self.preprocess_feature()
        
        return

    def train(self):
        self.preprocess()
        stopper = EarlyStopping(self.args.patience, self._checkpoint)
        epoch_iter = tqdm(range(self.max_epoch))
        for epoch in epoch_iter:
            if self.args.mini_batch_flag:
                train_loss = self._mini_train_step()
            else:
                train_loss = self._full_train_step()
            if epoch % self.evaluate_interval == 0:
                if self.args.mini_batch_flag and hasattr(self, 'val_loader'):
                    train_score, train_loss = self._mini_test_step(mode='train')
                    val_score, val_loss = self._mini_test_step(mode='validation')
                else:
                    score, losses = self._full_test_step()
                    train_score = score["train"]
                    val_score = score["val"]
                    val_loss = losses["val"]

                printInfo(self.metric, epoch, train_score, train_loss, val_score, val_loss)

                early_stop = stopper.loss_step(val_loss, self.model)
                if early_stop:
                    print('Early Stop!\tEpoch:' + str(epoch))
                    break

        stopper.load_model(self.model)
        # save results for HGBn
        if self.args.dataset[:4] == 'HGBn':

            if self.args.mini_batch_flag and hasattr(self, 'val_loader'):
                val_score, val_loss = self._mini_test_step(mode='validation')
            else:
                val_score, val_loss = self._full_test_step(mode='validation')

            printMetric(self.metric, val_score, 'validation')
            self.model.eval()
            with torch.no_grad():
                h_dict = self.input_feature()
                logits = self.model(self.hg, h_dict)[self.category]
                self.task.dataset.save_results(logits=logits, file_path=self.args.HGB_results_path)
            return
        if self.args.mini_batch_flag and hasattr(self, 'val_loader'):
            test_score, _ = self._mini_test_step(mode='test')
            val_score, val_loss = self._mini_test_step(mode='validation')
        else:
            test_score, _ = self._full_test_step(mode='test')
            val_score, val_loss = self._full_test_step(mode='validation')

        printMetric(self.metric, val_score, 'validation')
        printMetric(self.metric, test_score, 'test')
        return dict(Acc=test_score, ValAcc=val_score)

    def _full_train_step(self):

        h = self.input_feature()
        feat_src = h[self.ntypes[self.args.src_node_type]]
        # feat_src = self.hg.nodes[self.ntypes[self.args.src_node_type]].data['h']
        # attribute completion
        feat_src_re = self.hgnn_ac(self.adj[self.ntypes[self.args.src_node_type]][:, self.feat_keep_idx],
                                   self.hg.nodes[self.ntypes[self.args.src_node_type]
                                                 ].data['emb'],
                                   self.hg.nodes[self.ntypes[self.args.src_node_type]
                                                 ].data['emb'][self.feat_keep_idx],
                                   feat_src[self.feat_keep_idx]
                                   )
        loss_ac = F.mse_loss(
            feat_src[self.feat_drop_idx], feat_src_re[self.feat_drop_idx, :])

        with self.hg.local_scope():
            for i, opt in enumerate(list(self.args.feats_opt)):
                if opt == '1':
                    feat_ac = self.hgnn_ac(self.adj[self.ntypes[i]].t(),
                                           self.hg.nodes[self.ntypes[i]
                                                         ].data['emb'],
                                           self.hg.nodes[self.ntypes[self.args.src_node_type]
                                                         ].data['emb'],
                                           feat_src[self.hg.nodes(self.ntypes[self.args.src_node_type])])
                    h[self.ntypes[i]] = feat_ac
                    
            # Combination with HIN model, e.g. MAGNN
            self.model.train()
            logits = self.model(self.hg, h)[self.category]
        loss = self.loss_fn(logits[self.train_idx],
                            self.labels[self.train_idx])

        # L = lambda * L_completion + L_prediction
        loss = self.args.loss_lambda * loss_ac + loss
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()
        return loss.item()

    def _mini_train_step(self,):
        pass

    def _full_test_step(self, mode=None, logits=None):
        self.model.eval()
        with torch.no_grad():
            h_dict = self.input_feature()
            logits = logits if logits else self.model(self.hg, h_dict)[self.category]
            if mode == "train":
                mask = self.train_idx
            elif mode == "validation":
                mask = self.valid_idx
            elif mode == "test":
                mask = self.test_idx
            else:
                mask = None

            if mask is not None:
                loss = self.loss_fn(logits[mask], self.labels[mask]).item()
                if self.task.multi_label:
                    pred = (logits[mask].cpu().numpy()>0).astype(int)
                else:
                    pred = logits[mask].argmax(dim=1).to('cpu')
                metric = self.task.evaluate(pred, name=self.metric, mask=mask)

                return metric, loss
            else:
                masks = {'train': self.train_idx, 'val': self.valid_idx, 'test': self.test_idx}
                metrics = {key: self.task.evaluate((logits[mask].cpu().numpy()>0).astype(int) if self.task.multi_label
                                                   else logits[mask].argmax(dim=1).to('cpu'),
                                                   name=self.metric, mask=mask) for
                           key, mask in masks.items()}
                losses = {key: self.loss_fn(logits[mask], self.labels[mask]).item() for key, mask in masks.items()}
                return metrics, losses

    def _mini_test_step(self, mode):
        self.model.eval()
        with torch.no_grad():
            y_trues = []
            y_predicts = []
            loss_all = 0.0
            if mode == 'train':
                loader_tqdm = tqdm(self.train_loader, ncols=120)
            elif mode == 'validation':
                loader_tqdm = tqdm(self.val_loader, ncols=120)
            elif mode == 'test':
                loader_tqdm = tqdm(self.test_loader, ncols=120)
            for i, (input_nodes, seeds, blocks) in enumerate(loader_tqdm):
                blocks = [blk.to(self.device) for blk in blocks]
                seeds = seeds[self.category]
                lbl = self.labels[seeds].to(self.device)
                logits = self.model(blocks)[self.category]
                loss = self.loss_fn(logits, lbl)

                loss_all += loss.item()
                y_trues.append(lbl.detach().cpu())
                y_predicts.append(logits.detach().cpu())
            loss_all /= (i + 1)
            y_trues = torch.cat(y_trues, dim=0)
            y_predicts = torch.cat(y_predicts, dim=0)
        evaluator = self.task.get_evaluator(name='f1')
        metric = evaluator(y_trues,y_predicts.argmax(dim=1).to('cpu'))
        return metric, loss