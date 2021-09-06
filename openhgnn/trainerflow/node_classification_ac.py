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
from ..tasks import build_task
from ..utils import EarlyStopping


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
        self.args = args
        self.model_name = args.model[0:-3]
        print(self.model_name)
        self.device = args.device

        self.task = build_task(args)

        self.hg = self.task.get_graph().to(self.device)

        self.num_classes = self.task.dataset.num_classes
        if hasattr(self.task.dataset, 'in_dim'):
            self.args.in_dim = self.task.dataset.in_dim
        if not hasattr(self.task.dataset, 'out_dim') or args.out_dim != self.num_classes:
            print('Modify the out_dim with num_classes')
            args.out_dim = self.num_classes

        self.args.category = self.task.dataset.category
        self.category = self.args.category
        self.model = build_model(
            self.model_name).build_model_from_args(self.args, self.hg)
        self.hgnn_ac = build_model(
            "HGNN_AC").build_model_from_args(self.args, self.hg)
        self.optimizer = torch.optim.Adam([{'params': self.model.parameters()},
                                        {'params': self.hgnn_ac.parameters()}],
                                        lr = args.lr, weight_decay = args.weight_decay)

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
        self.loss_fn = self.task.get_loss_fn()
        
        self.patience = args.patience
        self.max_epoch = args.max_epoch

        self.train_idx, self.val_idx, self.test_idx = self.task.get_idx()
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
        self.ntypes = list(self.model.in_feats.keys())
        self.in_dim = []
        self.adj = {}
        for ntype in self.ntypes:
            self.in_dim.append(self.hg.nodes[ntype].data['feat'].shape[0])
        # feat_dict = self.hg.ndata['feat']
        # for ntype in self.input_projection.keys():
        #     self.hg.nodes[ntype].data['feat'] = self.feat_drop(self.input_projection[ntype](feat_dict[ntype]))
        self.feat_keep_idx, self.feat_drop_idx = train_test_split(np.arange(self.in_dim[self.args.src_node_type]),
                                                        test_size = self.args.feats_drop_rate)
        src = self.ntypes[self.args.src_node_type]
        
        for ntype in self.ntypes:
            dst = ntype
            edge = src + '-' + dst
            matrix = torch.zeros((self.hg.num_nodes(ntype = src), self.hg.num_nodes(ntype = dst)))
            if edge in self.model.edge_type_list:
                for i, j in enumerate(self.hg.edges(etype = edge)[0]):
                    i = self.hg.edges(etype = edge)[1][i]
                    matrix[j][i] = 1
            matrix = matrix.to(self.device)
            self.adj[dst] = matrix
        return

    def train(self):
        self.preprocess()
        stopper = EarlyStopping(self.args.patience, self._checkpoint)
        epoch_iter = tqdm(range(self.max_epoch))
        for epoch in epoch_iter:
            if self.args.mini_batch_flag:
                loss = self._mini_train_step()
            else:
                loss = self._full_train_step()
            f1, losses = self._test_step()

            train_f1 = f1["train"]
            val_f1 = f1["val"]
            test_f1 = f1['test']
            val_loss = losses["val"]
            # epoch_iter.set_description(
            #     f"Epoch: {epoch:03d}, Train_macro_f1: {train_f1[0]:.4f}, Train_micro_f1: {train_f1[1]:.4f}, Val_macro_f1: {val_f1[0]:.4f}, Val_micro_f1: {val_f1[1]:.4f}, ValLoss:{val_loss: .4f}"
            # )
            print((
                f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Train_macro_f1: {train_f1[0]:.4f}, Train_micro_f1: {train_f1[1]:.4f}, "
                f"Val_macro_f1: {val_f1[0]:.4f}, Test_macro_f1: {test_f1[0]:.4f}, ValLoss:{val_loss: .4f}"
            ))
            early_stop = stopper.step(val_loss, val_f1[0], self.model)
            if early_stop:
                print('Early Stop!\tEpoch:' + str(epoch))
                break

        print(
            f"Valid_micro_f1 = {stopper.best_score: .4f}, Min_loss = {stopper.best_loss: .4f}")
        stopper.load_model(self.model)
        test_f1, _ = self._test_step(split="test")
        val_f1, _ = self._test_step(split="val")
        print(
            f"Test_macro_f1 = {test_f1[0]:.4f}, Test_micro_f1: {test_f1[1]:.4f}")
        
        return dict(Acc=test_f1, ValAcc=val_f1)

    def _full_train_step(self):
        
        feat_src = self.hg.nodes[self.ntypes[self.args.src_node_type]].data['feat']
        #attribute completion
        feat_src_re = self.hgnn_ac(self.adj[self.ntypes[self.args.src_node_type]][:, self.feat_keep_idx], 
                                   self.hg.nodes[self.ntypes[self.args.src_node_type]].data['emb'],
                                   self.hg.nodes[self.ntypes[self.args.src_node_type]].data['emb'][self.feat_keep_idx],
                                   feat_src[self.feat_keep_idx]
                                   )
        loss_ac = F.mse_loss(feat_src[self.feat_drop_idx], feat_src_re[self.feat_drop_idx, :])
        
        
        with self.hg.local_scope():
            
            for i, opt in enumerate(list(self.args.feats_opt)):
                if opt == '1':
                    feat_ac = self.hgnn_ac(self.adj[self.ntypes[i]].t(),
                                        self.hg.nodes[self.ntypes[i]].data['emb'],
                                        self.hg.nodes[self.ntypes[self.args.src_node_type]].data['emb'],
                                        feat_src[self.hg.nodes(self.ntypes[self.args.src_node_type])])
                    self.hg.nodes[self.ntypes[i]].data['feat'] = feat_ac
            
            #Combination with HIN model, e.g. MAGNN
            self.model.train()
            logits = self.model(self.hg)[self.category]
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

    def _test_step(self, split=None, logits=None):
        self.model.eval()
        with torch.no_grad():
            logits = logits if logits else self.model(self.hg)[self.category]
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
                metric = self.task.evaluate(logits[mask].argmax(
                    dim=1).to('cpu'), name='f1', mask=mask)
                return metric, loss
            else:
                masks = {'train': self.train_idx,
                         'val': self.val_idx, 'test': self.test_idx}
                metrics = {key: self.task.evaluate(logits[mask].argmax(dim=1).to(
                    'cpu'), name='f1', mask=mask) for key, mask in masks.items()}
                losses = {key: self.loss_fn(
                    logits[mask], self.labels[mask]) for key, mask in masks.items()}
                return metrics, losses
