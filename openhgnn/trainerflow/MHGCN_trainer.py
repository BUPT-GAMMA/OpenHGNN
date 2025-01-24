from collections import OrderedDict
import os
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from ..models import build_model
from . import BaseFlow, register_flow
from ..tasks import build_task
from ..dataset import build_dataset
from sklearn.metrics import f1_score
from torch.utils.tensorboard import SummaryWriter
# from ..models.comp_model import MHGCN
import time
import dgl
from sklearn.metrics import precision_recall_curve, roc_auc_score, f1_score, auc

class LogReg(nn.Module):
    """
    Logical classifier
    """
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret

@register_flow("MHGCN_NC_Trainer")
class MHGCN_NC_Trainer(BaseFlow):

    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.model_name = args.model
        self.device = args.device
        self.dataset = self.task.dataset
        args.feature_dim = self.dataset.features.shape[1]
        args.etype_num = self.dataset.etype_num
        self.dataset.g = self.dataset.g.to(self.device)
        self.model = build_model(self.model_name).build_model_from_args(
            self.args,self.dataset.g)
        self.model = self.model.to(self.device)
        self.patience = args.patience
        self.max_epoch = args.max_epoch
        self.args = args
        self.loss_fn = nn.CrossEntropyLoss()
        self.writer = SummaryWriter(f'./openhgnn/output/{self.model_name}/')


    def get_graph(self):
        return self.dataset.g
    def train_step(self):
        self.model.train()
        self.log.train()
        embeds = self.model(self.dataset.g, self.features)
        train_embs = embeds[self.idx_train]
        self.optimizer.zero_grad()
        logits = self.log(train_embs)
        train_loss = self.loss_fn(logits,self.labels[self.idx_train])
        train_loss.backward()
        self.optimizer.step()
        return train_loss
    def test_step(self):
        self.model.eval()
        self.log.eval()
        embeds = self.model(self.dataset.g, self.features)
        logits = self.log(embeds)
        masks = {'train': self.idx_train, 'valid': self.idx_val, 'test': self.idx_test}
        metric_dict = {key:self.task.evaluate(logits,mode=key) for key in masks}
        loss_dict = {key:self.loss_fn(logits[mask],self.labels[mask]).item() for key ,mask in masks.items()}
        return metric_dict,loss_dict

    def train(self):
        """Node classification training process"""
        start_time = time.time()
        self.features = torch.tensor(self.dataset.features,dtype=torch.float32,device=self.device)
        embeds = self.model(self.dataset.g, self.features)
        idx_train = self.dataset.train_idx
        idx_val = self.dataset.val_idx
        idx_test = self.dataset.test_idx
        self.idx_train = torch.LongTensor(idx_train).to(self.device)
        self.idx_val = torch.LongTensor(idx_val).to(self.device)
        self.idx_test = torch.LongTensor(idx_test).to(self.device)
        self.labels = self.dataset.get_labels().to(self.device)
        hid_units = embeds.shape[1]
        nb_classes = self.dataset.labels.shape[1]
        self.log = LogReg(hid_units, nb_classes)
        self.log.to(self.device)
        self.optimizer = torch.optim.Adam([{'params': self.model.parameters(), 'lr': self.args.model_lr}, {'params': self.log.parameters()}], lr=self.args.lr, weight_decay=self.args.weight_decay)
        epoch_iter = tqdm(range(self.max_epoch))
        for iter_ in epoch_iter:
            train_loss = self.train_step()
            metric_dict,loss_dict = self.test_step()
            val_loss = loss_dict['valid']
            current_time = time.time()
            self.logger.train_info(f"Epoch: {iter_}, Train loss: {train_loss:.4f}, Valid loss: {val_loss:.4f}. "
                        + self.logger.metric2str(metric_dict) + f"time: {current_time - start_time:.10f} .")
            self.writer.add_scalars('loss', {'train': train_loss, 'valid': val_loss}, global_step=iter_)
            for mode in ['train','valid','test']:
                    self.writer.add_scalars(f'metric_{mode}', metric_dict[mode], global_step=iter_)
        self.writer.close()

@register_flow("MHGCN_LP_Trainer")
class MHGCN_LP_Trainer(BaseFlow):

    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.model_name = args.model
        self.device = args.device
        self.dataset = self.task.dataset
        args.feature_dim = self.dataset.features.shape[1]
        args.etype_num = self.dataset.etype_num
        self.dataset.g = self.dataset.g.to(self.device)
        self.dataset.train_g = self.dataset.train_g.to(self.device)
        self.dataset.val_g = self.dataset.val_g.to(self.device)
        self.dataset.test_g = self.dataset.test_g.to(self.device)
        self.dataset.val_neg_g = self.dataset.val_neg_g.to(self.device)
        self.dataset.test_neg_g = self.dataset.test_neg_g.to(self.device)
        self.model = build_model(self.model_name).build_model_from_args(
            self.args,self.dataset.g)
        self.model = self.model.to(self.device)
        self.args = args
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        self.writer = SummaryWriter(f'./openhgnn/output/{self.model_name}/')

    def get_score(self,local_model, node1, node2):
        """
        Calculate embedding similarity
        """
        try:
            vector1 = local_model[node1]
            vector2 = local_model[node2]
            if type(vector1) != np.ndarray:
                vector1 = vector1.toarray()[0]
                vector2 = vector2.toarray()[0]
            return np.dot(vector1, vector2)
        except Exception as e:
            pass

    def embeding_process(self,embeds,g):
        g.ndata['embeds'] = embeds
        g.apply_edges(dgl.function.u_dot_v('embeds', 'embeds', 'score'))
        return g.edata['score']

    def get_graph(self):
        return self.dataset.g

    def train_step(self):
        self.model.train()
        emb = self.model(self.dataset.g, self.features)
        pos_score = self.embeding_process(emb,self.dataset.val_g).squeeze(-1)
        neg_score = -self.embeding_process(emb,self.dataset.val_neg_g).squeeze(-1)
        loss = -torch.mean(F.logsigmoid(pos_score) + F.logsigmoid(neg_score))
        loss = loss.requires_grad_()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def test_step(self):
        self.model.eval()
        emb = self.model(self.dataset.g, self.features)
        pos_valid_score = self.embeding_process(emb,self.dataset.val_g).squeeze(-1)
        neg_valid_score = self.embeding_process(emb,self.dataset.val_neg_g).squeeze(-1)
        pos_test_score = self.embeding_process(emb,self.dataset.test_g).squeeze(-1)
        neg_test_score = self.embeding_process(emb,self.dataset.test_neg_g).squeeze(-1)
        return {"valid":self.evaluate(pos_valid_score,neg_valid_score),"test":self.evaluate(pos_test_score,neg_test_score)}

    def train(self):
        """
        Link prediction training proces
        """
        self.model.to(self.device)
        self.features = torch.tensor(self.dataset.features,dtype=torch.float32,device=self.device)
        epoch_iter = tqdm(range(self.max_epoch))
        start_time=time.time()
        for epoch in epoch_iter:
            loss = self.train_step()
            test_metric = self.test_step()
            current_time = time.time()
            self.logger.train_info(
                        f"Epoch: {epoch:03d}, train loss: {loss:.4f}. " + self.logger.metric2str(test_metric) + f"time: {current_time - start_time:.10f} .")
            self.writer.add_scalar('train_loss', loss, global_step=epoch)
            self.writer.add_scalars('valid_metric', test_metric['valid'], global_step=epoch)
            self.writer.add_scalars('test_metric', test_metric['test'], global_step=epoch)
        self.writer.close()

    def evaluate(self,pos_scores,neg_scores):
        true_num = len(pos_scores)
        true_list = np.concatenate([np.ones(pos_scores.shape[0]),np.zeros(neg_scores.shape[0])])
        prediction_list = torch.cat([pos_scores,neg_scores])
        sorted_pred = prediction_list[:].detach().cpu().numpy()
        sorted_pred.sort()
        threshold = sorted_pred[-true_num]
        y_pred = np.array([1 if x >= threshold else 0 for x in prediction_list.detach().cpu().numpy()])
        y_true = np.array(true_list)
        y_scores = np.array(prediction_list.detach().cpu())
        ps, rs, _ = precision_recall_curve(y_true, y_scores)
        return dict(r_auc=roc_auc_score(y_true, y_scores), pr_auc=auc(rs, ps), f1=f1_score(y_true, y_pred)) 