import copy
from openhgnn.trainerflow.base_flow import BaseFlow
import dgl
from networkx.algorithms.centrality.betweenness import edge_betweenness
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm, trange
import random
import os
import pickle
import json
import time
from typing import List
import shutil
import copy

import torch.nn.functional as F
from . import BaseFlow, register_flow
from ..tasks import build_task
from openhgnn.models import build_model
from openhgnn.models.SLiCE import SLiCE, FinetuneLayer
from ..utils import extract_embed, EarlyStopping

@register_flow("slicetrainer")
class SLiCETrainer(BaseFlow):
    def __init__(self,args):
        super(SLiCETrainer, self).__init__(args)
        self.args=args
        self.model_name=args.model
        self.device=args.device
        self.task=build_task(args)
        self.phase=['train','valid','test']
        self.g=self.task.get_graph().to(self.device)
        self.model=dict()
        self.model['pretrain']=SLiCE.build_model_from_args(self.args,self.g)
        self.model['finetune']=FinetuneLayer.build_model_from_args(args)
        self.evaluator=self.task.get_evaluator('slice')
        #loss function
        self.loss_fn=self.task.get_loss_fn()
        #optimizer
        self.optimizer=dict()
        self.optimizer['pretrain']=optim.Adam(self.model['pretrain'].parameters(), args.lr)
        self.optimizer['finetune']=optim.Adam(self.model['finetune'].parameters(), lr=args.ft_lr)

        self.patience=5 #for early stopping
        self.n_epochs=dict()
        self.n_epochs['pretrain']=args.n_epochs
        self.n_epochs['finetune']=args.ft_n_epochs

        self.train_idx, self.val_idx, self.test_idx = self.task.get_idx()
        self.labels = self.task.get_labels().to(self.device)

        self.out_dir=args.out_dir
        self.pretrain_path=self.out_dir+'pretrain/'
        self.pretrain_save_path=self.pretrain_path+'best_pretrain_model.pt'
        self.finetune_path=self.out_dir+'finetune/'
        self.finetune_save_path=self.finetune_path+'best_finetune_model.pt'
    def preprocess(self):
        N_SAMPLE_NODES_PER_TYPE = 1280  # number of nodes to sample per node type per sampler step
        N_SAMPLE_STEPS = 6  # number of sampler steps
        #get dataloader for pretrain and finetune evaluation on link prediction


    def train(self):
        self.preprocess()
        metrics_pretrain=self.pretrain()
        metrics_finetune=self.finetune()
    def pretrain(self):
        print("Start Pretraining...")
        stopper=EarlyStopping(self.patience)
        epoch_iter = tqdm(range(self.n_epochs['pretrain']))
        
        for epoch in epoch_iter:
            loss=self._mini_pretrain_step()
            torch.save(self.model['pretrain'],self.pretrain_path+'model_'+str(epoch)+'SLiCE.pt')
            
            epoch_iter.set_description('Epoch{}: Loss:{:.4f}'.format(epoch,loss))
            early_stop=stopper.loss_step(loss,self.model['pretrain'])
        #用early stopping还是valid？
        # self.model['pretrain']=stopper.best_model
        # torch.save(self.model,self.pretrain_save_path)
        #run validation to find the best epoch
        print("Evaluating for pretraining...")
        val_f1,val_auc=self._test_step(split='val')
        test_f1,test_auc=self._test_step(split='test')
    def finetune(self):
        if not os.path.exists(self.pretrain_save_path):
            print("Model not pretrained!")
        else:
            ck_pt=torch.load(self.pretrain_save_path)

        self.model['pretrain'].load_state_dict(ck_pt)
        
        print("Start Finetuning...")
        stopper=EarlyStopping(self.patience)
        epoch_iter = tqdm(range(self.n_epochs['finetune']))
        for epoch in epoch_iter:
            loss=self._mini_finetune_step()
            torch.save(self.model['finetune'],self.finetune_path+'model_'+str(epoch)+'SLiCE.pt')
            epoch_iter.set_description('Epoch{}: Loss:{:.4f}'.format(epoch,loss))
            early_stop=stopper.loss_step(loss,self.model['finetune'])
        # self.model['finetune']=stopper.best_model
        # torch.save(self.model,self.finetune_save_path)
        #run validation to find the best epoch
        print("Evaluating for pretraining...")
        val_f1,val_auc=self._test_step(split='val')
        test_f1,test_auc=self._test_step(split='test')
    def _mini_pretrain_step(self):
        self.model['pretrain'].train()
        loss_all = 0
        for i, (sg, seed_nodes) in tqdm(enumerate(self.dataloader['pretrain'])):
            sg = sg.to(self.device)
            #h = sg.ndata.pop('h')
            #logits = self.model(sg_list)[self.category]
            #labels = self.labels[sg.ndata[dgl.NID][self.category]].squeeze()
            loss = self.loss_fn(logits, labels)
            loss_all += loss.item()
            self.optimizer['pretrain'].zero_grad()
            loss.backward()
            self.optimizer['pretrain'].step()
        return loss_all
    def _mini_finetune_step(self):
        self.model['pretrain'].eval()
        self.model['finetune'].train()
        loss_all=0
        for i, (sg, seed_nodes) in tqdm(enumerate(self.dataloader['pretrain'])):
            sg = sg.to(self.device)
            with torch.no_grad():
                #get the subgraph from dataloader
                #get the embedding from SLiCE
                _,layer_output,_=self.model['pretrain']()
                #train FinetuneLayer
            #logits=
            #labels=
            loss = self.loss_fn(logits, labels)
            loss_all += loss.item()
            self.optimizer['finetune'].zero_grad()
            loss.backward()
            self.optimizer['finetune'].step()
        return loss_all
    def _test_step(self,split=None, logits=None):
        with torch.no_grad():
            if logits==None:
                pass#get loss by forward propagation
            print("Test for {} set...".format(split))
            if split == "train":
                mask = self.train_idx
            elif split == "val":
                mask = self.val_idx
            elif split == "test":
                mask = self.test_idx
            else:
                mask = None
            if mask is not None:
                loss = self.loss_fn(logits[mask], self.labels[mask]).item()
                metric = self.evaluator(self.labels[mask].to('cpu'), logits[mask].argmax(dim=1).to('cpu'))
                return metric, loss
            else:
                masks = {'train': self.train_idx, 'val': self.val_idx, 'test': self.test_idx}
                metrics = {key: self.evaluator(self.labels[mask].to('cpu'), logits[mask].argmax(dim=1).to('cpu')) for key, mask in masks.items()}
                losses = {key: self.loss_fn(logits[mask], self.labels[mask]).item() for key, mask in masks.items()}
                return metrics, losses
    def loss_calculation(self, pos_score, neg_score):
        # an example hinge loss
        loss = []
        for i in pos_score:
            loss.append(F.logsigmoid(pos_score[i]))
            loss.append(F.logsigmoid(-neg_score[i]))
        loss = torch.cat(loss)
        return -loss.mean()

    def ScorePredictor(self, edge_subgraph, x):
        with edge_subgraph.local_scope():
            for ntype in edge_subgraph.ntypes:
                edge_subgraph.nodes[ntype].data['x'] = x[ntype]
            edge_subgraph.apply_edges(
                dgl.function.u_dot_v('x', 'x', 'score'))
            score = edge_subgraph.edata['score']
            return score.squeeze()