import copy
from openhgnn import sampler

from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import BatchSampler
from torch.utils.data import TensorDataset
from openhgnn.trainerflow.base_flow import BaseFlow
import dgl
from networkx.algorithms.centrality.betweenness import edge_betweenness
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from tqdm import tqdm, trange
from sklearn.model_selection import train_test_split
import random
import os
import pickle
import json
import time
from typing import List
import shutil
import copy



from . import BaseFlow, register_flow
from ..tasks import build_task
from openhgnn.models import build_model
from openhgnn.models.SLiCE import SLiCE, SLiCEFinetuneLayer
from ..utils import extract_embed, EarlyStopping

from ..sampler.SLiCE_sampler import SLiCESampler
@register_flow("slicetrainer")
class SLiCETrainer(BaseFlow):
    def __init__(self,args):
        super(SLiCETrainer, self).__init__(args)
        self.args=args
        self.model_name=args.model
        self.device=args.device
        self.task=build_task(args)
        self.phase=['train','valid','test']
        self.g,_=dgl.load_graphs(self.task.dataset.data_path)
        self.g=dgl.to_homogeneous(self.g[0],ndata=['feature'],edata=['train_mask','valid_mask','test_mask','label'])
        self.model=dict()
        self.model['pretrain']=SLiCE.build_model_from_args(self.args,self.g)
        self.model['finetune']=SLiCEFinetuneLayer.build_model_from_args(args)
        self.evaluator=self.task.get_evaluator('slice')
        #loss function
        self.loss_fn=torch.nn.CrossEntropyLoss()
        #optimizer
        self.optimizer=dict()
        self.optimizer['pretrain']=optim.Adam(self.model['pretrain'].parameters(), args.lr)
        self.optimizer['finetune']=optim.Adam(self.model['finetune'].parameters(), lr=args.ft_lr)

        self.patience=5 #for early stopping
        self.n_epochs=dict()
        self.n_epochs['pretrain']=args.n_epochs
        self.n_epochs['finetune']=args.ft_n_epochs
        self.batch_size=dict()
        self.batch_size['pretrain']=args.batch_size
        self.batch_size['finetune']=args.ft_batch_size

        self.train_idx, self.val_idx, self.test_idx = self.task.get_idx()
        self.labels = self.task.get_labels().to(self.device)
        #pretrain
        self.node_subgraphs=dict()
        #finetune
        self.edges=dict()
        self.edges_label=dict()
        self.edge_subgraphs=dict()

        self.out_dir=args.outdir
        self.pretrain_path=os.path.join(self.out_dir,'pretrain/')
        self.pretrain_save_path=os.path.join(self.pretrain_path,'best_pretrain_model.pt')
        self.finetune_path=os.path.join(self.out_dir,'finetune/')
        self.finetune_save_path=os.path.join(self.finetune_path,'best_finetune_model.pt')
        self.graphs=dict()
    def preprocess(self):
        self.labels=self.g.edata['label']
        # self.g=dgl.to_homogeneous(
        #     self.g,
        #     ndata=['feature'],
        #     edata=['train_mask','valid_mask','test_mask','label'])
        self.graphs['train']=copy.deepcopy(self.g)
        if not os.path.exists(self.pretrain_path):
            os.makedirs(self.pretrain_path)
        if not os.path.exists(self.finetune_path):
            os.makedirs(self.finetune_path)
        for task in ['train','valid','test']:
            mask=self.g.edata[task+'_mask']
            index = torch.nonzero(mask.squeeze()).squeeze()
            if task=='train':
                self.train_idx=index
            elif task=='valid':
                self.valid_idx=index
            else:
                self.test_idx=index
            self.edges[task]=self.g.find_edges(index)
            #finally, g should be a graph containing just train_edges
            if task in ['valid','test']:
                #self.g.remove_edges(index)
                #built for valid and test phase(use apply_edge(u_mult_v()) to get similarity score)
                self.graphs[task]=dgl.graph(self.edges[task])
        #sample walks
        sampler=SLiCESampler(self.g,num_walks_per_node=self.args.n_pred,beam_width=self.args.beam_width,
                            max_num_edges=self.args.max_length,walk_type=self.args.walk_type,
                            path_option=self.args.path_option,save_path=self.out_dir)#full graph
        #get dataloader for pretrain and finetune evaluation on link prediction
        g=self.g
        #pretrain
        node_walk_path=self.pretrain_path+'node_walks.pickle'
        if os.path.exists(node_walk_path):
            with open(str(node_walk_path),'rb') as f:
                node_walks=pickle.load(f)
        else:
            node_walks=sampler.get_node_subgraph(g.nodes())
            with open(node_walk_path,'wb') as f:
                pickle.dump(node_walks,f)
        random.shuffle(node_walks)
        total_len=len(node_walks)
        train_size=int(0.8*total_len)
        valid_size=int(0.1*total_len)
        self.node_subgraphs['train']=node_walks[:train_size]
        self.node_subgraphs['valid']=node_walks[train_size:train_size+valid_size]
        self.node_subgraphs['test']=node_walks[train_size+valid_size:]
        #finetune
        src,dst=g.find_edges(self.train_idx)
        self.edges['train']=list(zip(src.tolist(),dst.tolist()))
        src,dst=g.find_edges(self.valid_idx)
        self.edges['valid']=list(zip(src.tolist(),dst.tolist()))
        src,dst=g.find_edges(self.test_idx)
        self.edges['test']=list(zip(src.tolist(),dst.tolist()))
        edges_label=self.edges_label
        edges_label['train']=list()
        edges_label['valid']=self.labels[self.valid_idx]
        edges_label['test']=self.labels[self.test_idx]
        #generate pretrain subgraph
        train_file=os.path.join(self.finetune_path,'train_edges.pickle')
        if os.path.exists(train_file):
            with open(train_file,'rb') as f:
                self.edges['train'],edges_label['train']=pickle.load(f)
        else:
            self.edges['train'],edges_label['train']=sampler.generate_false_edges2(self.edges['train'],train_file)
        #generate finetune subgraph
        finetune_input=self.finetune_path+'finetune_input.pickle'
        if os.path.exists(finetune_input):
            with open(finetune_input,'rb') as f:
                self.edge_subgraphs=pickle.load(f)
        else:
            for task in ['train','valid','test']:
                self.edge_subgraphs[task]=sampler.get_edge_subgraph(self.edges[task])
            with open(finetune_input,'wb') as f:
                pickle.dump(self.edge_subgraphs,f)

    def train(self):
        self.preprocess()
        metrics_pretrain=self.pretrain()
        metrics_finetune=self.finetune()
    def pretrain(self):
        print("Start Pretraining...")
        stopper=EarlyStopping(self.patience)
        epoch_iter = tqdm(range(self.n_epochs['pretrain']))
        batch_size=self.batch_size['pretrain']
        self.model['pretrain'].train()
        for epoch in epoch_iter:
            print("Epoch {}:".format(epoch))
            i=0
            total_len=len(self.node_subgraphs['train'])
            n_batch=int(total_len/batch_size)
            bar=tqdm(range(n_batch))
            for batch in bar:
                i=batch*batch_size
                if i+batch_size<total_len:
                    subgraph_list=self.node_subgraphs['train'][i:i+batch_size]
                else:
                    subgraph_list=self.node_subgraphs['train'][i:]
                pred_data,true_data=self.model['pretrain'](subgraph_list)
                loss=self.loss_fn(pred_data.transpose(1,2).cuda(),true_data.cuda())
                self.optimizer['pretrain'].zero_grad()
                loss.backward()
                self.optimizer['pretrain'].step()
                i+=batch_size
                bar.set_description("Batch {} Loss: {:.3f}".format(batch,loss))
            torch.save(self.model['pretrain'],self.pretrain_path+'model_'+str(epoch)+'SLiCE.pt')
            early_stop=stopper.loss_step(loss,self.model['pretrain'])
        print("Evaluating for pretraining...")
        val_f1,val_auc=self._test_step(split='val')
        test_f1,test_auc=self._test_step(split='test')
    def finetune(self):
        if not os.path.exists(self.pretrain_save_path):
            print("Model not pretrained!")
        else:
            ck_pt=torch.load(self.pretrain_save_path)
        self.model['pretrain'].load_state_dict(ck_pt)
        self.model['pretrain'].eval()
        self.model['finetune'].train()
        print("Start Finetuning...")
        stopper=EarlyStopping(self.patience)
        epoch_iter = tqdm(range(self.n_epochs['finetune']))
        batch_size=self.batch_size['finetune']
        for epoch in epoch_iter:
            batch=0
            total_len=len(self.node_subgraphs['train'])
            start_time=time.time()
            print("Eopch {}:".format(epoch))
            while batch*batch_size<total_len:
                i=batch*batch_size
                if i+batch_size<total_len:
                    subgraph_list=self.edge_subgraphs['train'][i:i+batch_size]
                else:
                    subgraph_list=self.edge_subgraphs['train'][i:]
                self.model['pretrain'].set_finetune_layer()

                with torch.no_grad():
                    _,layer_output,_=self.model['pretrain'](subgraph_list)
                pred_scores,_,_=self.model['finetune'](layer_output)
                loss=self.loss_fn(pred_scores,self.labels['finetune']['train'])
                epoch_iter.set_description('Epoch{}: Loss:{:.4f}'.format(batch,loss))
                batch+=1
            torch.save(self.model['finetune'],self.finetune_path+'model_'+str(epoch)+'SLiCE.pt')
            
            early_stop=stopper.loss_step(loss,self.model['finetune'])
            end_time=time.time()
            print("Epoch time: {}(s)".format(end_time-start_time))
        # self.model['finetune']=stopper.best_model
        # torch.save(self.model,self.finetune_save_path)
        #run validation to find the best epoch
        print("Evaluating for pretraining...")
        val_f1,val_auc=self._test_step(split='val')
        test_f1,test_auc=self._test_step(split='test')
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
            labels = edge_subgraph.edata['label']
            return score.squeeze(),labels.squeeze()