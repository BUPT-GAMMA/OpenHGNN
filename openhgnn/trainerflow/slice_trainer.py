import copy

from dgl._ffi.base import DGLError
from openhgnn import sampler

from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import BatchSampler
from torch.utils.data import TensorDataset
from openhgnn.trainerflow.base_flow import BaseFlow
import dgl
from networkx.algorithms.centrality.betweenness import edge_betweenness_centrality
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
import pandas as pd
import math
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    mean_squared_error,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

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
        self.out_dir=args.outdir
        self.pretrain_path=os.path.join(self.out_dir,'pretrain/')
        self.pretrain_save_path=os.path.join(self.pretrain_path,'best_pretrain_model.pt')
        self.finetune_path=os.path.join(self.out_dir,'finetune/')
        self.finetune_save_path=os.path.join(self.finetune_path,'best_finetune_model.pt')
        self.g=self.task.dataset.g
        self.g=dgl.to_homogeneous(self.g,edata=['train_mask','valid_mask','test_mask','label'])

        self.model=dict()
        self.model['pretrain']=SLiCE.build_model_from_args(self.args,self.g)
        self.model['finetune']=SLiCEFinetuneLayer.build_model_from_args(args)
        #loss function
        self.loss_fn=torch.nn.CrossEntropyLoss()
        #optimizer
        self.optimizer=dict()
        self.optimizer['pretrain']=optim.Adam(self.model['pretrain'].parameters(), lr=args.lr)
        self.optimizer['finetune']=optim.Adam(self.model['finetune'].parameters(), lr=args.ft_lr)

        self.patience=5 #for early stopping
        #number of epochs
        self.n_epochs=dict()
        self.n_epochs['pretrain']=args.n_epochs
        self.n_epochs['finetune']=args.ft_n_epochs
        #batch size
        self.batch_size=dict()
        self.batch_size['pretrain']=args.batch_size
        self.batch_size['finetune']=args.ft_batch_size

        self.labels = self.g.edata['label']
        self.idx=dict()
        #pretrain
        self.node_subgraphs=dict()
        #finetune
        self.edges=dict()
        self.edges_label=dict()

        self.graphs=dict()
        self.best_epoch=dict()
        self.is_pretrained=False
        self.is_finetuned=False
        self.threshold=None
    def preprocess(self):
        if not os.path.exists(self.pretrain_path):
            os.makedirs(self.pretrain_path)
        if not os.path.exists(self.finetune_path):
            os.makedirs(self.finetune_path)
        for task in ['train','valid','test']:
            #make directories
            if not os.path.exists(os.path.join(self.finetune_path,task)):
                os.makedirs(os.path.join(self.finetune_path,task))
            mask=self.g.edata[task+'_mask']
            index = torch.nonzero(mask.squeeze()).squeeze()
            if task=='train':
                self.idx['train']=index
            elif task=='valid':
                self.idx['valid']=index
            else:
                self.idx['test']=index
            self.edges[task]=self.g.find_edges(index)
            #finally, g should be a graph containing just train_edges
            self.graphs[task]=dgl.edge_subgraph(self.g,index)
        #sample walks
        self.sampler=SLiCESampler(self.g,self.graphs['train'],num_walks_per_node=self.args.n_pred,beam_width=self.args.beam_width,
                            max_num_edges=self.args.max_length,walk_type=self.args.walk_type,
                            path_option=self.args.path_option,save_path=self.out_dir)#full graph
        sampler=self.sampler
        #get dataloader for pretrain and finetune evaluation on link prediction
        g=self.g
        #pretrain
        node_walk_path=self.pretrain_path+'node_walks.bin'
        if os.path.exists(node_walk_path):
            node_walks,_=dgl.load_graphs(node_walk_path)
        else:
            node_walks=sampler.get_node_subgraph(g.nodes())
            dgl.save_graphs(node_walk_path,node_walks)
        random.shuffle(node_walks)
        total_len=len(node_walks)
        train_size=int(0.8*total_len)
        valid_size=int(0.1*total_len)
        self.node_subgraphs['train']=node_walks[:train_size]
        self.node_subgraphs['valid']=node_walks[train_size:train_size+valid_size]
        self.node_subgraphs['test']=node_walks[train_size+valid_size:]
        #finetune
        src,dst=g.find_edges(self.idx['train'])
        self.edges['train']=list(zip(src.tolist(),dst.tolist()))
        src,dst=g.find_edges(self.idx['valid'])
        self.edges['valid']=list(zip(src.tolist(),dst.tolist()))
        src,dst=g.find_edges(self.idx['test'])
        self.edges['test']=list(zip(src.tolist(),dst.tolist()))
        self.edges_label['train']=list()
        self.edges_label['valid']=[int(self.labels[x]) for x in self.idx['valid']]
        self.edges_label['test']=[int(self.labels[x]) for x in self.idx['test']]
        edges_label=self.edges_label
        #generate pretrain subgraph
        train_file=os.path.join(self.finetune_path,'train_edges.pickle')
        if os.path.exists(train_file):
            with open(train_file,'rb') as f:
                self.edges['train'],edges_label['train']=pickle.load(f)
        else:
            self.edges['train'],edges_label['train']=sampler.generate_false_edges2(self.edges['train'],train_file)
        self.edges['valid'],self.edges_label['valid']=sampler.shuffle_edge_label(self.edges['valid'],self.edges_label['valid'])
        self.edges['test'],self.edges_label['test']=sampler.shuffle_edge_label(self.edges['test'],self.edges_label['test'])
        #generate finetune subgraph
        for task in ['train','valid','test']:
            edges=self.edges[task]
            batch_size=self.batch_size['finetune']
            n_batch=int(len(edges)/batch_size)
            total_len=len(edges)
            for batch in range(n_batch):
                i=batch*batch_size
                if i+batch_size<total_len:
                    end=i+batch_size
                else:
                    end=total_len
                batch_file=os.path.join(self.finetune_path,'{}/edge_subgraph_{}.bin'.format(task,batch))
                #pair_file=self.finetune_path+'{}/pair_subgraph_{}.pickle'.format(task,batch)
                if not os.path.exists(batch_file):
                    subgraph_list=self.sampler.get_edge_subgraph(self.edges[task][i:end])
                    dgl.save_graphs(batch_file,subgraph_list)

    def train(self):
        self.preprocess()
        self.pretrain()
        self.finetune()
    def pretrain(self):
        print("Start Pretraining...")
        stopper=EarlyStopping(self.patience)
        batch_size=self.batch_size['pretrain']
        self.model['pretrain'].train()
        self.is_pretrained=True
        if os.path.exists(self.pretrain_save_path):
            pass
        for epoch in range(self.n_epochs['pretrain']):
            print("Epoch {}:".format(epoch))
            i=0
            total_len=len(self.node_subgraphs['train'])
            n_batch=math.ceil(total_len/batch_size)
            bar=tqdm(range(n_batch))
            avg_loss=0
            for batch in bar:
                i=batch*batch_size
                if i+batch_size<total_len:
                    subgraph_list=self.node_subgraphs['train'][i:i+batch_size]
                else:
                    subgraph_list=self.node_subgraphs['train'][i:]
                pred_data,true_data=self.model['pretrain'](subgraph_list)
                loss=self.loss_fn(pred_data.transpose(1,2),true_data)
                avg_loss+=float(loss)
                self.optimizer['pretrain'].zero_grad()
                loss.backward()
                self.optimizer['pretrain'].step()
                i+=batch_size
                bar.set_description("Batch {} Loss: {:.3f}".format(batch,loss))
            #torch.save(self.model['pretrain'],self.pretrain_path+'model_'+str(ii)+'SLiCE.pt')
            avg_loss=avg_loss/n_batch
            print("AvgLoss: {:.3f}".format(avg_loss))
            early_stop=stopper.loss_step(avg_loss,self.model['pretrain'])
            if early_stop:
                print('Early Stop!\tEpoch:' + str(epoch))
                break
        
        self.best_epoch['pretrain']=epoch
        torch.save(self.model['pretrain'].state_dict(),self.pretrain_save_path)
        print("Evaluating for pretraining...")
        
    def finetune(self):
        if not os.path.exists(self.pretrain_save_path):
            print("Model not pretrained!")
        else:
            ck_pt=torch.load(self.pretrain_save_path)
        self.model['pretrain'].load_state_dict(ck_pt)
        self.model['pretrain'].eval()
        self.model['pretrain'].set_fine_tuning()
        self.model['finetune'].train()
        print("Start Finetuning...")
        stopper=EarlyStopping(self.patience)
        batch_size=self.batch_size['finetune']
        for epoch in range(self.n_epochs['finetune']):
            batch=0
            total_len=len(self.edges['train'])
            print("Eopch {}:".format(epoch))
            n_batch=math.ceil(total_len/batch_size)
            bar=tqdm(range(n_batch))
            avg_loss=0
            for batch in bar:
                i=batch*batch_size
                if i+batch_size<total_len:
                    end=i+batch_size
                else:
                    end=total_len
                batch_file=os.path.join(self.finetune_path,'{}/edge_subgraph_{}.bin'.format('train',batch))
                
                if os.path.exists(batch_file):
                    subgraph_list,_=dgl.load_graphs(batch_file)
                else:
                    subgraph_list=self.sampler.get_edge_subgraph(self.edges['train'][i:end])
                    dgl.save_graphs(batch_file,subgraph_list)
                self.model['pretrain'].set_fine_tuning()

                with torch.no_grad():
                    _,layer_output,_=self.model['pretrain'](subgraph_list)
                pred_scores,_,_=self.model['finetune'](layer_output)
                loss=F.binary_cross_entropy(pred_scores,torch.tensor(self.edges_label['train'][i:end],dtype=torch.float).reshape(-1,1))
                bar.set_description('Batch {}: Loss:{:.3f}'.format(batch,loss))
                avg_loss+=float(loss)
                self.optimizer['finetune'].zero_grad()
                loss.backward()
                self.optimizer['finetune'].step()
            torch.save(self.model['finetune'].state_dict(),self.finetune_path+'model_'+str(epoch)+'SLiCE.pt')
            avg_loss=avg_loss/n_batch
            print("AvgLoss: {:.3f}".format(avg_loss))
            early_stop=stopper.loss_step(avg_loss,self.model['finetune'])
            if early_stop:
                print('Early Stop!\tEpoch:' + str(epoch))
                break
        self.model['finetune']=stopper.best_model
        torch.save(self.model,self.finetune_save_path)
        #run validation to find the best epoch
        self.is_finetuned=True
        print("Evaluating for pretraining...")
        self._test_step()
    def _test_step(self):
        with torch.no_grad():
            #validation and find best threshold
            pred_data={'train':[],'valid':[],'test':[]}
            true_data={'train':[],'valid':[],'test':[]}
            self.model['pretrain'].eval()
            for task in ['valid','test']:
                
                total_len=len(self.edges[task])
                batch_size=self.batch_size['finetune']
                n_batch=int(total_len/batch_size)
                for batch in range(n_batch):
                    i=batch*batch_size
                    if i+batch_size<total_len:
                        end=i+batch_size
                    else:
                        end=total_len
                    #get edge subgraphs for test
                    batch_file=os.path.join(self.finetune_path,'{}/edge_subgraph_{}.bin'.format(task,batch))
                    if os.path.exists(batch_file):
                        subgraph_list,_=dgl.load_graphs(batch_file)
                    else:
                        subgraph_list=self.sampler.get_edge_subgraph(self.edges[task][i:end])
                        dgl.save_graphs(batch_file,subgraph_list)
                    #get score and label
                    #output: 100*7*200  layer_output: 100*6*7*200
                    output,layer_output,_=self.model['pretrain'](subgraph_list)
                    if not self.is_finetuned:
                        source_embed = output[:, 0, :].unsqueeze(1)
                        target_embed = output[:, 1, :].unsqueeze(1).transpose(1, 2)
                        score = torch.bmm(source_embed, target_embed).squeeze(1)#embedding相乘得到相似度分数
                        score = torch.sigmoid(score).data.cpu().numpy().tolist()
                    else:
                        #score:[ft_batch_size,1]
                        #src_embedding/dst_embedding:[ft_batch_size,1,embedding_dim]
                        score,_,_=self.model['finetune'](layer_output)
                    labels=self.edges_label[task][i:end]
                    for ii, _ in enumerate(score):
                        pred_data[task].append(float(score[ii][0]))
                        true_data[task].append(labels[ii])
                    i+=batch_size
            #test and get result
            real_true_data=np.array(true_data['valid'],dtype=np.int)
            self.threshold=self.get_threshold(real_true_data,pred_data['valid'])[0]
            prediction_data=pred_data['test']
            sorted_pred = prediction_data[:]
            sorted_pred.sort()
            # threshold = sorted_pred[-true_num]
            y_pred = np.zeros(len(prediction_data), dtype=np.int32)

            for i, _ in enumerate(prediction_data):
                if prediction_data[i] >= self.threshold:
                    y_pred[i] = 1

            y_true = np.array(true_data['test'])
            y_scores = np.array(prediction_data)
            ps, rs, _ = precision_recall_curve(y_true, y_scores)
            if self.is_finetuned:
                header="Finetuning"
            else:
                header="Pretraining"
            print(f"----------------------Testing for {header}()------------")
            print(
                f"y_true.shape: {y_true.shape}, y_scores.shape: {y_scores.shape}"
                f", y_pred.shape: {y_pred.shape}"
            )
            try:
                roc_auc = roc_auc_score(y_true, y_scores)
            except ValueError:
                roc_auc = 'UNDEFINED'
            f1 = f1_score(y_true, y_pred)
            auc_value = auc(rs,ps)
            
            print(
                f"{header} : ROC-AUC: {roc_auc},"
                f" F1: {f1}, AUC: {auc_value}"
            )
    def get_threshold(self, target, predicted): 
        fpr, tpr, threshold = roc_curve(target, predicted,pos_label=1)
        i = np.arange(len(tpr),dtype=np.int)
        roc = pd.DataFrame(
            { 
                "tf": pd.Series(tpr - (1 - fpr), index=i),
                "threshold": pd.Series(threshold, index=i),
            }
        )
        print()
        roc_t = roc.loc[(roc.tf - 0).abs().argsort()[:1]]
        return list(roc_t["threshold"])
    def loss_calculation(self, pos_score, neg_score):
        # an example hinge loss
        loss = []
        for i in pos_score:
            loss.append(F.logsigmoid(pos_score[i]))
            loss.append(F.logsigmoid(-neg_score[i]))
        loss = torch.cat(loss)
        return -loss.mean()

    def ScorePredictor(self, edge_subgraphs, pairs, x):
        #x:[batch_size*num_nodes*embed_dim]
        score=[]
        labels=[]
        for ii,edge_subgraph in enumerate(edge_subgraphs):
            src_embed=x[ii,0,:]
            dst_embed=x[ii,1,:]
            score.append(torch.dot(src_embed,dst_embed))
            src,dst,label=pairs[ii]
            labels.append(label)
        score=torch.sigmoid(torch.tensor(score))
        res=F.binary_cross_entropy(score,torch.FloatTensor(labels))
        return res
    def nid_to_id(self,subgraph,src):
        for ii,each in enumerate(subgraph.ndata[dgl.NID]):
            if each==src:
                return ii
        return -1