from os import close
import random
import dgl
import numpy as np
from numpy.lib.function_base import select
import torch as th
from torch.utils import data
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
from dgl.nn.functional import edge_softmax
from openhgnn.models import build_model
import torch.nn.functional as F
from . import BaseFlow, register_flow
from ..tasks import build_task
from sklearn.metrics import f1_score, roc_auc_score

@register_flow("kgcntrainer")
class KGCNTrainer(BaseFlow):
    """Demo flows."""

    def __init__(self, args):
        super(KGCNTrainer, self).__init__(args)
        self.args = args
        self.in_dim = args.in_dim
        self.out_dim = args.out_dim
        self.model_name = args.model
        self.device = args.device
        self.l2_weight = args.weight_decay
        self.task = build_task(args)
        self.g = self.hg

        if args.dataset == 'LastFM4KGCN':
            self.ratingsGraph = self.task.dataset.g_1.to(self.device)
            self.neighborList = [8]
            self.trainIndex, self.evalIndex, self.testIndex = self.task.get_idx()

        self.model = build_model(self.model_name).build_model_from_args(self.args, self.g)
        self.optimizer = th.optim.Adam(self.model.parameters(),lr = self.args.lr, weight_decay=self.args.weight_decay)   
        self.model = self.model.to(self.device)


    def KGCNCollate(self,index):
        item, user = self.ratingsGraph.find_edges(th.tensor(index).to(self.device))
        label = self.ratingsGraph.edata['label'][th.tensor(index).to(self.device)]
        inputData = th.stack([user,item,label]).t().cpu().numpy()
        deleteindex = []
        item_indices = []
        for i in range(len(inputData)):
            if inputData[i][1] in item_indices:
                deleteindex.append(i)
            else:
                item_indices.append(inputData[i][1])
        inputData = np.delete(inputData,deleteindex,axis = 0)
        self.renew_weight(inputData)
        sampler = dgl.dataloading.MultiLayerNeighborSampler(self.neighborList)
        dataloader = dgl.dataloading.NodeDataLoader(
            self.g, list(inputData[:,1]), sampler,
            batch_size=1024,
            shuffle=True,
            drop_last=False,
            num_workers=0)

        block = next(iter(dataloader))[2]
        return block, inputData

    def preprocess(self,dataIndex):
        self.user_emb_matrix,self.entity_emb_matrix,self.relation_emb_matrix = self.model.get_embeddings()
        self.g.ndata['embedding'] = self.entity_emb_matrix
        dataloader = DataLoader(dataIndex, batch_size = self.args.batch_size, shuffle=True,collate_fn = self.KGCNCollate)
        self.dataloader_it = iter(dataloader)
        return

    def train(self):   
        epoch_iter = self.args.epoch_iter
        for self.epoch in range(epoch_iter):
            random.shuffle(self.trainIndex)
            self._mini_train_step()
            print('train_data:')
            self.evaluate(self.trainIndex)

            print('eval_data:')
            self.evaluate(self.evalIndex)

            print('test_data:')
            self.evaluate(self.testIndex)
        print('********************here_train****************')
        pass

    def _mini_train_step(self,):

        self.preprocess(self.trainIndex)
        L = 0
        for block, inputData in self.dataloader_it:
            self.labels, self.scores = self.model(block, inputData)
            loss = self.loss_calculation()
            self.optimizer.zero_grad() 
            loss.backward()
            self.optimizer.step()
            L = L+loss

        f = open('result.txt','a') 
        res = "step: "+str(self.epoch)+'full_Loss: '+str(L)+'\n'
        f.write(res)
        print("step:",self.epoch,'full_Loss:', L)


    def evaluate(self,dataIndex):
        self.preprocess(dataIndex)
        labelsList = []
        scoresList = []

        for block, inputData in self.dataloader_it:
            self.labels, self.scores = self.model(block, inputData)
            labelsList+=(self.labels.detach().cpu().numpy().tolist())
            scoresList+=(th.sigmoid(self.scores).detach().cpu().numpy().tolist())

        auc = roc_auc_score(y_true = np.array(labelsList), y_score = np.array(scoresList))    
        for i in range(len(scoresList)):
            if scoresList[i] >= 0.5:
                scoresList[i] = 1
            else:
                scoresList[i] = 0
    
        f1 = f1_score(y_true = np.array(labelsList), y_pred = np.array(scoresList))

        f = open('result.txt','a')
        f.write('auc:'+str(auc)+'   f1:'+str(f1)+'\n')
        print('auc:',auc,'   f1:',f1)    
        return auc ,f1
    
    def loss_calculation(self):
        labels, logits = self.labels, self.scores

        # output =  -labels * th.log(th.sigmoid(logits)) - (1-labels) * th.log(1-th.sigmoid(logits))
        output = F.binary_cross_entropy_with_logits(logits,labels.to(th.float32))
        self.base_loss = th.mean(output)

        self.l2_loss = th.norm(self.user_emb_matrix) ** 2/2 + th.norm(self.entity_emb_matrix) **2/2 + th.norm(self.relation_emb_matrix) ** 2/2
        '''
        for aggregator in self.aggregators:
            self.l2_loss = self.l2_loss + torch.norm(aggregator.weights) **2/2
        '''
 
        loss = self.base_loss + self.l2_weight * self.l2_loss
        return loss

    def _full_train_setp(self):
        pass

    def _test_step(self, split=None, logits=None):
        pass
    
    def renew_weight(self,inputData):
        user_indices = inputData[:,0]
        self.user_embeddings = self.user_emb_matrix[user_indices]
        weight = th.mm(self.relation_emb_matrix[self.g.edata['relation'].cpu().numpy()], self.user_embeddings.t())   
        weight = weight.unsqueeze(dim=-1)
        self.g.edata['weight'] = edge_softmax(self.g, th.as_tensor(weight))

    