
import dgl
import numpy as np
import torch as th
import torch.nn as nn
import dgl.function as fn
import torch.nn.functional as F
from . import BaseModel, register_model
from dgl.nn.functional import edge_softmax
import torch.nn.functional as F

@register_model('KGCN')
class KGCN(BaseModel):
    r"""

    Description
    -----------
    """
    @classmethod
    def build_model_from_args(cls, args, g):
        return cls(g, args)

    def __init__(self, g, args):
        super(KGCN, self).__init__()
        self.g = g
        self.args = args
        self.in_dim = args.in_dim
        self.out_dim = args.out_dim

        self.entity_emb_matrix = nn.Parameter(th.rand(self.g.num_nodes(), self.in_dim)*2-1)
        self.relation_emb_matrix = nn.Parameter(th.rand(args.n_relation, self.in_dim)*2-1)
        self.user_emb_matrix  = nn.Parameter(th.rand(args.n_user, self.in_dim)*2-1)#随机生成[-1,1]

        if self.args.aggregate == 'CONCAT':
            self.agg = nn.Linear(self.in_dim*2, self.out_dim)
        else:
            self.agg = nn.Linear(self.in_dim, self.out_dim)


    def aggregate(self):
       
        self.sub_g.update_all(fn.u_mul_e('embedding', 'weight', 'm'),fn.sum('m', 'ft'))
        
        self.userList = []
        self.labelList = []
        embeddingList = []
        for i in range(len(self.data)):
            weightIndex = np.where(self.itemlist==int(self.sub_g.dstdata['_ID'][i]))
            if self.args.aggregate == 'SUM':
                embeddingList.append(self.sub_g.dstdata['embedding'][i] + self.sub_g.dstdata['ft'][i][weightIndex]) 
            elif self.args.aggregate == 'CONCAT':
                embeddingList.append(th.cat([self.sub_g.dstdata['embedding'][i], self.sub_g.dstdata['ft'][i][weightIndex].squeeze(0)],dim=-1)) 
            elif self.args.aggregate == 'NEIGHBOR':
                embeddingList.append(self.sub_g.dstdata['embedding'][i])
            self.userList.append(int(self.user_indices[weightIndex]))
            self.labelList.append(int(self.labels[weightIndex]))

        self.sub_g.dstdata['embedding'] = th.stack(embeddingList).squeeze(1)
        output = F.dropout(self.sub_g.dstdata['embedding'],p=0)
        if self.layer+1 == len(self.blocks):
            self.item_embeddings = th.tanh(self.agg(output))
        else:
            self.item_embeddings = th.relu(self.agg(output))
    
    def get_score(self):
        self.user_embeddings = self.user_emb_matrix[np.array(self.userList)]
        self.scores = th.sum(self.user_embeddings * self.item_embeddings, dim=1)
        self.scores_normalized = th.sigmoid(self.scores)


    def get_embeddings(self):
        return self.user_emb_matrix,self.entity_emb_matrix,self.relation_emb_matrix

        
    def forward(self,blocks, inputdata):
        self.data = inputdata
        self.blocks = blocks
        self.user_indices = self.data[:,0]
        self.itemlist = self.data[:,1]

        self.labels = self.data[:,2]
        for self.layer in range(len(blocks)):
            self.sub_g = blocks[self.layer]
            self.aggregate()

        self.get_score()
        self.labels = th.tensor(self.labelList).to(self.args.device)
        #loss = self.loss_calculation()

        return self.labels, self.scores