from dgl.sparse import SparseMatrix
import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
import math
# from torch.nn.parameter import Parameter
from torch.nn import Parameter

from dgl.nn.pytorch.conv import GraphConv
import dgl
import numpy
import scipy.sparse
from . import BaseModel, register_model

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self,g,input):
        g.ndata['h'] = input 
        g.update_all(message_func=dgl.function.u_mul_e('h','w','m'),reduce_func=dgl.function.sum('m','h'))
        support = g.ndata['h']
        output = support @ self.weight
        return output + self.bias

@register_model('MHGCN')
class MHGCN(BaseModel):
    """
    **Title:** `Multiplex Heterogeneous Graph Convolutional Network <https://doi.org/10.1145/3534678.3539482>`_
    **Authors:** Pengyang Yu, Chaofan Fu, Yanwei Yu, Chao Huang, Zhongying Zhao, Junyu Dong.
    Parameters
    ----------
    in_dim : int
        Input feature size.
    out_dim : int
        Output feature size.
    etype_num : int
        Number of relation types. 
    num_hidden_layers : int
        Number of hidden GCN layers
    """

    @classmethod 
    def build_model_from_args(cls,args,g):
        return cls(args.feature_dim,
                   args.emb_dim,
                   args.etype_num,
                   args.num_layers - 2)

    def __init__(self, 
                 in_dim,
                 out_dim,
                 etype_num,
                 num_hidden_layers=1,):
        super(MHGCN,self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.input_layer = GraphConvolution(in_dim, out_dim)
        # hidden layers
        for i in range(num_hidden_layers):
            self.layers.append(GraphConvolution(out_dim, out_dim))
        # output layer
        self.layers.append(GraphConvolution(out_dim, out_dim))
        self.weight_b = torch.nn.Parameter(torch.FloatTensor(etype_num, 1), requires_grad=True)
        torch.nn.init.uniform_(self.weight_b, a=0, b=0.1)

    def _multiplex_relation_aggregation(self, g:dgl.DGLGraph) -> dgl.DGLGraph:
        """
        Multiplex relation aggregation
        """
        g.edata['w'] = g.edata['tag'] @ self.weight_b
        return g

    def forward(self, g, features):
        # features = features.int().float()
        """
        Multiplex relation aggregation (2)
        """
        g.edata['w'] = g.edata['tag'] @ self.weight_b
        """
        Multiplex layer graph convolution
        """
        emb = self.input_layer(g,features)
        emb_accumulate = emb
        for i, layer in enumerate(self.layers):
            emb = layer(g, emb)
            emb_accumulate = emb_accumulate +  emb
        # Average pooling
        emb_avg = emb_accumulate / (len(self.layers) + 1)
        return emb_avg