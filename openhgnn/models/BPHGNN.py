import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
import math
from torch.nn.parameter import Parameter
import numpy as np
from scipy.sparse import coo_matrix

device=torch.device('cuda')
def coototensor(A): 
    """
    Convert a coo_matrix to a torch sparse tensor
    """

    values = A.data
    indices = np.vstack((A.row, A.col)) 
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = A.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))

def construct_adj(encode, struct_weight):  
    weight=torch.diag(struct_weight)
    adjust_encode=torch.mm(encode.to(torch.float32),weight)
    # print(adjust_encode)
    struct_adj=torch.mm(adjust_encode,adjust_encode.t())
    normal_struct_adj=torch.nn.functional.softmax(struct_adj, dim=1)
    return normal_struct_adj

def adj_matrix_weight_merge(A, adj_weight): 
    """
    Multiplex Relation Aggregation
    """

    N = A[0][0].shape[0]
    temp = coo_matrix((N, N))
    temp = coototensor(temp)  
    # Alibaba_small
    a = coototensor(A[0][0].tocoo())
    b = coototensor(A[0][1].tocoo())
    c = coototensor(A[0][2].tocoo())
    d = coototensor(A[0][3].tocoo())
    e = coototensor(A[0][4].tocoo())
    f = coototensor(A[0][5].tocoo())
    g = coototensor(A[0][6].tocoo())

    A_t = torch.stack([a, b,c,d,e,f,g], dim=2).to_dense()

    temp = torch.matmul(A_t, adj_weight)  
    temp = torch.squeeze(temp, 2)

    return temp + temp.transpose(0, 1)

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features)).to(device)
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features)).to(device) 
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        try:
            input = input.float()
        except:
            pass
        support = torch.mm(input, self.weight).to(device)  
        output = torch.spmm(adj, support).to(device)   
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class GCN(nn.Module):
    """
    A Two-layer GCN.
    """
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj, use_relu=True):
        x = self.gc1(x, adj)
        if use_relu:
            x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x






class BPHGNN(nn.Module):
    def __init__(self, nfeat, nhid, out, dropout):
        super(BPHGNN, self).__init__()
        """
        # Multilayer Graph Convolution
        """
        self.gc1 = GraphConvolution(nfeat, out)
        self.gc2 = GraphConvolution(out, out)
        # self.gc3 = GraphConvolution(out, out)
        # self.gc3 = GraphConvolution(out, out)
        # self.gc4 = GraphConvolution(out, out)
        # self.gc5 = GraphConvolution(out, out)
        self.dropout = dropout

        """
        Set the trainable weight of adjacency matrix aggregation
        """

        # Alibaba_small
        self.weight_b = torch.nn.Parameter(torch.FloatTensor(7, 1), requires_grad=True)
        torch.nn.init.uniform_(self.weight_b,a = 0,b = 0.1)
        self.struct_weight=torch.nn.Parameter(torch.ones(7), requires_grad=True)
        torch.nn.init.uniform_(self.struct_weight, a=0, b=0.1)

    def forward(self, feature, A,encode,use_relu=True):

        final_A = adj_matrix_weight_merge(A, self.weight_b).to(device)

        try:
            feature = torch.tensor(feature.astype(float).toarray()).to(device)
        except:
            try:
                feature = torch.from_numpy(feature.toarray()).to(device)
            except:
                pass

        # # # Output of single-layer GCN
        U1 = self.gc1(feature, final_A)
        # Output of two-layer GCN
        U2 = self.gc2(U1, final_A)
        # return (U1+U2)/2, (U1+U2)/2, (U1+U2)/2

        struct_adj=construct_adj(encode,self.struct_weight).to(device)
        print(self.struct_weight)
        U3 = self.gc1(feature, struct_adj)
        U4 = self.gc2(U3, struct_adj)
        # result=(U1+U2+U4)/2
        result=((U1+U2)/2+U4)/2
        return result,(U1+U2)/2, U4