import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
import math
from torch.nn.parameter import Parameter
import numpy as np
from scipy.sparse import coo_matrix

def coototensor(A):  # 将一个 SciPy 的 coo_matrix（坐标列表格式的稀疏矩阵）转换为 PyTorch 的稀疏张量。
    """
    Convert a coo_matrix to a torch sparse tensor
    """

    values = A.data
    indices = np.vstack((A.row, A.col))  # 将行和列索引堆叠在一起以形成坐标。
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = A.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))

def adj_matrix_weight_merge(A, adj_weight):  # 将多个稀疏矩阵（通过 A 参数传入）转换为 PyTorch 的稠密矩阵，并进行加权聚合
    """
    Multiplex Relation Aggregation
    """

    N = A[0][0].shape[0]
    temp = coo_matrix((N, N))
    temp = coototensor(temp)  # 将 A 中的多个 COO 矩阵转换为 PyTorch 的稀疏张量

    # Alibaba_small
    a = coototensor(A[0][0].tocoo())
    b = coototensor(A[0][1].tocoo())
    c = coototensor(A[0][2].tocoo())
    d = coototensor(A[0][3].tocoo())
    e = coototensor(A[0][4].tocoo())
    f = coototensor(A[0][5].tocoo())
    g = coototensor(A[0][6].tocoo())

    A_t = torch.stack([a, b,c,d,e,f,g], dim=2).to_dense()

    # # Alibaba_large
    # a = coototensor(A[0][0].tocoo())
    # b = coototensor(A[0][1].tocoo())
    # c = coototensor(A[0][2].tocoo())
    # d = coototensor(A[0][3].tocoo())
    # e = coototensor(A[0][4].tocoo())
    # f = coototensor(A[0][5].tocoo())
    # g = coototensor(A[0][6].tocoo())
    #
    # A_t = torch.stack([a, b,c,d,e,f,g], dim=2).to_dense()



    # # Alibaba
    # a = coototensor(A[0][0].tocoo())
    #
    # f = coototensor(A[0][5].tocoo())
    # g = coototensor(A[0][6].tocoo())
    # h = coototensor(A[0][7].tocoo())
    #
    # j = coototensor(A[0][9].tocoo())
    # k = coototensor(A[0][10].tocoo())
    # l = coototensor(A[0][11].tocoo())
    # m = coototensor(A[0][12].tocoo())
    # n = coototensor(A[0][13].tocoo())
    # o = coototensor(A[0][14].tocoo())
    #
    # A_t = torch.stack([a, f,g,h,j,k,l,m,n,o], dim=2).to_dense()

    # # DBLP_small
    # a = coototensor(A[0][0].tocoo())
    # b = coototensor(A[0][1].tocoo())
    # A_t = torch.stack([a, b], dim=2).to_dense()

    # # imdb_small
    # a = coototensor(A[0][0].tocoo())
    # b = coototensor(A[0][1].tocoo())
    # A_t = torch.stack([a, b], dim=2).to_dense()


    # DBLP
    # a = coototensor(A[0][0].tocoo())
    # b = coototensor(A[0][1].tocoo())
    # c = coototensor(A[0][2].tocoo())
    # A_t = torch.stack([a, b, c], dim=2).to_dense()

    # Aminer
    # a = coototensor(A[0][0].tocoo())
    # b = coototensor(A[0][1].tocoo())
    # c = coototensor(A[0][2].tocoo())
    # A_t = torch.stack([a, c], dim=2).to_dense()


    # IMDB
    # a = coototensor(A[0][0].tocoo())
    # b = coototensor(A[0][2].tocoo())
    # A_t = torch.stack([a, b], dim=2).to_dense()

    temp = torch.matmul(A_t, adj_weight)  # 矩阵乘法
    temp = torch.squeeze(temp, 2)

    return temp + temp.transpose(0, 1)

def normarlize(H):  # 对输入的张量 H 进行归一化处理
    # DV = np.sum(H, axis=1)
    # DV += 1e-12
    # DV2 = np.mat(np.diag(np.power(DV, -1)))
    # G = DV2 * H

    DV = torch.sum(H, dim=1)
    DV += 1e-12

    DE=torch.sum(H,dim=0)
    DE += 1e-12
    DV2 = torch.diag(torch.pow(DV, -1))
    DE2 = torch.diag(torch.pow(DE, -1/2))
    G = torch.mm(DV2,H)
    G = torch.mm(G,DE2)

    return G

def construct_adj(encode, struct_weight):  # 构建权重对角矩阵（用于广度行为模式聚合）
    weight=torch.diag(struct_weight)
    adjust_encode=torch.mm(encode.to(torch.float32),weight)
    # print(adjust_encode)
    struct_adj=torch.mm(adjust_encode,adjust_encode.t())
    normal_struct_adj=torch.nn.functional.softmax(struct_adj, dim=1)
    return normal_struct_adj

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))#鏉冮噸鐭╅樀
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))#鍋忕Щ鍚戦噺
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
        support = torch.mm(input, self.weight)  # 矩阵乘法
        output = torch.spmm(adj, support)  # 稀疏矩阵乘法
        if self.bias is not None:
            return output + self.bias
        else:
            return output
    #
    # def __repr__(self):
    #     return self.__class__.__name__ + ' (' \
    #            + str(self.in_features) + ' -> ' \
    #            + str(self.out_features) + ')'

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






class MHGCN(nn.Module):
    def __init__(self, nfeat, nhid, out, dropout):
        super(MHGCN, self).__init__()
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


        # # Alibaba_large
        # self.weight_b = torch.nn.Parameter(torch.FloatTensor(7, 1), requires_grad=True)
        # torch.nn.init.uniform_(self.weight_b,a = 0,b = 0.1)
        # self.struct_weight=torch.nn.Parameter(torch.ones(7), requires_grad=True)
        # torch.nn.init.uniform_(self.struct_weight, a=0, b=0.1)

        # # Alibaba
        # self.weight_b = torch.nn.Parameter(torch.FloatTensor(10, 1), requires_grad=True)
        # torch.nn.init.uniform_(self.weight_b,a = 0,b = 0.1)
        # self.struct_weight=torch.nn.Parameter(torch.ones(15), requires_grad=True)
        # torch.nn.init.uniform_(self.struct_weight, a=0, b=0.1)
        # DBLP
        # self.weight_b = torch.nn.Parameter(torch.FloatTensor(3, 1), requires_grad=True)
        # torch.nn.init.uniform_(self.weight_b, a=0, b=0.1)

        # #dblp_small
        # self.weight_b = torch.nn.Parameter(torch.FloatTensor(2, 1), requires_grad=True)
        # torch.nn.init.uniform_(self.weight_b,a = 0,b = 0.1)
        # self.struct_weight = torch.nn.Parameter(torch.ones(3), requires_grad=True)
        # torch.nn.init.uniform_(self.struct_weight, a=0, b=0.1)
        
        
        # #imdb_small
        # self.weight_b = torch.nn.Parameter(torch.FloatTensor(2, 1), requires_grad=True)
        # torch.nn.init.uniform_(self.weight_b,a = 0,b = 0.1)
        # self.struct_weight = torch.nn.Parameter(torch.ones(3), requires_grad=True)
        # torch.nn.init.uniform_(self.struct_weight, a=0, b=0.1)

        # Aminer
        # self.weight_b = torch.nn.Parameter(torch.FloatTensor(2, 1), requires_grad=True)
        # torch.nn.init.uniform_(self.weight_b, a=0, b=1)

        # IMDB
        # self.weight_b = torch.nn.Parameter(torch.FloatTensor(2, 1), requires_grad=True)
        # torch.nn.init.uniform_(self.weight_b, a=0, b=0.1)
        # self.weight_b2 = torch.nn.Parameter(torch.FloatTensor(2, 1), requires_grad=True)
        # torch.nn.init.uniform_(self.weight_b2, a=0, b=0.1)
        # self.struct_weight=torch.nn.Parameter(torch.ones(3), requires_grad=True)
        # torch.nn.init.uniform_(self.struct_weight, a=0, b=0.1)

    def forward(self, feature, A,encode,use_relu=True):

        final_A = adj_matrix_weight_merge(A, self.weight_b)
        # final_A2 = adj_matrix_weight_merge(A, self.weight_b2)
        # final_A=final_A+torch.eye(final_A.size()[0])

        # final_A2 = adj_matrix_weight_merge(A, self.weight_b2)
        # final_A2=final_A2+torch.eye(final_A2.size()[0])
        try:
            feature = torch.tensor(feature.astype(float).toarray())
        except:
            try:
                feature = torch.from_numpy(feature.toarray())
            except:
                pass

        # # # Output of single-layer GCN
        U1 = self.gc1(feature, final_A)
        # Output of two-layer GCN
        U2 = self.gc2(U1, final_A)
        # return (U1+U2)/2, (U1+U2)/2, (U1+U2)/2

        struct_adj=construct_adj(encode,self.struct_weight)
        print(self.struct_weight)
        U3 = self.gc1(feature, struct_adj)
        U4 = self.gc2(U3, struct_adj)
        # result=(U1+U2+U4)/2
        result=((U1+U2)/2+U4)/2
        return result,(U1+U2)/2, U4

        # Output of single-layer GCN
        # U1 = self.gc1(feature, final_A)
        # # Output of two-layer GCN
        # U2 = self.gc2(U1, final_A)
        # # return (U1+U2)/2
        # # struct_adj=construct_adj(encode,self.struct_weight)
        # # print(self.struct_weight)
        # U3 = self.gc1(feature, new_adj)
        # U4 = self.gc2(U3, new_adj)
        # result=((U1+U2)/2+U4)/2
        # #
        # return result, (U1+U2)/2, U4



        # # # Output of single-layer GCN
        # U1 = self.gc1(feature, final_A)
        # # Output of two-layer GCN
        # U2 = self.gc2(U1, final_A)
        # U3 = torch.tanh(self.gc1(feature, new_adj))
        # U4 = torch.tanh(self.gc2(U3, new_adj))
        # return (U2+U4)/2


        # # # Output of single-layer GCN
        # U1 = self.gc1(feature, new_adj)
        # # Output of two-layer GCN
        # U2 = self.gc2(U1, new_adj)
        # return U2,U2,U2
        #
        # # U3 = self.gc3(U2, final_A)
        # # U4 = self.gc4(U2, final_A)
        # # U5 = self.gc5(U2, final_A)
        #
        # # Average pooling
        #
        # return U2
