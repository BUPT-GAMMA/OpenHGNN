import torch as th
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
from openhgnn.utils.utils import extract_edge_with_id_edge
import pdb
import dgl
from dgl.nn.pytorch import GraphConv
import torch_sparse_old
# from torch_geometric.utils.num_nodes import maybe_num_nodes
# from torch_geometric.utils import remove_self_loops, add_self_loops


class GTN(nn.Module):

    def __init__(self, num_edge, num_channels, w_in, w_out, num_class, num_layers):
        super(GTN, self).__init__()
        self.num_edge = num_edge
        self.num_channels = num_channels
        #self.num_nodes = num_nodes
        self.w_in = w_in
        self.w_out = w_out
        self.num_class = num_class
        self.num_layers = num_layers
        num_nodes =8994
        self.num_nodes = num_nodes
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(GTLayer(num_edge, num_channels, num_nodes, first=True))
            else:
                layers.append(GTLayer(num_edge, num_channels, num_nodes, first=False))
        self.layers = nn.ModuleList(layers)
        self.loss = nn.CrossEntropyLoss()
        self.gcn = GraphConv(in_feats=self.w_in, out_feats=w_out)
        self.linear1 = nn.Linear(self.w_out * self.num_channels, self.w_out)
        self.linear2 = nn.Linear(self.w_out, self.num_class)

    def normalization(self, H):
        norm_H = []
        for i in range(self.num_channels):
            edge, value = H[i]
            #edge, value = remove_self_loops(edge, value)
            deg_row, deg_col = self.norm(edge.detach(), self.num_nodes, value)
            value = deg_col * value
            norm_H.append((edge, value))
        return norm_H

    def norm(self, edge_index, num_nodes, edge_weight, improved=False, dtype=None):
        if edge_weight is None:
            edge_weight = th.ones((edge_index.size(1),),
                                     dtype=dtype,
                                     device=edge_index.device)
        edge_weight = edge_weight.view(-1)
        assert edge_weight.size(0) == edge_index.size(1)
        row, col = edge_index
        deg = th.zeros(num_nodes, dtype=edge_weight.dtype, device=edge_weight.device)
        deg.scatter_add_(0, col, edge_weight)
        deg_inv_sqrt = deg.pow(-1)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return deg_inv_sqrt[row], deg_inv_sqrt[col]

    def forward(self, g_homo):
        with g_homo.local_scope():
            #Ws = []
            # * =============== Extract edges in original graph ================
            A = extract_edge_with_id_edge(g_homo)
            # * =============== Get new graph structure ================
            for i in range(self.num_layers):
                if i == 0:
                    H, W = self.layers[i](A)
                else:
                    H, W = self.layers[i](A, H)
                H = self.normalization(H)
                #Ws.append(W)
            # * =============== GCN Encoder ================
            h = g_homo.ndata['h']
            for i in range(self.num_channels):
                edge_index, edge_weight = H[i][0], H[i][1]
                new_g = dgl.graph((edge_index[0], edge_index[1]), idtype=th.int32)
                if i == 0:
                    X_ = self.gcn(new_g, h, edge_weight=edge_weight)
                    X_ = F.relu(X_)
                else:
                    X_ = th.cat((X_, F.relu(self.gcn(new_g, h, edge_weight=edge_weight))),
                                   dim=1)
            # x = self.gcn(g_homo.to('cpu'), h)
            X_ = self.linear1(X_)
            X_ = F.relu(X_)
            y = self.linear2(X_)
            return y


class GTLayer(nn.Module):

    def __init__(self, in_channels, out_channels, num_nodes, first=True):
        super(GTLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.first = first
        self.num_nodes = num_nodes
        if self.first == True:
            self.conv1 = GTConv(in_channels, out_channels, num_nodes)
            self.conv2 = GTConv(in_channels, out_channels, num_nodes)
        else:
            self.conv1 = GTConv(in_channels, out_channels, num_nodes)

    def forward(self, A, H_=None):
        if self.first == True:
            result_A = self.conv1(A)
            result_B = self.conv2(A)
            W = [(F.softmax(self.conv1.weight, dim=1)).detach(), (F.softmax(self.conv2.weight, dim=1)).detach()]
        else:
            result_A = H_
            result_B = self.conv1(A)
            W = [(F.softmax(self.conv1.weight, dim=1)).detach()]
        H = []
        for i in range(len(result_A)):
            a_edge, a_value = result_A[i]
            b_edge, b_value = result_B[i]

            a_edge = a_edge.cpu()
            a_value = a_value.cpu()
            b_edge = b_edge.cpu()
            b_value = b_value.cpu()
            edges, values = torch_sparse_old.spspmm(a_edge, a_value, b_edge, b_value, self.num_nodes, self.num_nodes,
                                                    self.num_nodes)
            edges = edges.cuda()
            values = values.cuda()
            H.append((edges, values))
        return H, W


# class GTConv(nn.Module):
#
#     def __init__(self, in_channels, out_channels, num_nodes):
#         super(GTConv, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.weight = nn.Parameter(th.Tensor(in_channels, out_channels))
#         self.bias = None
#         self.num_nodes = num_nodes
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         n = self.in_channels
#         nn.init.normal_(self.weight, std=0.01)
#         if self.bias is not None:
#             fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
#             bound = 1 / math.sqrt(fan_in)
#             nn.init.uniform_(self.bias, -bound, bound)
#
#     def forward(self, A):
#         filter = F.softmax(self.weight, dim=0)
#         results = []
#         for i in range(self.out_channels):
#             for j, (edge_index, edge_value) in enumerate(A):
#                 if j == 0:
#                     total_edge_index = edge_index
#                     total_edge_value = edge_value * filter[j][i]
#                 else:
#                     total_edge_index = th.cat((total_edge_index, edge_index), dim=1)
#                     total_edge_value = th.cat((total_edge_value, edge_value * filter[j][i]))
#             index, value = torch_sparse_old.coalesce(total_edge_index.detach(), total_edge_value, m=self.num_nodes,
#                                                  n=self.num_nodes)
#             # index, value = total_edge_index, total_edge_value
#             results.append((index, value))
#         return results


class GTConv(nn.Module):

    def __init__(self, in_channels, out_channels, num_nodes):
        super(GTConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(th.Tensor(out_channels, in_channels))
        self.bias = None
        self.num_nodes = num_nodes
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        nn.init.normal_(self.weight, std=0.01)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, A):
        filter = F.softmax(self.weight, dim=1)
        num_channels = filter.shape[0]
        results = []
        for i in range(num_channels):
            for j, (edge_index, edge_value) in enumerate(A):
                if j == 0:
                    total_edge_index = edge_index
                    total_edge_value = edge_value * filter[i][j]
                else:
                    total_edge_index = th.cat((total_edge_index, edge_index), dim=1)
                    total_edge_value = th.cat((total_edge_value, edge_value * filter[i][j]))
            index, value = torch_sparse_old.coalesce(total_edge_index.detach(), total_edge_value, m=self.num_nodes,
                                                 n=self.num_nodes)
            results.append((index, value))
        return results