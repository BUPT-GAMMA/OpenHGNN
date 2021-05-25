import dgl
import math
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.pytorch import GraphConv
from ..utils import extract_edge_with_id_edge
from . import BaseModel, register_model


@register_model('GTN')
class GTN(BaseModel):
    @classmethod
    def build_model_from_args(cls, args, hg):
        num_edge_type = len(hg.canonical_etypes) + 1
        # add self-loop edge
        return cls(num_edge_type=num_edge_type, num_channels=args.num_channels,
                    in_dim=args.in_dim, hidden_dim=args.hidden_dim, num_class=args.out_dim,
                    num_layers=args.num_layers, norm=args.norm_emd_flag)

    def __init__(self, num_edge_type, num_channels, in_dim, hidden_dim, num_class, num_layers, norm):
        super(GTN, self).__init__()
        self.num_edge_type = num_edge_type
        self.num_channels = num_channels
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_class = num_class
        self.num_layers = num_layers
        self.is_norm = norm

        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(GTLayer(num_edge_type, num_channels, first=True))
            else:
                layers.append(GTLayer(num_edge_type, num_channels, first=False))
        self.layers = nn.ModuleList(layers)
        self.gcn = GraphConv(in_feats=self.in_dim, out_feats=hidden_dim)
        self.linear1 = nn.Linear(self.hidden_dim * self.num_channels, self.hidden_dim)
        self.linear2 = nn.Linear(self.hidden_dim, self.num_class)

    def normalization(self, H):
        norm_H = []
        for i in range(self.num_channels):
            g = H[i]
            #edge, value = remove_self_loops(edge, value)
            #g = dgl.remove_self_loop(g)
            g = self.norm(g)
            norm_H.append(g)
        return norm_H

    def norm(self, g):
        in_deg = g.in_degrees(range(g.number_of_nodes())).float()
        norm = 1.0 / in_deg
        norm[th.isinf(norm)] = 0
        g.ndata['norm'] = norm
        g.apply_edges(fn.e_mul_v('w_sum', 'norm', 'w_sum'))
        return g

    def forward(self, hg, h=None):
        with hg.local_scope():
            #Ws = []
            # * =============== Extract edges in original graph ================
            A, h = extract_edge_with_id_edge(hg)
            # * =============== Get new graph structure ================
            for i in range(self.num_layers):
                if i == 0:
                    H, W = self.layers[i](A)
                else:
                    H, W = self.layers[i](A, H)
                if self.is_norm:
                    H = self.normalization(H)
                #Ws.append(W)
            # * =============== GCN Encoder ================
            for i in range(self.num_channels):
                g = H[i]
                edge_weight = g.edata['w_sum']
                if i == 0:
                    X_ = self.gcn(g, h, edge_weight=edge_weight)
                    X_ = F.relu(X_)
                else:
                    X_ = th.cat((X_, F.relu(self.gcn(g, h, edge_weight=edge_weight))),
                                   dim=1)
            X_ = self.linear1(X_)
            X_ = F.relu(X_)
            y = self.linear2(X_)
            return {'paper': y[5912:8937]}


class GTLayer(nn.Module):

    def __init__(self, in_channels, out_channels, first=True):
        super(GTLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.first = first
        if self.first == True:
            self.conv1 = GTConv(in_channels, out_channels)
            self.conv2 = GTConv(in_channels, out_channels)
        else:
            self.conv1 = GTConv(in_channels, out_channels)

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
            g = dgl.adj_product_graph(result_A[i], result_B[i], 'w_sum')
            H.append(g)
        return H, W


class GTConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(GTConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(th.Tensor(out_channels, in_channels))
        self.bias = None
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
            for j, g in enumerate(A):
                A[j].edata['w_sum'] = g.edata['w'] * filter[i][j]
            sum_g = dgl.adj_sum_graph(A, 'w_sum')
            results.append(sum_g)
        return results