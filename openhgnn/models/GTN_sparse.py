import dgl
import math
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv, EdgeWeightNorm
from ..utils import transform_relation_graph_list
from . import BaseModel, register_model


@register_model('GTN')
class GTN(BaseModel):
    @classmethod
    def build_model_from_args(cls, args, hg):
        if args.identity == True:
            num_edge_type = len(hg.canonical_etypes) + 1
        else:
            num_edge_type = len(hg.canonical_etypes)
        # add self-loop edge
        return cls(num_edge_type=num_edge_type, num_channels=args.num_channels,
                    in_dim=args.in_dim, hidden_dim=args.hidden_dim, num_class=args.out_dim,
                    num_layers=args.num_layers, category=args.category, norm=args.norm_emd_flag, identity=args.identity)

    def __init__(self, num_edge_type, num_channels, in_dim, hidden_dim, num_class, num_layers, category, norm, identity):
        super(GTN, self).__init__()
        self.num_edge_type = num_edge_type
        self.num_channels = num_channels
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_class = num_class
        self.num_layers = num_layers
        self.is_norm = norm
        self.category = category
        self.identity = identity

        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(GTLayer(num_edge_type, num_channels, first=True))
            else:
                layers.append(GTLayer(num_edge_type, num_channels, first=False))
        self.layers = nn.ModuleList(layers)
        self.gcn = GraphConv(in_feats=self.in_dim, out_feats=hidden_dim, norm='none', activation=F.relu)
        self.norm = EdgeWeightNorm(norm='right')
        self.linear1 = nn.Linear(self.hidden_dim * self.num_channels, self.hidden_dim)
        self.linear2 = nn.Linear(self.hidden_dim, self.num_class)
        self.category_idx = None
        self.A = None
        self.h = None

    def normalization(self, H):
        norm_H = []
        for i in range(self.num_channels):
            g = H[i]
            g = dgl.remove_self_loop(g)
            g.edata['w_sum'] = self.norm(g, g.edata['w_sum'])
            norm_H.append(g)
        return norm_H


    def forward(self, hg, h=None):
        with hg.local_scope():
            #Ws = []
            # * =============== Extract edges in original graph ================
            if self.category_idx is None:
                self.A, self.h, self.category_idx = transform_relation_graph_list(hg, category=self.category,
                                                        identity=self.identity)
            # g = dgl.to_homogeneous(hg, ndata='h')
            #X_ = self.gcn(g, self.h)
            A = self.A
            h = self.h
            # * =============== Get new graph structure ================
            for i in range(self.num_layers):
                if i == 0:
                    H, W = self.layers[i](A)
                else:
                    H, W = self.layers[i](A, H)
                if self.is_norm == True:
                    H = self.normalization(H)
                #Ws.append(W)
            # * =============== GCN Encoder ================
            for i in range(self.num_channels):
                g = dgl.remove_self_loop(H[i])
                edge_weight = g.edata['w_sum']
                g = dgl.add_self_loop(g)
                edge_weight = th.cat((edge_weight, th.full((g.number_of_nodes(),), 1, device=g.device)))
                edge_weight = self.norm(g, edge_weight)
                if i == 0:
                    X_ = self.gcn(g, h, edge_weight=edge_weight)
                else:
                    X_ = th.cat((X_, self.gcn(g, h, edge_weight=edge_weight)), dim=1)
            X_ = self.linear1(X_)
            X_ = F.relu(X_)
            y = self.linear2(X_)
            return {self.category: y[self.category_idx]}


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