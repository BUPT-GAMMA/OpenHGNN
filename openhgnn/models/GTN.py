import dgl
import torch as th
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
from openhgnn.utils.utils import extract_mtx_with_id_edge
from . import BaseModel, register_model


@register_model('GTN')
class GTN(BaseModel):
    @classmethod
    def build_model_from_args(cls, args, hg):
        num_edges = len(hg.canonical_etypes) + 1
        in_dim = hg.ndata['h']['paper'].shape[1]

        return cls(num_edge=num_edges, num_channels=args.num_channels,
                    w_in=in_dim,
                    w_out=args.hidden_dim,
                    num_class=args.out_dim,
                    num_layers=args.num_layers,
                   norm=args.norm_emd_flag)

    def __init__(self, num_edge, num_channels, w_in, w_out, num_class, num_layers, norm):
        super(GTN, self).__init__()

        self.num_edge = num_edge
        self.num_channels = num_channels
        self.w_in = w_in
        self.w_out = w_out
        self.num_class = num_class
        self.num_layers = num_layers
        self.is_norm = norm

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.layers.append(GTLayer(num_edge, num_channels, first=True))
            else:
                self.layers.append(GTLayer(num_edge, num_channels, first=False))


        self.weight = nn.Parameter(th.Tensor(w_in, w_out))
        self.bias = nn.Parameter(th.Tensor(w_out))
        self.loss = nn.CrossEntropyLoss()
        self.linear1 = nn.Linear(self.w_out * self.num_channels, self.w_out)
        self.linear2 = nn.Linear(self.w_out, self.num_class)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def gcn_conv(self, X, H):
        X = th.mm(X, self.weight)
        H = self.norm(H, add=True)
        return th.mm(H.t(), X)

    def normalization(self, H):
        for i in range(self.num_channels):
            if i == 0:
                H_ = self.norm(H[i, :, :]).unsqueeze(0)
            else:
                H_ = th.cat((H_, self.norm(H[i, :, :]).unsqueeze(0)), dim=0)
        return H_

    def norm(self, H, add=False):
        H = H.t()
        id_matirx = (th.eye(H.shape[0]) == 0).type(th.FloatTensor).to(H.device)
        if add == False:
            H = H * id_matirx
        else:
            H = H * id_matirx + th.eye(H.shape[0]).type(th.FloatTensor).to(H.device)
        deg = th.sum(H, dim=1)
        deg_inv = deg.pow(-1)
        deg_inv[deg_inv == float('inf')] = 0
        deg_inv = deg_inv * th.eye(H.shape[0]).type(th.FloatTensor).to(deg_inv.device)
        H = th.mm(deg_inv, H)
        H = H.t()
        return H

    def forward(self, g):
        g_homo = dgl.to_homogeneous(g, ndata=['h'])
        with g_homo.local_scope():
            ctx = g_homo.device
            A = extract_mtx_with_id_edge(g_homo)
            X = g_homo.ndata['h']
            A = A.unsqueeze(0)
            Ws = []
            for i in range(self.num_layers):
                if i == 0:
                    H, W = self.layers[i](A)
                else:
                    H = self.normalization(H)
                    H, W = self.layers[i](A, H)
                Ws.append(W)

            # H,W1 = self.layer1(A)
            # H = self.normalization(H)
            # H,W2 = self.layer2(A, H)
            # H = self.normalization(H)
            # H,W3 = self.layer3(A, H)
            for i in range(self.num_channels):
                if i == 0:
                    X_ = F.relu(self.gcn_conv(X, H[i]))
                else:
                    X_tmp = F.relu(self.gcn_conv(X, H[i]))
                    X_ = th.cat((X_, X_tmp), dim=1)
            #GCN

            X_ = self.linear1(X_)
            X_ = F.relu(X_)
            y = self.linear2(X_)
        return {'paper': y}


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
            a = self.conv1(A)
            b = self.conv2(A)
            H = th.bmm(a, b)
            W = [(F.softmax(self.conv1.weight, dim=1)).detach(), (F.softmax(self.conv2.weight, dim=1)).detach()]
        else:
            a = self.conv1(A)
            H = th.bmm(H_, a)
            W = [(F.softmax(self.conv1.weight, dim=1)).detach()]
        return H, W


class GTConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(GTConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(th.Tensor(out_channels, in_channels, 1, 1))
        self.bias = None
        self.scale = nn.Parameter(th.Tensor([0.1]), requires_grad=False)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        nn.init.constant_(self.weight, 0.1)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, A):
        x = F.softmax(self.weight, dim=1)
        A = th.sum(A * x, dim=1)
        return A
