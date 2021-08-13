import dgl
import math
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv, EdgeWeightNorm
from ..utils import transform_relation_graph_list
from . import BaseModel, register_model


@register_model('MHNF')
class MHNF(BaseModel):
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
        super(MHNF, self).__init__()
        self.num_edge_type = num_edge_type
        self.num_channels = num_channels
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_class = num_class
        self.num_layers = num_layers
        self.is_norm = norm
        self.category = category
        self.identity = identity
        # gcn_list will be used in HLHIA_layer
        self.gcn_list = nn.ModuleList()
        for i in range(num_channels):
            self.gcn_list.append(GraphConv(in_feats=self.in_dim, out_feats=hidden_dim, norm='none', activation=F.relu))
        self.HLHIA_layer = HLHIA(self.gcn_list, num_edge_type, self.num_channels, self.num_layers)
        self.channel_attention = nn.Sequential(
            nn.Linear(self.hidden_dim, 1),
            nn.Tanh(),
            nn.Linear(1, 1, bias=False),
            nn.ReLU()
        )
        self.layers_attention = nn.ModuleList()
        for i in range(num_channels):
            self.layers_attention.append(nn.Sequential(
            nn.Linear(self.hidden_dim, 1),
            nn.Tanh(),
            nn.Linear(1, 1, bias=False),
            nn.ReLU()
        ))
        self.linear = nn.Linear(self.hidden_dim, self.num_class)
        self.category_idx = None
        self.A = None
        self.h = None

    '''
        HSAF operation use two level attention mechanism
        aggregate representation list to final representation
    '''
    def HSAF(self, attention_list):
        channel_attention_list = []
        for i in range(self.num_channels):
            layer_level_feature_list = attention_list[i]
            layer_attention = self.layers_attention[i]
            for j in range(self.num_layers+1):
                layer_level_feature = layer_level_feature_list[j]
                if j == 0:
                    layer_level_alpha = layer_attention(layer_level_feature)
                else:
                    layer_level_alpha = th.cat((layer_level_alpha, layer_attention(layer_level_feature)),dim=-1)
            layer_level_beta = th.softmax(layer_level_alpha, dim=-1)
            channel_attention_list.append(th.bmm(th.stack(layer_level_feature_list, dim=-1), layer_level_beta.unsqueeze(-1)).squeeze(-1))

        for i in range(self.num_channels):
            channel_level_feature = channel_attention_list[i]
            if i == 0:
                channel_level_alpha = self.channel_attention(channel_level_feature)
            else:
                channel_level_alpha = th.cat((channel_level_alpha, self.channel_attention(channel_level_feature)), dim=-1)
        channel_level_beta = th.softmax(channel_level_alpha, dim=-1)
        channel_attention = th.bmm(th.stack(channel_attention_list, dim=-1), channel_level_beta.unsqueeze(-1)).squeeze(-1)
        return channel_attention

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
            attention_list = self.HLHIA_layer(A, h)
            final_representation = self.HSAF(attention_list)
            y = self.linear(final_representation)
            return {self.category: y[self.category_idx]}

class  HLHIA(nn.Module):
    """
        HLHIA layer record node embedding
        in all channel level and all layer level
        which will be used in HSAF to generate final representation
        by attention mechanism.
    """
    def __init__(self, gcn_list, num_edge_type, num_channels, num_layers):
        super(HLHIA, self).__init__()
        self.num_channels = num_channels
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(HMAELayer(num_edge_type, num_channels, first=True))
            else:
                layers.append(HMAELayer(num_edge_type, num_channels, first=False))
        self.layers = nn.ModuleList(layers)
        self.gcn_list = gcn_list
        self.norm = EdgeWeightNorm(norm='right')

    """
        The number of representation in channel_attention_list
        is num_channels*num_layers.
        Each representation is a tensor with dimension n*out_dim
    """
    def forward(self, A, h):
        layer_list = []
        for i in range(len(self.layers)):
            if i == 0:
                H, W, first_adj = self.layers[i](A)
                layer_list.append(first_adj)
                layer_list.append(H)
            else:
                H, W, first_adj = self.layers[i](A, H)
                layer_list.append(H)
        # * =============== GCN Encoder ================
        channel_attention_list = []
        for i in range(self.num_channels):
            gcn = self.gcn_list[i]
            layer_attention_list = []
            for j in range(len(layer_list)):
                layer = layer_list[j][i]
                edge_weight = layer.edata['w_sum']
                edge_weight = self.norm(layer, edge_weight)
                layer_attention_list.append(gcn(layer, h, edge_weight=edge_weight))
            channel_attention_list.append(layer_attention_list)
        return channel_attention_list

class HMAELayer(nn.Module):
    """
        HMAE layer is similar to GTLayer in GTN,
        but the softmax operation is added to
        hybrid adjacency matrix directly.
    """
    def __init__(self, in_channels, out_channels, first=True):
        super(HMAELayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.first = first
        self.norm = EdgeWeightNorm(norm='right')
        if self.first == True:
            self.conv1 = GTConv(in_channels, out_channels)
            self.conv2 = GTConv(in_channels, out_channels)
        else:
            self.conv1 = GTConv(in_channels, out_channels)

    """
        Perform a Softmax operation on each row of the extracted
        hybrid relationship matrix
    """
    def softmax_norm(self, H):
        norm_H = []
        for i in range(len(H)):
            g = H[i]
            g.edata['w_sum'] = self.norm(g, th.exp(g.edata['w_sum'])) # normalize the hybrid relationship matrix
            norm_H.append(g)
        return norm_H

    def forward(self, A, H_=None):
        if self.first == True:
            result_A = self.softmax_norm(self.conv1(A))
            result_B = self.softmax_norm(self.conv2(A))
            W = [self.conv1.weight.detach(), self.conv2.weight.detach()]
        else:
            result_A = H_
            result_B = self.conv1(A)
            W = [self.conv1.weight.detach().detach()]
        H = []
        for i in range(len(result_A)):
            g = dgl.adj_product_graph(result_A[i], result_B[i], 'w_sum')
            H.append(g)
        return H, W, result_A


class GTConv(nn.Module):
    """
        The method to extract hybrid relationship matrix is similar to GTN
    """
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
        filter = self.weight # If remove softmax
        num_channels = filter.shape[0]
        results = []
        for i in range(num_channels):
            for j, g in enumerate(A):
                A[j].edata['w_sum'] = g.edata['w'] * filter[i][j]
            sum_g = dgl.adj_sum_graph(A, 'w_sum')
            results.append(sum_g)
        return results