import torch
import torch.nn as nn
import dgl
from dgl.nn.pytorch import GraphConv
import torch.nn.functional as F

import dgl.function as fn
from dgl.nn.pytorch import edge_softmax, GATConv



##############################################          conv.py文件
# from conv import myGATConv

"""Torch modules for graph attention networks(GAT)."""
# pylint: disable= no-member, arguments-differ, invalid-name
import torch as th
from torch import nn

from dgl import function as fn
from dgl.nn.pytorch import edge_softmax
from dgl._ffi.base import DGLError
from dgl.nn.pytorch.utils import Identity
from dgl.utils import expand_as_pair

# pylint: enable=W0235
class myGATConv(nn.Module):
    """
    Adapted from
    https://docs.dgl.ai/_modules/dgl/nn/pytorch/conv/gatconv.html#GATConv
    """
    def __init__(self,
                 edge_feats,
                 num_etypes,
                 in_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None,
                 allow_zero_in_degree=False,
                 bias=False,
                 alpha=0.):
        super(myGATConv, self).__init__()
        self._edge_feats = edge_feats
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self.edge_emb = nn.Embedding(num_etypes, edge_feats)
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
        self.fc_e = nn.Linear(edge_feats, edge_feats*num_heads, bias=False)
        self.attn_l = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_e = nn.Parameter(th.FloatTensor(size=(1, num_heads, edge_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation
        self.bias = bias
        if bias:
            self.bias_param = nn.Parameter(th.zeros((1, num_heads, out_feats)))
        self.alpha = alpha

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        nn.init.xavier_normal_(self.attn_e, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_e.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, e_feat, res_attn=None):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')

            if isinstance(feat, tuple):
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, 'fc_src'):
                    self.fc_src, self.fc_dst = self.fc, self.fc
                feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
                feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
            else:
                h_src = h_dst = self.feat_drop(feat)
                feat_src = feat_dst = self.fc(h_src).view(
                    -1, self._num_heads, self._out_feats)
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
            e_feat = self.edge_emb(e_feat)
            e_feat = self.fc_e(e_feat).view(-1, self._num_heads, self._edge_feats)
            #print(e_feat.size())
            #print(self.attn_e.size())
            ee = (e_feat * self.attn_e).sum(dim=-1).unsqueeze(-1)
            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            #print(ee.size())
            graph.srcdata.update({'ft': feat_src, 'el': el})
            graph.dstdata.update({'er': er})
            graph.edata.update({'ee': ee})
            graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
            e = self.leaky_relu(graph.edata.pop('e')+graph.edata.pop('ee'))
            #################################
            # compute softmax
            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
            if res_attn is not None:
                graph.edata['a'] = graph.edata['a'] * (1-self.alpha) + res_attn * self.alpha
            # message passing
            graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                             fn.sum('m', 'ft'))
            rst = graph.dstdata['ft']
            # residual
            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
                rst = rst + resval
            # bias
            if self.bias:
                rst = rst + self.bias_param
            # activation
            if self.activation:
                rst = self.activation(rst)
            return rst, graph.edata.pop('a').detach()



###########################################

class myGAT(nn.Module):
    def __init__(self,
                 g,
                 edge_dim,
                 num_etypes,
                 in_dims,
                 num_hidden,
                 num_classes,
                 num_layers,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual,
                 alpha):
        super(myGAT, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        self.fc_list = nn.ModuleList([nn.Linear(in_dim, num_hidden, bias=True) for in_dim in in_dims])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)
        # input projection (no residual)
        self.gat_layers.append(myGATConv(edge_dim, num_etypes,
            num_hidden, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation, alpha=alpha))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(myGATConv(edge_dim, num_etypes,
                num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation, alpha=alpha))
        # output projection
        self.gat_layers.append(myGATConv(edge_dim, num_etypes,
            num_hidden * heads[-2], num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None, alpha=alpha))
        self.epsilon = torch.FloatTensor([1e-12]).cuda()


    def forward(self, features_list, e_feat):
        h = []
        for fc, feature in zip(self.fc_list, features_list):
            h.append(fc(feature))
        h = torch.cat(h, 0)
        res_attn = None
        for l in range(self.num_layers):
            h, res_attn = self.gat_layers[l](self.g, h, e_feat, res_attn=res_attn)
            h = h.flatten(1)
        # output projection
        logits, _ = self.gat_layers[-1](self.g, h, e_feat, res_attn=None)
        logits = logits.mean(1)
        logits = logits / (torch.max(torch.norm(logits, dim=1, keepdim=True), self.epsilon))
        return logits

class GCN(nn.Module):
    def __init__(self,
                 g,
                 in_dims,
                 num_hidden,
                 num_classes,
                 num_layers,
                 activation,
                 dropout):
        super(GCN, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        self.fc_list = nn.ModuleList([nn.Linear(in_dim, num_hidden, bias=True) for in_dim in in_dims])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)
        # input layer
        self.layers.append(GraphConv(num_hidden, num_hidden, activation=activation, weight=False))
        # hidden layers
        for i in range(num_layers - 1):
            self.layers.append(GraphConv(num_hidden, num_hidden, activation=activation))
        # output layer
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features_list):
        h = []
        for fc, feature in zip(self.fc_list, features_list):
            h.append(fc(feature))
        h = torch.cat(h, 0)
        for i, layer in enumerate(self.layers):
            h = self.dropout(h)
            h = layer(self.g, h)
        return h

class acm_hGCN(nn.Module):
    def __init__(self,
                 g,
                 in_dims,
                 num_hidden,
                 num_classes,
                 num_layers,
                 activation,
                 dropout):
        super(acm_hGCN, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        self.fc_list = nn.ModuleList([nn.Linear(in_dim, num_hidden, bias=True) for in_dim in in_dims])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)
        # input layer
        self.layers.append(GraphConv(num_hidden, num_hidden, activation=activation, weight=False))
        # hidden layers
        for i in range(num_layers - 1):
            self.layers.append(GraphConv(num_hidden, num_hidden, activation=activation))
        # output layer
        self.dropout = nn.Dropout(p=dropout)
        self.sigmoid= nn.Sigmoid()
    def message_func(self,edges):
        #all type subgraph
        res=edges.src['ft']
        #type0 subgraph
        _mask = edges.data['e'] == 0
        # print(res.size())
        # print(_mask.size())
        mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft0'], torch.zeros_like(edges.src['ft']))
        res0=msg
        #type2 subgraph
        _mask = edges.data['e'] == 2
        mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft2'], torch.zeros_like(edges.src['ft']))
        res2=msg
        #type4 subgraph
        _mask = edges.data['e'] == 4
        mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft4'], torch.zeros_like(edges.src['ft']))
        res4=msg
        #type6 subgraph
        _mask = edges.data['e'] == 6
        mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft6'], torch.zeros_like(edges.src['ft']))
        res6=msg
        return {'m': res+res0+res2+res4+res6}
    def forward(self, graph, features_list, e_feat):
        e_feat=e_feat.reshape(-1,1).float()
        h = []
        for fc, feature in zip(self.fc_list, features_list):
            h.append(fc(feature))
        h = torch.cat(h, 0)
        for i, layer in enumerate(self.layers):
            h = self.dropout(h)
            h = layer(self.g, h)
        # h=self.sigmoid(h)
        graph.edata.update({'e':e_feat})
        graph.srcdata.update({'ft': h,'ft0': h,'ft2': h,'ft4': h,'ft6': h})
        graph.update_all(self.message_func, fn.sum('m', 'ft'))
        res = graph.dstdata['ft']
        return res



class acm_hGCN_each_loss(nn.Module):
    def __init__(self,
                 g,
                 in_dims,
                 num_hidden,
                 num_classes,
                 num_layers,
                 activation,
                 dropout):
        super(acm_hGCN_each_loss, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        self.fc_list = nn.ModuleList([nn.Linear(in_dim, num_hidden, bias=True) for in_dim in in_dims])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)
        # input layer
        self.layers.append(GraphConv(num_hidden, num_hidden, activation=activation, weight=False))
        # hidden layers
        for i in range(num_layers - 1):
            self.layers.append(GraphConv(num_hidden, num_hidden, activation=activation))
        # output layer
        self.dropout = nn.Dropout(p=dropout)
        self.sigmoid= nn.Sigmoid()
    def message_func(self,edges):
        res=edges.src['ft']
        return {'m': res}
    def message_func0(self,edges):
        #type0 subgraph
        _mask = edges.data['e'] == 0
        # print(res.size())
        # print(_mask.size())
        mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft0'], torch.zeros_like(edges.src['ft']))
        return {'m': msg}
    def message_func2(self,edges):
        #type0 subgraph
        _mask = edges.data['e'] == 2
        # print(res.size())
        # print(_mask.size())
        mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft0'], torch.zeros_like(edges.src['ft']))
        return {'m': msg}
    def message_func4(self,edges):
        #type0 subgraph
        _mask = edges.data['e'] == 4
        # print(res.size())
        # print(_mask.size())
        mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft0'], torch.zeros_like(edges.src['ft']))
        return {'m': msg}
    def message_func6(self,edges):
        #type0 subgraph
        _mask = edges.data['e'] == 6
        # print(res.size())
        # print(_mask.size())
        mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft0'], torch.zeros_like(edges.src['ft']))
        return {'m': msg}
    def forward(self, graph, features_list, e_feat):
        e_feat=e_feat.reshape(-1,1).float()
        h = []
        for fc, feature in zip(self.fc_list, features_list):
            h.append(fc(feature))
        h = torch.cat(h, 0)
        for i, layer in enumerate(self.layers):
            h = self.dropout(h)
            h = layer(self.g, h)
        # h=self.sigmoid(h)
        graph.edata.update({'e':e_feat})
        graph.srcdata.update({'ft': h,'ft0': h,'ft2': h,'ft4': h,'ft6': h})
        graph.update_all(self.message_func, fn.sum('m', 'ft'))
        graph.update_all(self.message_func0, fn.sum('m', 'ft0'))
        graph.update_all(self.message_func0, fn.sum('m', 'ft2'))
        graph.update_all(self.message_func0, fn.sum('m', 'ft4'))
        graph.update_all(self.message_func0, fn.sum('m', 'ft6'))
        res = graph.dstdata['ft']
        res0 = graph.dstdata['ft0']
        res2 = graph.dstdata['ft2']
        res4 = graph.dstdata['ft4']
        res6 = graph.dstdata['ft6']
        return res,res0,res2,res4,res6


class acm_sem_hGCN(nn.Module):
    def __init__(self,
                 g,
                 in_dims,
                 num_hidden,
                 num_classes,
                 num_layers,
                 activation,
                 dropout):
        super(acm_sem_hGCN, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        self.fc_list = nn.ModuleList([nn.Linear(in_dim, num_hidden, bias=True) for in_dim in in_dims])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)
        # input layer
        self.layers.append(GraphConv(num_hidden, num_hidden, activation=activation, weight=False))
        # hidden layers
        for i in range(num_layers - 1):
            self.layers.append(GraphConv(num_hidden, num_hidden, activation=activation))
        self.semantic_weight= torch.nn.Parameter(torch.Tensor(1,5))
        self.dropout = nn.Dropout(p=dropout)
        self.reset_parameters()
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.semantic_weight)
    def message_func(self,edges):
        #all type subgraph
        res=edges.src['ft']*self.semantic_weight[0,0]
        #type0 subgraph
        _mask = edges.data['e'] == 0
        # print(res.size())
        # print(_mask.size())
        mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft0'], torch.zeros_like(edges.src['ft']))
        res0=msg*self.semantic_weight[0,1]
        #type2 subgraph
        _mask = edges.data['e'] == 2
        mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft2'], torch.zeros_like(edges.src['ft']))
        res2=msg*self.semantic_weight[0,2]
        #type4 subgraph
        _mask = edges.data['e'] == 4
        mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft4'], torch.zeros_like(edges.src['ft']))
        res4=msg*self.semantic_weight[0,3]
        #type6 subgraph
        _mask = edges.data['e'] == 6
        mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft6'], torch.zeros_like(edges.src['ft']))
        res6=msg*self.semantic_weight[0,4]
        return {'m': res+res0+res2+res4+res6}
    def forward(self, graph, features_list, e_feat):
        e_feat=e_feat.reshape(-1,1).float()
        h = []
        for fc, feature in zip(self.fc_list, features_list):
            h.append(fc(feature))
        h = torch.cat(h, 0)
        for i, layer in enumerate(self.layers):
            h = self.dropout(h)
            h = layer(self.g, h)
        graph.edata.update({'e':e_feat})
        graph.srcdata.update({'ft': h,'ft0': h,'ft2': h,'ft4': h,'ft6': h})
        graph.update_all(self.message_func, fn.sum('m', 'ft'))
        res = graph.dstdata['ft']
        return res

class semantic_GCN(nn.Module):
    def __init__(self,
                 g,
                 in_dims,
                 num_hidden,
                 num_classes,
                 num_layers,
                 activation,
                 dropout):
        super(semantic_GCN, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        self.fc_list = nn.ModuleList([nn.Linear(in_dim, num_hidden, bias=True) for in_dim in in_dims])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)
        # input layer
        self.layers.append(GraphConv(num_hidden, num_hidden, activation=activation, weight=False))
        # hidden layers
        for i in range(num_layers - 1):
            self.layers.append(GraphConv(num_hidden, num_hidden, activation=activation))
        self.semantic_weight= torch.nn.Parameter(torch.Tensor(1,5))
        # output layer
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features_list):
        h = []
        for fc, feature in zip(self.fc_list, features_list):
            h.append(fc(feature))
        h = torch.cat(h, 0)
        for i, layer in enumerate(self.layers):
            h = self.dropout(h)
            h = layer(self.g, h)
        return h,self.semantic_weight

#book as source
class freebase_source_hGCN(nn.Module):
    def __init__(self,
                 g,
                 in_dims,
                 num_hidden,
                 num_classes,
                 num_layers,
                 activation,
                 dropout):
        super(freebase_source_hGCN, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        self.fc_list = nn.ModuleList([nn.Linear(in_dim, num_hidden, bias=True) for in_dim in in_dims])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)
        # input layer
        self.layers.append(GraphConv(num_hidden, num_hidden, activation=activation, weight=False))
        # hidden layers
        for i in range(num_layers - 1):
            self.layers.append(GraphConv(num_hidden, num_hidden, activation=activation))
        # output layer
        self.dropout = nn.Dropout(p=dropout)
        self.sigmoid= nn.Sigmoid()
    def message_func(self,edges):
        #all type subgraph
        res=edges.src['ft']
        #type0 subgraph
        _mask = edges.data['e'] == 0
        # print(res.size())
        # print(_mask.size())
        mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft0'], torch.zeros_like(edges.src['ft']))
        res0=msg
        #type2 subgraph
        _mask = edges.data['e'] == 1
        mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft1'], torch.zeros_like(edges.src['ft']))
        res1=msg
        #type4 subgraph
        _mask = edges.data['e'] == 2
        mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft2'], torch.zeros_like(edges.src['ft']))
        res2=msg
        #type6 subgraph
        _mask = edges.data['e'] == 3
        mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft3'], torch.zeros_like(edges.src['ft']))
        res3=msg
        _mask = edges.data['e'] == 4
        mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft4'], torch.zeros_like(edges.src['ft']))
        res4=msg
        return {'m': res+res0+res1+res2+res3+res4}
    def forward(self, graph, features_list, e_feat):
        e_feat=e_feat.reshape(-1,1).float()
        h = []
        for fc, feature in zip(self.fc_list, features_list):
            h.append(fc(feature))
        h = torch.cat(h, 0)
        for i, layer in enumerate(self.layers):
            h = self.dropout(h)
            h = layer(self.g, h)
        # h=self.sigmoid(h)
        graph.edata.update({'e':e_feat})
        graph.srcdata.update({'ft': h,'ft0': h,'ft1': h,'ft2': h,'ft3': h,'ft4': h})
        graph.update_all(self.message_func, fn.sum('m', 'ft'))
        res = graph.dstdata['ft']
        return res


#book as destination
class freebase_des_hGCN(nn.Module):
    def __init__(self,
                 g,
                 in_dims,
                 num_hidden,
                 num_classes,
                 num_layers,
                 activation,
                 dropout):
        super(freebase_des_hGCN, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        self.fc_list = nn.ModuleList([nn.Linear(in_dim, num_hidden, bias=True) for in_dim in in_dims])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)
        # input layer
        self.layers.append(GraphConv(num_hidden, num_hidden, activation=activation, weight=False))
        # hidden layers
        for i in range(num_layers - 1):
            self.layers.append(GraphConv(num_hidden, num_hidden, activation=activation))
        # output layer
        self.dropout = nn.Dropout(p=dropout)
        self.sigmoid= nn.Sigmoid()
    def message_func(self,edges):
        #all type subgraph
        res=edges.src['ft']
        #type0 subgraph
        _mask = edges.data['e'] == 0
        # print(res.size())
        # print(_mask.size())
        mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft0'], torch.zeros_like(edges.src['ft']))
        res0=msg
        #type2 subgraph
        _mask = edges.data['e'] == 6
        mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft1'], torch.zeros_like(edges.src['ft']))
        res1=msg
        #type4 subgraph
        _mask = edges.data['e'] == 14
        mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft2'], torch.zeros_like(edges.src['ft']))
        res2=msg
        #type6 subgraph
        _mask = edges.data['e'] == 30
        mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft3'], torch.zeros_like(edges.src['ft']))
        res3=msg
        return {'m': res+res0+res1+res2+res3}
    def forward(self, graph, features_list, e_feat):
        e_feat=e_feat.reshape(-1,1).float()
        h = []
        for fc, feature in zip(self.fc_list, features_list):
            h.append(fc(feature))
        h = torch.cat(h, 0)
        for i, layer in enumerate(self.layers):
            h = self.dropout(h)
            h = layer(self.g, h)
        # h=self.sigmoid(h)
        graph.edata.update({'e':e_feat})
        graph.srcdata.update({'ft': h,'ft0': h,'ft1': h,'ft2': h,'ft3': h})
        graph.update_all(self.message_func, fn.sum('m', 'ft'))
        res = graph.dstdata['ft']
        return res

#freebase as source and des
class freebase_bi_hGCN(nn.Module):
    def __init__(self,
                 g,
                 in_dims,
                 num_hidden,
                 num_classes,
                 num_layers,
                 activation,
                 dropout):
        super(freebase_bi_hGCN, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        self.fc_list = nn.ModuleList([nn.Linear(in_dim, num_hidden, bias=True) for in_dim in in_dims])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)
        # input layer
        self.layers.append(GraphConv(num_hidden, num_hidden, activation=activation, weight=False))
        # hidden layers
        for i in range(num_layers - 1):
            self.layers.append(GraphConv(num_hidden, num_hidden, activation=activation))
        # output layer
        self.dropout = nn.Dropout(p=dropout)
        self.sigmoid= nn.Sigmoid()
    #book as source and 1hop subgraph
    def message_func(self,edges):
        #all type subgraph
        res=edges.src['ft']
        #type0 subgraph
        _mask = edges.data['e'] == 0
        # print(res.size())
        # print(_mask.size())
        mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft0'], torch.zeros_like(edges.src['ft']))
        res0=msg
        #type2 subgraph
        _mask = edges.data['e'] == 1
        mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft1'], torch.zeros_like(edges.src['ft']))
        res1=msg
        #type4 subgraph
        _mask = edges.data['e'] == 2
        mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft2'], torch.zeros_like(edges.src['ft']))
        res2=msg
        #type6 subgraph
        _mask = edges.data['e'] == 3
        mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft3'], torch.zeros_like(edges.src['ft']))
        res3=msg
        _mask = edges.data['e'] == 4
        mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft4'], torch.zeros_like(edges.src['ft']))
        res4=msg
        return {'m': res+res0+res1+res2+res3+res4}

    #book as destination
    def message_func0(self,edges):
        #type2 subgraph
        _mask = edges.data['e'] == 6
        mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft6'], torch.zeros_like(edges.src['ft']))
        res1=msg
        #type4 subgraph
        _mask = edges.data['e'] == 14
        mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft14'], torch.zeros_like(edges.src['ft']))
        res2=msg
        #type6 subgraph
        _mask = edges.data['e'] == 30
        mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft30'], torch.zeros_like(edges.src['ft']))
        res3=msg
        return {'m': res1+res2+res3}
    def forward(self, graph, trans_graph, features_list, e_feat):
        e_feat=e_feat.reshape(-1,1).float()
        h = []
        for fc, feature in zip(self.fc_list, features_list):
            h.append(fc(feature))
        h = torch.cat(h, 0)
        for i, layer in enumerate(self.layers):
            h = self.dropout(h)
            h = layer(self.g, h)
        # h=self.sigmoid(h)
        graph.edata.update({'e':e_feat})
        graph.srcdata.update({'ft': h,'ft0': h,'ft1': h,'ft2': h,'ft3': h,'ft4': h})
        graph.update_all(self.message_func, fn.sum('m', 'ft'))
        trans_graph.edata.update({'e':e_feat})
        trans_graph.srcdata.update({'ft': h,'ft6': h,'ft14': h,'ft30': h})
        trans_graph.update_all(self.message_func0, fn.sum('m', 'ft'))
        res = graph.dstdata['ft']+trans_graph.dstdata['ft']
        return res



class GAT(nn.Module):
    def __init__(self,
                 g,
                 in_dims,
                 num_hidden,
                 num_classes,
                 num_layers,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual):
        super(GAT, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        self.fc_list = nn.ModuleList([nn.Linear(in_dim, num_hidden, bias=True) for in_dim in in_dims])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)
        # input projection (no residual)
        self.gat_layers.append(GATConv(
            num_hidden, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation))
        # output projection
        # self.gat_layers.append(GATConv(
        #     num_hidden * heads[-2], num_classes, heads[-1],
        #     feat_drop, attn_drop, negative_slope, residual, None))
        self.gat_layers.append(GATConv(
            num_hidden * heads[-2], num_hidden, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None))


    def forward(self, features_list):
        h = []
        for fc, feature in zip(self.fc_list, features_list):
            h.append(fc(feature))
        h = torch.cat(h, 0)
        for l in range(self.num_layers):
            h = self.gat_layers[l](self.g, h).flatten(1)
        # output projection
        logits = self.gat_layers[-1](self.g, h).mean(1)
        return logits



class dblp_hGAT(nn.Module):
    def __init__(self,
                 g,
                 in_dims,
                 num_hidden,
                 num_classes,
                 num_layers,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual):
        super(dblp_hGAT, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        self.fc_list = nn.ModuleList([nn.Linear(in_dim, num_hidden, bias=True) for in_dim in in_dims])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)
        self.gat_layers.append(GATConv(
            num_hidden, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation))
        # output projection
        # self.gat_layers.append(GATConv(
        #     num_hidden * heads[-2], num_classes, heads[-1],
        #     feat_drop, attn_drop, negative_slope, residual, None))
        self.gat_layers.append(GATConv(
            num_hidden * heads[-2], num_hidden, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None))
    def message_func_onehop(self,edges):
        _mask = edges.data['e'] == 3
        mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft_twohop'], torch.zeros_like(edges.src['ft']))
        #paper->author
        res0=msg
        return {'m': res0}

    def message_func_twohop(self, edges):
        # type0 subgraph
        res=edges.src['ft']
        _mask = edges.data['e'] == 0
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft0'], torch.zeros_like(edges.src['ft']))
        #author->paper->author
        res0 = msg
        # type2 subgraph
        _mask = edges.data['e'] == 4
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft4'], torch.zeros_like(edges.src['ft']))
        #term->paper->author
        res1 = msg
        # type4 subgraph
        _mask = edges.data['e'] == 5
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft5'], torch.zeros_like(edges.src['ft']))
        #venue->paper->author
        res2 = msg
        return {'m': res+ res0 + res1 + res2}
    def forward(self, graph, features_list, e_feat):
        e_feat=e_feat.reshape(-1,1).float()
        h = []
        for fc, feature in zip(self.fc_list, features_list):
            h.append(fc(feature))
        h = torch.cat(h, 0)
        for l in range(self.num_layers):
            h = self.gat_layers[l](self.g, h).flatten(1)
        # output projection
        h = self.gat_layers[-1](self.g, h).mean(1)
        # h=self.sigmoid(h)
        graph.edata.update({'e':e_feat})
        graph.srcdata.update({'ft': h,'ft0': h,'ft3': h,'ft4': h,'ft5': h})
        graph.update_all(self.message_func_twohop, fn.sum('m', 'ft_twohop'))
        graph.srcdata.update({'ft_twohop': graph.dstdata['ft_twohop']})
        graph.update_all(self.message_func_onehop, fn.sum('m', 'ft_onehop'))
        res = graph.dstdata['ft_onehop']
        return res


class GIN(nn.Module):
    def __init__(self,
                 g,
                 in_dims,
                 num_hidden,
                 num_classes,
                 num_layers,
                 activation,
                 dropout):
        super(GIN, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.act=torch.nn.ReLU()
        self.activation=activation
        self.fc_list = nn.ModuleList([nn.Linear(in_dim, num_hidden, bias=True) for in_dim in in_dims])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)
        # input layer
        self.ginfunc=torch.nn.Sequential(torch.nn.Linear(num_hidden, num_hidden), self.act, torch.nn.Linear(num_hidden, num_hidden))
        self.GINlayer=dgl.nn.pytorch.conv.GINConv(apply_func=self.ginfunc,aggregator_type='sum')
        self.bn=torch.nn.BatchNorm1d(num_hidden)
        self.layers.append(self.GINlayer)
        self.bn_layers.append(self.bn)
        # hidden layers
        for i in range(num_layers - 1):
            self.layers.append(self.GINlayer)
            self.bn_layers.append(self.bn)
        # output layer
        # self.output_ginfunc=torch.nn.Sequential(torch.nn.Linear(num_hidden, num_classes), self.act, torch.nn.Linear(num_classes, num_classes))
        # self.output_GINlayer=dgl.nn.pytorch.conv.GINConv(apply_func=self.output_ginfunc,aggregator_type='sum')
        # self.output_bn=torch.nn.BatchNorm1d(num_classes)
        # self.layers.append(self.output_GINlayer)
        # self.bn_layers.append(self.output_bn)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features_list):
        h = []
        for fc, feature in zip(self.fc_list, features_list):
            h.append(fc(feature))
        h = torch.cat(h, 0)
        for i, layer in enumerate(self.layers):
            h = self.dropout(h)
            h = layer(self.g, h)
            h=self.activation(h)
            h=self.bn_layers[i](h)
        return h






################################            hprompt.py   文件
    
import torch
import torch.nn as nn
from dgl import function as fn
import dgl
import torch.nn.functional as F

class node_prompt_layer_feature_weighted_sum(nn.Module):
    def __init__(self,input_dim):
        super(node_prompt_layer_feature_weighted_sum, self).__init__()
        self.weight= torch.nn.Parameter(torch.Tensor(1,input_dim))
        # self.dropout = nn.Dropout(p=0.2)
        self.reset_parameters()
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
    def forward(self, graph, graph_embedding):
        # emb=graph_embedding*self.weight
        emb=F.elu(graph_embedding*self.weight)
        # emb = self.dropout(emb)
        graph.srcdata.update({'ft': emb})
        graph.update_all(fn.copy_u('ft','m'), fn.sum('m', 'ft'))
        return graph.dstdata['ft']

class freebase_node_prompt_layer_feature_weighted_sum(nn.Module):
    def __init__(self,input_dim, g):
        super(freebase_node_prompt_layer_feature_weighted_sum, self).__init__()
        self.weight= torch.nn.Parameter(torch.Tensor(1,input_dim))
        # self.dropout = nn.Dropout(p=0.2)
        self.r_graph=dgl.reverse(g)
        self.reset_parameters()
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
    def forward(self, graph, graph_embedding):
        # emb=graph_embedding*self.weight
        emb=F.elu(graph_embedding*self.weight)
        # emb = self.dropout(emb)
        r_graph=self.r_graph
        graph.srcdata.update({'ft': emb})
        graph.update_all(fn.copy_u('ft','m'), fn.sum('m', 'ft'))
        r_graph.srcdata.update({'ft': emb})
        r_graph.update_all(fn.copy_u('ft','m'), fn.sum('m', 'ft'))
        return graph.dstdata['ft']+r_graph.dstdata['ft']


class node_bottle_net(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim):
        super(node_bottle_net, self).__init__()
        # self.weight0= torch.nn.Parameter(torch.Tensor(input_dim,hidden_dim))
        # self.weight1= torch.nn.Parameter(torch.Tensor(hidden_dim,output_dim))
        self.linear0=torch.nn.Linear(input_dim,hidden_dim)
        self.linear1=torch.nn.Linear(hidden_dim,output_dim)
        # self.reset_parameters()
    # def reset_parameters(self):
    #     torch.nn.init.xavier_uniform_(self.weight0)
    #     torch.nn.init.xavier_uniform_(self.weight1)
    def message_func(edges):
        return {'m': edges.dst['h']}
    def forward(self, graph, graph_embedding):
        # emb=graph_embedding*self.weight
        # emb=F.elu(torch.matmul(torch.matmul(graph_embedding,self.weight0),self.weight1))
        emb=F.elu(self.linear1(self.linear0(graph_embedding)))
        # emb = self.dropout(emb)
        graph.srcdata.update({'ft': emb})
        graph.dstdata.update({'ft':emb})
        graph.update_all(fn.copy_u('ft','m'), fn.sum('m', 'ft'))
        return graph.dstdata['ft']

class hnode_prompt_layer_feature_weighted_sum(nn.Module):
    def __init__(self,
                 input_dim,
                 negative_slope=0.2
                 ):
        super(hnode_prompt_layer_feature_weighted_sum, self).__init__()
        self.weight= torch.nn.Parameter(torch.Tensor(1,input_dim))
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.reset_parameters()
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
    def message_func(self,edges):
        return {'m': torch.cat((edges.src['ft'],edges.data['e']),dim=1)}
    def forward(self, graph, graph_embedding,e_feat):
        e_feat=e_feat.reshape(-1,1).float()
        emb=graph_embedding*self.weight
        graph.srcdata.update({'ft': emb})
        graph.edata.update({'e': e_feat})
        graph.update_all(self.message_func, fn.sum('m', 'ft'))
        res = graph.dstdata['ft']
        return res

class acm_hnode_prompt_layer_feature_weighted_sum(nn.Module):
    def __init__(self,
                 input_dim,
                 pre_semantic_weight=None,
                 negative_slope=0.2,
                 ):
        super(acm_hnode_prompt_layer_feature_weighted_sum, self).__init__()
        self.weight= torch.nn.Parameter(torch.Tensor(1,input_dim))
        self.weight0= torch.nn.Parameter(torch.Tensor(1,input_dim))
        self.weight2= torch.nn.Parameter(torch.Tensor(1,input_dim))
        self.weight4= torch.nn.Parameter(torch.Tensor(1,input_dim))
        self.weight6= torch.nn.Parameter(torch.Tensor(1,input_dim))
        self.pre_semantic_weight=pre_semantic_weight
        self.semantic_weight= torch.nn.Parameter(torch.Tensor(1,5))
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.reset_parameters()
    def weighted_sum(self, input, weights):
        weights = weights.expand(input.permute(1, 2, 0).shape).permute(2, 0, 1)
        weighted_inputs = input * weights
        output = weighted_inputs
        return output

    def message_func(self,edges):
        #all type subgraph
        res=edges.src['ft']
        #type0 subgraph
        _mask = edges.data['e'] == 0
        mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft0'], torch.zeros_like(edges.src['ft']))
        res0=msg
        #type2 subgraph
        _mask = edges.data['e'] == 2
        mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft2'], torch.zeros_like(edges.src['ft']))
        res2=msg
        #type4 subgraph
        _mask = edges.data['e'] == 4
        mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft4'], torch.zeros_like(edges.src['ft']))
        res4=msg
        #type6 subgraph
        _mask = edges.data['e'] == 6
        mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft6'], torch.zeros_like(edges.src['ft']))
        res6=msg
        return {'m': res+res0+res2+res4+res6}



    #add pre semantic prompt
    # def message_func(self,edges):
    #     #all type subgraph
    #     semantic=self.pre_semantic_weight
    #     semantic=F.normalize(semantic, p=2, dim=1)
    #     res=edges.src['ft']*semantic[0,0]
    #     #type0 subgraph
    #     _mask = edges.data['e'] == 0
    #     mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
    #     msg = torch.where(mask, edges.src['ft0'], torch.zeros_like(edges.src['ft']))
    #     res0=msg*semantic[0,1]
    #     #type2 subgraph
    #     _mask = edges.data['e'] == 2
    #     mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
    #     msg = torch.where(mask, edges.src['ft2'], torch.zeros_like(edges.src['ft']))
    #     res2=msg*semantic[0,2]
    #     #type4 subgraph
    #     _mask = edges.data['e'] == 4
    #     mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
    #     msg = torch.where(mask, edges.src['ft4'], torch.zeros_like(edges.src['ft']))
    #     res4=msg*semantic[0,3]
    #     #type6 subgraph
    #     _mask = edges.data['e'] == 6
    #     mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
    #     msg = torch.where(mask, edges.src['ft6'], torch.zeros_like(edges.src['ft']))
    #     res6=msg*semantic[0,4]
    #     return {'m': res+res0+res2+res4+res6}

    #add semantic prompt
    # def message_func(self,edges):
    #     #all type subgraph
    #     semantic=self.semantic_weight
    #     semantic=F.normalize(semantic, p=2, dim=1)
    #     res=edges.src['ft']*semantic[0,0]
    #     #type0 subgraph
    #     _mask = edges.data['e'] == 0
    #     mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
    #     msg = torch.where(mask, edges.src['ft0'], torch.zeros_like(edges.src['ft']))
    #     res0=msg*semantic[0,1]
    #     #type2 subgraph
    #     _mask = edges.data['e'] == 2
    #     mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
    #     msg = torch.where(mask, edges.src['ft2'], torch.zeros_like(edges.src['ft']))
    #     res2=msg*semantic[0,2]
    #     #type4 subgraph
    #     _mask = edges.data['e'] == 4
    #     mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
    #     msg = torch.where(mask, edges.src['ft4'], torch.zeros_like(edges.src['ft']))
    #     res4=msg*semantic[0,3]
    #     #type6 subgraph
    #     _mask = edges.data['e'] == 6
    #     mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
    #     msg = torch.where(mask, edges.src['ft6'], torch.zeros_like(edges.src['ft']))
    #     res6=msg*semantic[0,4]
    #     return {'m': res+res0+res2+res4+res6}

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        torch.nn.init.xavier_uniform_(self.weight0)
        torch.nn.init.xavier_uniform_(self.weight2)
        torch.nn.init.xavier_uniform_(self.weight4)
        torch.nn.init.xavier_uniform_(self.weight6)
        torch.nn.init.xavier_uniform_(self.semantic_weight)

    def forward(self, graph, graph_embedding,e_feat):
        e_feat=e_feat.reshape(-1,1).float()

        # emb=graph_embedding*self.weight
        # emb0=graph_embedding*self.weight0
        # emb2=graph_embedding*self.weight2
        # emb4=graph_embedding*self.weight4
        # emb6=graph_embedding*self.weight6
        emb=emb0=emb2=emb4=emb6=F.elu(graph_embedding*self.weight)

        graph.edata.update({'e':e_feat})
        graph.srcdata.update({'ft': emb,'ft0': emb0,'ft2': emb2,'ft4': emb4,'ft6': emb6})
        graph.update_all(self.message_func, fn.sum('m', 'ft'))
        res = graph.dstdata['ft']
        # res=F.sigmoid(res)
        return res

#acm semantic prompt
class acm_hnode_semantic_prompt_layer_feature_weighted_sum(nn.Module):
    def __init__(self,
                 input_dim,
                 pre_semantic_weight=None,
                 negative_slope=0.2,
                 semantic_prompt_weight=0.1
                 ):
        super(acm_hnode_semantic_prompt_layer_feature_weighted_sum, self).__init__()
        self.weight= torch.nn.Parameter(torch.Tensor(1,input_dim))
        self.weight0= torch.nn.Parameter(torch.Tensor(1,input_dim))
        self.weight2= torch.nn.Parameter(torch.Tensor(1,input_dim))
        self.weight4= torch.nn.Parameter(torch.Tensor(1,input_dim))
        self.weight6= torch.nn.Parameter(torch.Tensor(1,input_dim))
        self.pre_semantic_weight=pre_semantic_weight
        self.semantic_weight= torch.nn.Parameter(torch.Tensor(1,5))
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.semantic_prompt_weight=semantic_prompt_weight
        self.reset_parameters()
    def weighted_sum(self, input, weights):
        weights = weights.expand(input.permute(1, 2, 0).shape).permute(2, 0, 1)
        weighted_inputs = input * weights
        output = weighted_inputs
        return output

    def message_func(self,edges):
        #all type subgraph
        res=edges.src['ft']
        #type0 subgraph
        _mask = edges.data['e'] == 0
        mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft0'], torch.zeros_like(edges.src['ft']))
        res0=msg
        #type2 subgraph
        _mask = edges.data['e'] == 2
        mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft2'], torch.zeros_like(edges.src['ft']))
        res2=msg
        #type4 subgraph
        _mask = edges.data['e'] == 4
        mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft4'], torch.zeros_like(edges.src['ft']))
        res4=msg
        #type6 subgraph
        _mask = edges.data['e'] == 6
        mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft6'], torch.zeros_like(edges.src['ft']))
        res6=msg
        return {'m': res+res0+res2+res4+res6}

    #add pre semantic prompt
    # def message_func(self,edges):
    #     #all type subgraph
    #     semantic=self.pre_semantic_weight
    #     semantic=F.normalize(semantic, p=2, dim=1)
    #     res=edges.src['ft']*semantic[0,0]
    #     #type0 subgraph
    #     _mask = edges.data['e'] == 0
    #     mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
    #     msg = torch.where(mask, edges.src['ft0'], torch.zeros_like(edges.src['ft']))
    #     res0=msg*semantic[0,1]
    #     #type2 subgraph
    #     _mask = edges.data['e'] == 2
    #     mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
    #     msg = torch.where(mask, edges.src['ft2'], torch.zeros_like(edges.src['ft']))
    #     res2=msg*semantic[0,2]
    #     #type4 subgraph
    #     _mask = edges.data['e'] == 4
    #     mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
    #     msg = torch.where(mask, edges.src['ft4'], torch.zeros_like(edges.src['ft']))
    #     res4=msg*semantic[0,3]
    #     #type6 subgraph
    #     _mask = edges.data['e'] == 6
    #     mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
    #     msg = torch.where(mask, edges.src['ft6'], torch.zeros_like(edges.src['ft']))
    #     res6=msg*semantic[0,4]
    #     return {'m': res+res0+res2+res4+res6}

    # add semantic prompt
    # def message_func_semantic(self,edges):
    #     #all type subgraph
    #     semantic=self.semantic_weight
    #     semantic=F.normalize(semantic, p=2, dim=1)
    #     res=edges.src['ft']*semantic[0,0]
    #     #type0 subgraph
    #     _mask = edges.data['e'] == 0
    #     mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
    #     msg = torch.where(mask, edges.src['ft0'], torch.zeros_like(edges.src['ft']))
    #     res0=msg*(1+self.semantic_prompt_weight*semantic[0,1])
    #     #type2 subgraph
    #     _mask = edges.data['e'] == 2
    #     mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
    #     msg = torch.where(mask, edges.src['ft2'], torch.zeros_like(edges.src['ft']))
    #     res2=msg*(1+self.semantic_prompt_weight*semantic[0,2])
    #     #type4 subgraph
    #     _mask = edges.data['e'] == 4
    #     mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
    #     msg = torch.where(mask, edges.src['ft4'], torch.zeros_like(edges.src['ft']))
    #     res4=msg*(1+self.semantic_prompt_weight*semantic[0,3])
    #     #type6 subgraph
    #     _mask = edges.data['e'] == 6
    #     mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
    #     msg = torch.where(mask, edges.src['ft6'], torch.zeros_like(edges.src['ft']))
    #     res6=msg*(1+self.semantic_prompt_weight*semantic[0,4])
    #     return {'m': res+res0+res2+res4+res6}

    def message_func_semantic(self,edges):
        #all type subgraph
        semantic=self.semantic_weight
        semantic=F.normalize(semantic, p=2, dim=1)
        res=edges.src['ft']*semantic[0,0]
        #type0 subgraph
        _mask = edges.data['e'] == 0
        mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft0'], torch.zeros_like(edges.src['ft']))
        res0=msg*semantic[0,1]
        #type2 subgraph
        _mask = edges.data['e'] == 2
        mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft2'], torch.zeros_like(edges.src['ft']))
        res2=msg*semantic[0,2]
        #type4 subgraph
        _mask = edges.data['e'] == 4
        mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft4'], torch.zeros_like(edges.src['ft']))
        res4=msg*semantic[0,3]
        #type6 subgraph
        _mask = edges.data['e'] == 6
        mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft6'], torch.zeros_like(edges.src['ft']))
        res6=msg*semantic[0,4]
        return {'m': res+res0+res2+res4+res6}


    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        torch.nn.init.xavier_uniform_(self.weight0)
        torch.nn.init.xavier_uniform_(self.weight2)
        torch.nn.init.xavier_uniform_(self.weight4)
        torch.nn.init.xavier_uniform_(self.weight6)
        torch.nn.init.xavier_uniform_(self.semantic_weight)

    def forward(self, graph, graph_embedding,e_feat):
        e_feat=e_feat.reshape(-1,1).float()

        # emb=graph_embedding*self.weight
        # emb0=graph_embedding*self.weight0
        # emb2=graph_embedding*self.weight2
        # emb4=graph_embedding*self.weight4
        # emb6=graph_embedding*self.weight6
        emb=emb0=emb2=emb4=emb6=F.elu(graph_embedding*self.weight)

        graph.edata.update({'e':e_feat})
        graph.srcdata.update({'ft': emb,'ft0': emb0,'ft2': emb2,'ft4': emb4,'ft6': emb6})
        graph.update_all(self.message_func_semantic, fn.sum('m', 'ft_s'))
        graph.update_all(self.message_func, fn.sum('m', 'ft'))
        res = graph.dstdata['ft']+self.semantic_prompt_weight*graph.dstdata['ft_s']
        # res = graph.dstdata['ft_s']
        # res=F.sigmoid(res)
        return res


class acm_eachloss_hnode_prompt_layer_feature_weighted_sum(nn.Module):
    def __init__(self,
                 input_dim,
                 pre_semantic_weight=None,
                 negative_slope=0.2,
                 ):
        super(acm_eachloss_hnode_prompt_layer_feature_weighted_sum, self).__init__()
        self.weight= torch.nn.Parameter(torch.Tensor(1,input_dim))
        self.weight0= torch.nn.Parameter(torch.Tensor(1,input_dim))
        self.weight2= torch.nn.Parameter(torch.Tensor(1,input_dim))
        self.weight4= torch.nn.Parameter(torch.Tensor(1,input_dim))
        self.weight6= torch.nn.Parameter(torch.Tensor(1,input_dim))
        self.pre_semantic_weight=pre_semantic_weight
        self.semantic_weight= torch.nn.Parameter(torch.Tensor(1,5))
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.reset_parameters()
    def weighted_sum(self, input, weights):
        weights = weights.expand(input.permute(1, 2, 0).shape).permute(2, 0, 1)
        weighted_inputs = input * weights
        output = weighted_inputs
        return output

    def message_func(self,edges):
        res=edges.src['ft']
        return {'m': res}
    def message_func0(self,edges):
        #type0 subgraph
        _mask = edges.data['e'] == 0
        # print(res.size())
        # print(_mask.size())
        mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft0'], torch.zeros_like(edges.src['ft']))
        return {'m': msg}
    def message_func2(self,edges):
        #type0 subgraph
        _mask = edges.data['e'] == 2
        # print(res.size())
        # print(_mask.size())
        mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft0'], torch.zeros_like(edges.src['ft']))
        return {'m': msg}
    def message_func4(self,edges):
        #type0 subgraph
        _mask = edges.data['e'] == 4
        # print(res.size())
        # print(_mask.size())
        mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft0'], torch.zeros_like(edges.src['ft']))
        return {'m': msg}
    def message_func6(self,edges):
        #type0 subgraph
        _mask = edges.data['e'] == 6
        # print(res.size())
        # print(_mask.size())
        mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft0'], torch.zeros_like(edges.src['ft']))
        return {'m': msg}

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        torch.nn.init.xavier_uniform_(self.weight0)
        torch.nn.init.xavier_uniform_(self.weight2)
        torch.nn.init.xavier_uniform_(self.weight4)
        torch.nn.init.xavier_uniform_(self.weight6)
        torch.nn.init.xavier_uniform_(self.semantic_weight)

    def forward(self, graph, graph_embedding,e_feat):
        e_feat=e_feat.reshape(-1,1).float()
        emb=graph_embedding*self.weight
        emb0=graph_embedding*self.weight
        emb2=graph_embedding*self.weight
        emb4=graph_embedding*self.weight
        emb6=graph_embedding*self.weight
        graph.edata.update({'e':e_feat})
        graph.srcdata.update({'ft': emb,'ft0': emb0,'ft2': emb2,'ft4': emb4,'ft6': emb6})
        graph.update_all(self.message_func, fn.sum('m', 'ft'))
        graph.update_all(self.message_func0, fn.sum('m', 'ft0'))
        graph.update_all(self.message_func0, fn.sum('m', 'ft2'))
        graph.update_all(self.message_func0, fn.sum('m', 'ft4'))
        graph.update_all(self.message_func0, fn.sum('m', 'ft6'))
        res = graph.dstdata['ft']
        res0 = graph.dstdata['ft0']
        res2 = graph.dstdata['ft2']
        res4 = graph.dstdata['ft4']
        res6 = graph.dstdata['ft6']
        return res,res0,res2,res4,res6


class acm_meta_path_hnode_prompt_layer_feature_weighted_sum(nn.Module):
    def __init__(self,
                 input_dim,
                 negative_slope=0.2
                 ):
        super(acm_meta_path_hnode_prompt_layer_feature_weighted_sum, self).__init__()
        self.weight= torch.nn.Parameter(torch.Tensor(1,input_dim))
        self.weight0= torch.nn.Parameter(torch.Tensor(1,input_dim))
        self.weight2= torch.nn.Parameter(torch.Tensor(1,input_dim))
        self.weight4= torch.nn.Parameter(torch.Tensor(1,input_dim))
        self.weight6= torch.nn.Parameter(torch.Tensor(1,input_dim))
        self.semantic_weight= torch.nn.Parameter(torch.Tensor(1,5))
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.reset_parameters()
    def weighted_sum(self, input, weights):
        weights = weights.expand(input.permute(1, 2, 0).shape).permute(2, 0, 1)
        weighted_inputs = input * weights
        output = weighted_inputs
        return output

    def message_func(self,edges):
        #all type subgraph
        res=edges.src['ft']*self.semantic_weight[0,0]
        #type0 subgraph
        _mask = edges.data['e'] == 0
        mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft0'], torch.zeros_like(edges.src['ft']))
        res0=msg*self.semantic_weight[0,1]
        #type2 subgraph
        _mask = edges.data['e'] == 2
        mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft2'], torch.zeros_like(edges.src['ft']))
        res2=msg*self.semantic_weight[0,2]
        #type4 subgraph
        _mask = edges.data['e'] == 4
        mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft4'], torch.zeros_like(edges.src['ft']))
        res4=msg*self.semantic_weight[0,3]
        #type6 subgraph
        _mask = edges.data['e'] == 6
        mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft6'], torch.zeros_like(edges.src['ft']))
        res6=msg*self.semantic_weight[0,4]
        return {'m': res+res0+res2+res4+res6}

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        torch.nn.init.xavier_uniform_(self.weight0)
        torch.nn.init.xavier_uniform_(self.weight2)
        torch.nn.init.xavier_uniform_(self.weight4)
        torch.nn.init.xavier_uniform_(self.weight6)
        torch.nn.init.xavier_uniform_(self.semantic_weight)

    def forward(self, graph, graph_embedding,e_feat):
        e_feat=e_feat.reshape(-1,1).float()
        # emb=graph_embedding*self.weight
        # emb0=graph_embedding*self.weight0
        # emb2=graph_embedding*self.weight2
        # emb4=graph_embedding*self.weight4
        # emb6=graph_embedding*self.weight6
        emb=graph_embedding*self.weight
        emb0=graph_embedding*self.weight
        emb2=graph_embedding*self.weight
        emb4=graph_embedding*self.weight
        emb6=graph_embedding*self.weight

        graph.edata.update({'e':e_feat})
        graph.srcdata.update({'ft': emb,'ft0': emb0,'ft2': emb2,'ft4': emb4,'ft6': emb6})
        graph.update_all(self.message_func, fn.sum('m', 'ft'))
        res = graph.dstdata['ft']
        return res

# book as source and target
class freebase_bidirection_semantic_hnode_prompt_layer_feature_weighted_sum(nn.Module):
    def __init__(self,
                 input_dim,
                 pre_semantic_weight=None,
                 negative_slope=0.2,
                 semantic_prompt_weight=0.1
                 ):
        super(freebase_bidirection_semantic_hnode_prompt_layer_feature_weighted_sum, self).__init__()
        self.weight= torch.nn.Parameter(torch.Tensor(1,input_dim))
        self.semantic_weight= torch.nn.Parameter(torch.Tensor(1,9))
        # self.weight0= torch.nn.Parameter(torch.Tensor(1,input_dim))
        # self.weight2= torch.nn.Parameter(torch.Tensor(1,input_dim))
        # self.weight4= torch.nn.Parameter(torch.Tensor(1,input_dim))
        # self.weight6= torch.nn.Parameter(torch.Tensor(1,input_dim))
        # self.pre_semantic_weight=pre_semantic_weight
        # self.semantic_weight= torch.nn.Parameter(torch.Tensor(1,5))
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.semantic_prompt_weight=semantic_prompt_weight
        self.reset_parameters()
    def weighted_sum(self, input, weights):
        weights = weights.expand(input.permute(1, 2, 0).shape).permute(2, 0, 1)
        weighted_inputs = input * weights
        output = weighted_inputs
        return output

    #book as source and 1hop subgraph
    def message_func(self,edges):
        #all type subgraph
        res=edges.src['ft']
        #type0 subgraph
        _mask = edges.data['e'] == 0
        # print(res.size())
        # print(_mask.size())
        mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft0'], torch.zeros_like(edges.src['ft']))
        res0=msg
        #type2 subgraph
        _mask = edges.data['e'] == 1
        mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft1'], torch.zeros_like(edges.src['ft']))
        res1=msg
        #type4 subgraph
        _mask = edges.data['e'] == 2
        mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft2'], torch.zeros_like(edges.src['ft']))
        res2=msg
        #type6 subgraph
        _mask = edges.data['e'] == 3
        mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft3'], torch.zeros_like(edges.src['ft']))
        res3=msg
        _mask = edges.data['e'] == 4
        mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft4'], torch.zeros_like(edges.src['ft']))
        res4=msg
        return {'m': res+res0+res1+res2+res3+res4}

    #book as destination
    def message_func0(self,edges):
        #type2 subgraph
        _mask = edges.data['e'] == 6
        mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft6'], torch.zeros_like(edges.src['ft']))
        res1=msg
        #type4 subgraph
        _mask = edges.data['e'] == 14
        mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft14'], torch.zeros_like(edges.src['ft']))
        res2=msg
        #type6 subgraph
        _mask = edges.data['e'] == 30
        mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft30'], torch.zeros_like(edges.src['ft']))
        res3=msg
        return {'m': res1+res2+res3}

    # book as source and 1hop subgraph
    def message_func_semantic(self, edges):
        # all type subgraph
        semantic=self.semantic_weight
        semantic=F.normalize(semantic, p=2, dim=1)
        res = edges.src['ft']
        res=res*semantic[0,0]
        # type0 subgraph
        _mask = edges.data['e'] == 0
        # print(res.size())
        # print(_mask.size())
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft0'], torch.zeros_like(edges.src['ft']))
        res0 = msg*semantic[0,1]
        # type2 subgraph
        _mask = edges.data['e'] == 1
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft1'], torch.zeros_like(edges.src['ft']))
        res1 = msg*semantic[0,2]
        # type4 subgraph
        _mask = edges.data['e'] == 2
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft2'], torch.zeros_like(edges.src['ft']))
        res2 = msg*semantic[0,3]
        # type6 subgraph
        _mask = edges.data['e'] == 3
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft3'], torch.zeros_like(edges.src['ft']))
        res3 = msg*semantic[0,4]
        _mask = edges.data['e'] == 4
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft4'], torch.zeros_like(edges.src['ft']))
        res4 = msg*semantic[0,5]
        return {'m': res + res0 + res1 + res2 + res3 + res4}

    # book as destination
    def message_func0_semantic(self, edges):
        # type2 subgraph
        semantic=self.semantic_weight
        semantic=F.normalize(semantic, p=2, dim=1)
        _mask = edges.data['e'] == 6
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft6'], torch.zeros_like(edges.src['ft']))
        res1 = msg*semantic[0,6]
        # type4 subgraph
        _mask = edges.data['e'] == 14
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft14'], torch.zeros_like(edges.src['ft']))
        res2 = msg*semantic[0,7]
        # type6 subgraph
        _mask = edges.data['e'] == 30
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft30'], torch.zeros_like(edges.src['ft']))
        res3 = msg*semantic[0,8]
        return {'m': res1 + res2 + res3}

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        # torch.nn.init.xavier_uniform_(self.weight0)
        # torch.nn.init.xavier_uniform_(self.weight2)
        # torch.nn.init.xavier_uniform_(self.weight4)
        # torch.nn.init.xavier_uniform_(self.weight6)
        torch.nn.init.xavier_uniform_(self.semantic_weight)

    def forward(self, graph, trans_graph, graph_embedding, e_feat):
        e_feat = e_feat.reshape(-1, 1).float()

        # emb=graph_embedding*self.weight
        # emb0=graph_embedding*self.weight0
        # emb2=graph_embedding*self.weight2
        # emb4=graph_embedding*self.weight4
        # emb6=graph_embedding*self.weight6
        # emb=emb0=emb2=emb4=emb6=graph_embedding*self.weight
        emb = emb0 = emb1 = emb2 = emb3 = emb4 = emb6 = emb14 = emb30 = F.elu(graph_embedding * self.weight)
        graph.edata.update({'e': e_feat})
        graph.srcdata.update({'ft': emb, 'ft0': emb0, 'ft1': emb1, 'ft2': emb2, 'ft3': emb3, 'ft4': emb4})
        # graph.update_all(self.message_func, fn.sum('m', 'ft'))
        graph.update_all(self.message_func_semantic, fn.sum('m', 'ft_semantic'))
        trans_graph.edata.update({'e': e_feat})
        trans_graph.srcdata.update({'ft': emb, 'ft6': emb6, 'ft14': emb14, 'ft30': emb30})
        # trans_graph.update_all(self.message_func0, fn.sum('m', 'ft'))
        trans_graph.update_all(self.message_func0_semantic, fn.sum('m', 'ft_semantic'))
        res = graph.dstdata['ft'] + trans_graph.dstdata['ft']\
              +self.semantic_prompt_weight*(graph.dstdata['ft_semantic'] + trans_graph.dstdata['ft_semantic'])
        # res = graph.dstdata['ft_semantic'] + trans_graph.dstdata['ft_semantic']
        return res



class freebase_bidirection_hnode_prompt_layer_feature_weighted_sum(nn.Module):
    def __init__(self,
                 input_dim,
                 pre_semantic_weight=None,
                 negative_slope=0.2,
                 ):
        super(freebase_bidirection_hnode_prompt_layer_feature_weighted_sum, self).__init__()
        self.weight = torch.nn.Parameter(torch.Tensor(1, input_dim))
        # self.weight0= torch.nn.Parameter(torch.Tensor(1,input_dim))
        # self.weight2= torch.nn.Parameter(torch.Tensor(1,input_dim))
        # self.weight4= torch.nn.Parameter(torch.Tensor(1,input_dim))
        # self.weight6= torch.nn.Parameter(torch.Tensor(1,input_dim))
        # self.pre_semantic_weight=pre_semantic_weight
        # self.semantic_weight= torch.nn.Parameter(torch.Tensor(1,5))
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.reset_parameters()

    def weighted_sum(self, input, weights):
        weights = weights.expand(input.permute(1, 2, 0).shape).permute(2, 0, 1)
        weighted_inputs = input * weights
        output = weighted_inputs
        return output

    # book as source and 1hop subgraph
    def message_func(self, edges):
        # all type subgraph
        res = edges.src['ft']
        # type0 subgraph
        _mask = edges.data['e'] == 0
        # print(res.size())
        # print(_mask.size())
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft0'], torch.zeros_like(edges.src['ft']))
        res0 = msg
        # type2 subgraph
        _mask = edges.data['e'] == 1
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft1'], torch.zeros_like(edges.src['ft']))
        res1 = msg
        # type4 subgraph
        _mask = edges.data['e'] == 2
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft2'], torch.zeros_like(edges.src['ft']))
        res2 = msg
        # type6 subgraph
        _mask = edges.data['e'] == 3
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft3'], torch.zeros_like(edges.src['ft']))
        res3 = msg
        _mask = edges.data['e'] == 4
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft4'], torch.zeros_like(edges.src['ft']))
        res4 = msg
        return {'m': res + res0 + res1 + res2 + res3 + res4}

    # book as destination
    def message_func0(self, edges):
        # type2 subgraph
        _mask = edges.data['e'] == 6
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft6'], torch.zeros_like(edges.src['ft']))
        res1 = msg
        # type4 subgraph
        _mask = edges.data['e'] == 14
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft14'], torch.zeros_like(edges.src['ft']))
        res2 = msg
        # type6 subgraph
        _mask = edges.data['e'] == 30
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft30'], torch.zeros_like(edges.src['ft']))
        res3 = msg
        return {'m': res1 + res2 + res3}

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        # torch.nn.init.xavier_uniform_(self.weight0)
        # torch.nn.init.xavier_uniform_(self.weight2)
        # torch.nn.init.xavier_uniform_(self.weight4)
        # torch.nn.init.xavier_uniform_(self.weight6)
        # torch.nn.init.xavier_uniform_(self.semantic_weight)

    def forward(self, graph, trans_graph, graph_embedding,e_feat):
        e_feat=e_feat.reshape(-1,1).float()
        # emb=graph_embedding*self.weight
        # emb0=graph_embedding*self.weight0
        # emb2=graph_embedding*self.weight2
        # emb4=graph_embedding*self.weight4
        # emb6=graph_embedding*self.weight6
        # emb=emb0=emb2=emb4=emb6=graph_embedding*self.weight
        emb=emb0=emb1=emb2=emb3=emb4=emb6=emb14=emb30=F.elu(graph_embedding*self.weight)
        graph.edata.update({'e':e_feat})
        graph.srcdata.update({'ft': emb,'ft0': emb0,'ft1': emb1,'ft2': emb2,'ft3': emb3,'ft4': emb4})
        graph.update_all(self.message_func, fn.sum('m', 'ft'))
        trans_graph.edata.update({'e':e_feat})
        trans_graph.srcdata.update({'ft': emb,'ft6': emb6,'ft14': emb14,'ft30': emb30})
        trans_graph.update_all(self.message_func0, fn.sum('m', 'ft'))
        res = graph.dstdata['ft']+trans_graph.dstdata['ft']
        return res


    # book as source
class freebase_source_hnode_prompt_layer_feature_weighted_sum(nn.Module):
    def __init__(self,
                 input_dim,
                 pre_semantic_weight=None,
                 negative_slope=0.2,
                 ):
        super(freebase_source_hnode_prompt_layer_feature_weighted_sum, self).__init__()
        self.weight = torch.nn.Parameter(torch.Tensor(1, input_dim))
        # self.weight0= torch.nn.Parameter(torch.Tensor(1,input_dim))
        # self.weight2= torch.nn.Parameter(torch.Tensor(1,input_dim))
        # self.weight4= torch.nn.Parameter(torch.Tensor(1,input_dim))
        # self.weight6= torch.nn.Parameter(torch.Tensor(1,input_dim))
        # self.pre_semantic_weight=pre_semantic_weight
        # self.semantic_weight= torch.nn.Parameter(torch.Tensor(1,5))
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.reset_parameters()

    def weighted_sum(self, input, weights):
        weights = weights.expand(input.permute(1, 2, 0).shape).permute(2, 0, 1)
        weighted_inputs = input * weights
        output = weighted_inputs
        return output

    def message_func(self, edges):
        # all type subgraph
        res = edges.src['ft']
        # type0 subgraph
        _mask = edges.data['e'] == 0
        # print(res.size())
        # print(_mask.size())
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft0'], torch.zeros_like(edges.src['ft']))
        res0 = msg
        # type2 subgraph
        _mask = edges.data['e'] == 1
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft1'], torch.zeros_like(edges.src['ft']))
        res1 = msg
        # type4 subgraph
        _mask = edges.data['e'] == 2
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft2'], torch.zeros_like(edges.src['ft']))
        res2 = msg
        # type6 subgraph
        _mask = edges.data['e'] == 3
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft3'], torch.zeros_like(edges.src['ft']))
        res3 = msg
        _mask = edges.data['e'] == 4
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft4'], torch.zeros_like(edges.src['ft']))
        res4 = msg
        return {'m': res + res0 + res1 + res2 + res3 + res4}

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        # torch.nn.init.xavier_uniform_(self.weight0)
        # torch.nn.init.xavier_uniform_(self.weight2)
        # torch.nn.init.xavier_uniform_(self.weight4)
        # torch.nn.init.xavier_uniform_(self.weight6)
        # torch.nn.init.xavier_uniform_(self.semantic_weight)

    def forward(self, graph, graph_embedding,e_feat):
        e_feat=e_feat.reshape(-1,1).float()

        # emb=graph_embedding*self.weight
        # emb0=graph_embedding*self.weight0
        # emb2=graph_embedding*self.weight2
        # emb4=graph_embedding*self.weight4
        # emb6=graph_embedding*self.weight6
        # emb=emb0=emb2=emb4=emb6=graph_embedding*self.weight
        emb=emb0=emb1=emb2=emb3=emb4=F.elu(graph_embedding*self.weight)

        graph.edata.update({'e':e_feat})
        graph.srcdata.update({'ft': emb,'ft0': emb0,'ft1': emb1,'ft2': emb2,'ft3': emb3,'ft4': emb4})
        graph.update_all(self.message_func, fn.sum('m', 'ft'))
        res = graph.dstdata['ft']
        # res=F.sigmoid(res)
        return res

#book as destination
class freebase_des_hnode_prompt_layer_feature_weighted_sum(nn.Module):
    def __init__(self,
                 input_dim,
                 pre_semantic_weight=None,
                 negative_slope=0.2,
                 ):
        super(freebase_des_hnode_prompt_layer_feature_weighted_sum, self).__init__()
        self.weight= torch.nn.Parameter(torch.Tensor(1,input_dim))
        # self.weight0= torch.nn.Parameter(torch.Tensor(1,input_dim))
        # self.weight2= torch.nn.Parameter(torch.Tensor(1,input_dim))
        # self.weight4= torch.nn.Parameter(torch.Tensor(1,input_dim))
        # self.weight6= torch.nn.Parameter(torch.Tensor(1,input_dim))
        # self.pre_semantic_weight=pre_semantic_weight
        # self.semantic_weight= torch.nn.Parameter(torch.Tensor(1,5))
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.reset_parameters()
    def weighted_sum(self, input, weights):
        weights = weights.expand(input.permute(1, 2, 0).shape).permute(2, 0, 1)
        weighted_inputs = input * weights
        output = weighted_inputs
        return output

    def message_func(self,edges):
        #all type subgraph
        res=edges.src['ft']
        #type0 subgraph
        _mask = edges.data['e'] == 0
        # print(res.size())
        # print(_mask.size())
        mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft0'], torch.zeros_like(edges.src['ft']))
        res0=msg
        #type2 subgraph
        _mask = edges.data['e'] == 6
        mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft1'], torch.zeros_like(edges.src['ft']))
        res1=msg
        #type4 subgraph
        _mask = edges.data['e'] == 14
        mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft2'], torch.zeros_like(edges.src['ft']))
        res2=msg
        #type6 subgraph
        _mask = edges.data['e'] == 30
        mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft3'], torch.zeros_like(edges.src['ft']))
        res3=msg
        return {'m': res+res0+res1+res2+res3}

    #add pre semantic prompt
    # def message_func(self,edges):
    #     #all type subgraph
    #     semantic=self.pre_semantic_weight
    #     semantic=F.normalize(semantic, p=2, dim=1)
    #     res=edges.src['ft']*semantic[0,0]
    #     #type0 subgraph
    #     _mask = edges.data['e'] == 0
    #     mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
    #     msg = torch.where(mask, edges.src['ft0'], torch.zeros_like(edges.src['ft']))
    #     res0=msg*semantic[0,1]
    #     #type2 subgraph
    #     _mask = edges.data['e'] == 2
    #     mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
    #     msg = torch.where(mask, edges.src['ft2'], torch.zeros_like(edges.src['ft']))
    #     res2=msg*semantic[0,2]
    #     #type4 subgraph
    #     _mask = edges.data['e'] == 4
    #     mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
    #     msg = torch.where(mask, edges.src['ft4'], torch.zeros_like(edges.src['ft']))
    #     res4=msg*semantic[0,3]
    #     #type6 subgraph
    #     _mask = edges.data['e'] == 6
    #     mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
    #     msg = torch.where(mask, edges.src['ft6'], torch.zeros_like(edges.src['ft']))
    #     res6=msg*semantic[0,4]
    #     return {'m': res+res0+res2+res4+res6}

    #add semantic prompt
    # def message_func(self,edges):
    #     #all type subgraph
    #     semantic=self.semantic_weight
    #     semantic=F.normalize(semantic, p=2, dim=1)
    #     res=edges.src['ft']*semantic[0,0]
    #     #type0 subgraph
    #     _mask = edges.data['e'] == 0
    #     mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
    #     msg = torch.where(mask, edges.src['ft0'], torch.zeros_like(edges.src['ft']))
    #     res0=msg*semantic[0,1]
    #     #type2 subgraph
    #     _mask = edges.data['e'] == 2
    #     mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
    #     msg = torch.where(mask, edges.src['ft2'], torch.zeros_like(edges.src['ft']))
    #     res2=msg*semantic[0,2]
    #     #type4 subgraph
    #     _mask = edges.data['e'] == 4
    #     mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
    #     msg = torch.where(mask, edges.src['ft4'], torch.zeros_like(edges.src['ft']))
    #     res4=msg*semantic[0,3]
    #     #type6 subgraph
    #     _mask = edges.data['e'] == 6
    #     mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
    #     msg = torch.where(mask, edges.src['ft6'], torch.zeros_like(edges.src['ft']))
    #     res6=msg*semantic[0,4]
    #     return {'m': res+res0+res2+res4+res6}

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        # torch.nn.init.xavier_uniform_(self.weight0)
        # torch.nn.init.xavier_uniform_(self.weight2)
        # torch.nn.init.xavier_uniform_(self.weight4)
        # torch.nn.init.xavier_uniform_(self.weight6)
        # torch.nn.init.xavier_uniform_(self.semantic_weight)

    def forward(self, graph, graph_embedding,e_feat):
        e_feat=e_feat.reshape(-1,1).float()

        # emb=graph_embedding*self.weight
        # emb0=graph_embedding*self.weight0
        # emb2=graph_embedding*self.weight2
        # emb4=graph_embedding*self.weight4
        # emb6=graph_embedding*self.weight6
        # emb=emb0=emb2=emb4=emb6=graph_embedding*self.weight
        emb=emb0=emb1=emb2=emb3=F.elu(graph_embedding*self.weight)

        graph.edata.update({'e':e_feat})
        graph.srcdata.update({'ft': emb,'ft0': emb0,'ft1': emb1,'ft2': emb2,'ft3': emb3})
        graph.update_all(self.message_func, fn.sum('m', 'ft'))
        res = graph.dstdata['ft']
        # res=F.sigmoid(res)
        return res



class dblp_hnode_prompt_layer_feature_weighted_sum(nn.Module):
    def __init__(self,
                 input_dim,
                 pre_semantic_weight=None,
                 negative_slope=0.2,
                 ):
        super(dblp_hnode_prompt_layer_feature_weighted_sum, self).__init__()
        self.weight= torch.nn.Parameter(torch.Tensor(1,input_dim))
        self.weight0= torch.nn.Parameter(torch.Tensor(1,input_dim))
        self.weight1= torch.nn.Parameter(torch.Tensor(1,input_dim))
        self.weight2= torch.nn.Parameter(torch.Tensor(1,input_dim))
        self.pre_semantic_weight=pre_semantic_weight
        self.semantic_weight= torch.nn.Parameter(torch.Tensor(1,4))
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.reset_parameters()
    def weighted_sum(self, input, weights):
        weights = weights.expand(input.permute(1, 2, 0).shape).permute(2, 0, 1)
        weighted_inputs = input * weights
        output = weighted_inputs
        return output

    def message_func_onehop(self,edges):
        _mask = edges.data['e'] == 3
        mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft_twohop'], torch.zeros_like(edges.src['ft']))
        #paper->author
        res0=msg
        return {'m': res0}

    def message_func_twohop(self, edges):
        # type0 subgraph
        res=edges.src['ft']
        _mask = edges.data['e'] == 0
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft0'], torch.zeros_like(edges.src['ft']))
        #author->paper->author
        res0 = msg
        # type2 subgraph
        _mask = edges.data['e'] == 4
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft4'], torch.zeros_like(edges.src['ft']))
        #term->paper->author
        res1 = msg
        # type4 subgraph
        _mask = edges.data['e'] == 5
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft5'], torch.zeros_like(edges.src['ft']))
        #venue->paper->author
        res2 = msg
        return {'m': res+ res0 + res1 + res2}

    #add pre semantic prompt
    # def message_func(self,edges):
    #     #all type subgraph
    #     semantic=self.pre_semantic_weight
    #     semantic=F.normalize(semantic, p=2, dim=1)
    #     res=edges.src['ft']*semantic[0,0]
    #     #type0 subgraph
    #     _mask = edges.data['e'] == 0
    #     mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
    #     msg = torch.where(mask, edges.src['ft0'], torch.zeros_like(edges.src['ft']))
    #     res0=msg*semantic[0,1]
    #     #type2 subgraph
    #     _mask = edges.data['e'] == 2
    #     mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
    #     msg = torch.where(mask, edges.src['ft2'], torch.zeros_like(edges.src['ft']))
    #     res2=msg*semantic[0,2]
    #     #type4 subgraph
    #     _mask = edges.data['e'] == 4
    #     mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
    #     msg = torch.where(mask, edges.src['ft4'], torch.zeros_like(edges.src['ft']))
    #     res4=msg*semantic[0,3]
    #     #type6 subgraph
    #     _mask = edges.data['e'] == 6
    #     mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
    #     msg = torch.where(mask, edges.src['ft6'], torch.zeros_like(edges.src['ft']))
    #     res6=msg*semantic[0,4]
    #     return {'m': res+res0+res2+res4+res6}

    #add semantic prompt
    # def message_func(self,edges):
    #     #all type subgraph
    #     semantic=self.semantic_weight
    #     semantic=F.normalize(semantic, p=2, dim=1)
    #     res=edges.src['ft']*semantic[0,0]
    #     #type0 subgraph
    #     _mask = edges.data['e'] == 0
    #     mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
    #     msg = torch.where(mask, edges.src['ft0'], torch.zeros_like(edges.src['ft']))
    #     res0=msg*semantic[0,1]
    #     #type2 subgraph
    #     _mask = edges.data['e'] == 2
    #     mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
    #     msg = torch.where(mask, edges.src['ft2'], torch.zeros_like(edges.src['ft']))
    #     res2=msg*semantic[0,2]
    #     #type4 subgraph
    #     _mask = edges.data['e'] == 4
    #     mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
    #     msg = torch.where(mask, edges.src['ft4'], torch.zeros_like(edges.src['ft']))
    #     res4=msg*semantic[0,3]
    #     #type6 subgraph
    #     _mask = edges.data['e'] == 6
    #     mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
    #     msg = torch.where(mask, edges.src['ft6'], torch.zeros_like(edges.src['ft']))
    #     res6=msg*semantic[0,4]
    #     return {'m': res+res0+res2+res4+res6}

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        torch.nn.init.xavier_uniform_(self.weight0)
        torch.nn.init.xavier_uniform_(self.weight1)
        torch.nn.init.xavier_uniform_(self.weight2)
        torch.nn.init.xavier_uniform_(self.semantic_weight)

    def forward(self, graph, graph_embedding,e_feat):
        e_feat=e_feat.reshape(-1,1).float()

        # emb=graph_embedding*self.weight
        # emb0=graph_embedding*self.weight0
        # emb2=graph_embedding*self.weight2
        # emb4=graph_embedding*self.weight4
        # emb6=graph_embedding*self.weight6
        # emb=emb0=emb2=emb4=emb6=graph_embedding*self.weight
        emb=emb0=emb3=emb4=emb5=F.elu(graph_embedding*self.weight)

        graph.edata.update({'e':e_feat})
        graph.srcdata.update({'ft': emb,'ft0': emb0,'ft3': emb3,'ft4': emb4,'ft5': emb5})
        graph.update_all(self.message_func_twohop, fn.sum('m', 'ft_twohop'))
        graph.srcdata.update({'ft_twohop': graph.dstdata['ft_twohop']})
        graph.update_all(self.message_func_onehop, fn.sum('m', 'ft_onehop'))
        res = graph.dstdata['ft_onehop']
        # res=F.sigmoid(res)
        return res

class dblp_hnode_semantic_prompt_layer_feature_weighted_sum(nn.Module):
    def __init__(self,
                 input_dim,
                 pre_semantic_weight=None,
                 negative_slope=0.2,
                 semantic_prompt_weight=0.1
                 ):
        super(dblp_hnode_semantic_prompt_layer_feature_weighted_sum, self).__init__()
        self.weight= torch.nn.Parameter(torch.Tensor(1,input_dim))
        self.weight0= torch.nn.Parameter(torch.Tensor(1,input_dim))
        self.weight1= torch.nn.Parameter(torch.Tensor(1,input_dim))
        self.weight2= torch.nn.Parameter(torch.Tensor(1,input_dim))
        self.pre_semantic_weight=pre_semantic_weight
        self.semantic_weight= torch.nn.Parameter(torch.Tensor(1,5))
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.semantic_prompt_weight=semantic_prompt_weight
        self.reset_parameters()
    def weighted_sum(self, input, weights):
        weights = weights.expand(input.permute(1, 2, 0).shape).permute(2, 0, 1)
        weighted_inputs = input * weights
        output = weighted_inputs
        return output
    def message_func_onehop(self,edges):
        _mask = edges.data['e'] == 3
        mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft_twohop'], torch.zeros_like(edges.src['ft']))
        #paper->author
        res0=msg
        return {'m': res0}
    def message_func_twohop(self, edges):
        # type0 subgraph
        res=edges.src['ft']
        _mask = edges.data['e'] == 0
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft0'], torch.zeros_like(edges.src['ft']))
        #author->paper->author
        res0 = msg
        # type2 subgraph
        _mask = edges.data['e'] == 4
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft4'], torch.zeros_like(edges.src['ft']))
        #term->paper->author
        res1 = msg
        # type4 subgraph
        _mask = edges.data['e'] == 5
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft5'], torch.zeros_like(edges.src['ft']))
        #venue->paper->author
        res2 = msg
        return {'m': res+ res0 + res1 + res2}

    def message_func_onehop_semantic(self,edges):
        semantic=self.semantic_weight
        semantic=F.normalize(semantic, p=2, dim=1)
        _mask = edges.data['e'] == 3
        mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft_twohop_s'], torch.zeros_like(edges.src['ft']))
        #paper->author
        res0=msg*semantic[0,0]
        return {'m': res0}
    def message_func_twohop_semantic(self, edges):
        # type0 subgraph
        semantic=self.semantic_weight
        semantic=F.normalize(semantic, p=2, dim=1)
        res=edges.src['ft']*semantic[0,1]
        _mask = edges.data['e'] == 0
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft0'], torch.zeros_like(edges.src['ft']))
        #author->paper->author
        res0 = msg*semantic[0,2]
        # type2 subgraph
        _mask = edges.data['e'] == 4
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft4'], torch.zeros_like(edges.src['ft']))
        #term->paper->author
        res1 = msg*semantic[0,3]
        # type4 subgraph
        _mask = edges.data['e'] == 5
        mask = torch.broadcast_to(_mask.permute(1, 0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft5'], torch.zeros_like(edges.src['ft']))
        #venue->paper->author
        res2 = msg*semantic[0,4]
        return {'m': res+ res0 + res1 + res2}

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        torch.nn.init.xavier_uniform_(self.weight0)
        torch.nn.init.xavier_uniform_(self.weight1)
        torch.nn.init.xavier_uniform_(self.weight2)
        torch.nn.init.xavier_uniform_(self.semantic_weight)

    def forward(self, graph, graph_embedding,e_feat):
        e_feat=e_feat.reshape(-1,1).float()

        # emb=graph_embedding*self.weight
        # emb0=graph_embedding*self.weight0
        # emb2=graph_embedding*self.weight2
        # emb4=graph_embedding*self.weight4
        # emb6=graph_embedding*self.weight6
        # emb=emb0=emb2=emb4=emb6=graph_embedding*self.weight
        emb=emb0=emb3=emb4=emb5=F.elu(graph_embedding*self.weight)

        graph.edata.update({'e':e_feat})
        graph.srcdata.update({'ft': emb,'ft0': emb0,'ft3': emb3,'ft4': emb4,'ft5': emb5})
        graph.update_all(self.message_func_twohop, fn.sum('m', 'ft_twohop'))
        graph.srcdata.update({'ft_twohop': graph.dstdata['ft_twohop']})
        graph.update_all(self.message_func_onehop, fn.sum('m', 'ft_onehop'))
        graph.update_all(self.message_func_twohop_semantic, fn.sum('m', 'ft_twohop_s'))
        graph.srcdata.update({'ft_twohop_s': graph.dstdata['ft_twohop_s']})
        graph.update_all(self.message_func_onehop_semantic, fn.sum('m', 'ft_onehop_s'))
        res = graph.dstdata['ft_onehop']+self.semantic_prompt_weight*graph.dstdata['ft_onehop_s']
        # res = graph.dstdata['ft_onehop_s']
        # res=F.sigmoid(res)
        return res


class node_prompt_layer_feature_cat(nn.Module):
    def __init__(self,prompt_dim):
        super(node_prompt_layer_feature_cat, self).__init__()
        self.weight= torch.nn.Parameter(torch.Tensor(1,prompt_dim))
        self.reset_parameters()
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
    def forward(self, graph,graph_embedding):
        graph_embedding=torch.cat([graph_embedding,torch.broadcast_to(self.weight,(graph_embedding.size(0),self.weight.size(1)))],dim=1)
        emb=graph_embedding
        graph.srcdata.update({'ft': emb})
        graph.update_all(fn.copy_u('ft','m'), fn.sum('m', 'ft'))
        res = graph.dstdata['ft']
        return res

class node_prompt_layer_feature_cat_edge(nn.Module):
    def __init__(self,prompt_dim):
        super(node_prompt_layer_feature_cat_edge, self).__init__()
        self.weight= torch.nn.Parameter(torch.Tensor(1,prompt_dim))
        self.prompt_dim=prompt_dim
        self.reset_parameters()
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
    def message_func(self,edges):
        return {'m': torch.cat((edges.src['ft'],edges.data['p']),dim=1)}
    def forward(self, graph,graph_embedding):
        emb=graph_embedding
        graph.srcdata.update({'ft': emb})
        enum=graph.num_edges()
        graph.edata.update({'p':torch.broadcast_to(self.weight,(enum,self.prompt_dim))})
        graph.update_all(self.message_func, fn.sum('m', 'ft'))
        res = graph.dstdata['ft']
        return res

class hnode_prompt_layer_feature_cat_edge(nn.Module):
    # def __init__(self,prompt_dim):
    def __init__(self,prompt_dim,heterprompt_dim):
        super(hnode_prompt_layer_feature_cat_edge, self).__init__()
        self.weight= torch.nn.Parameter(torch.Tensor(1,prompt_dim))
        self.hetero_prompt = torch.nn.Parameter(torch.Tensor(1, heterprompt_dim))
        self.hetero_dim=heterprompt_dim
        self.hetero_prompt=torch.nn.Parameter(torch.Tensor(1,heterprompt_dim))
        self.prompt_dim=prompt_dim
        self.reset_parameters()
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        torch.nn.init.xavier_uniform_(self.hetero_prompt)
    def message_func(self,edges):
        return {'m': torch.cat((edges.src['ft']*edges.data['p'],edges.data['e']),dim=1)}
    def forward(self, graph, graph_embedding,e_feat):
        graph.srcdata.update({'ft': graph_embedding})
        enum=graph.num_edges()
        graph.edata.update({'p':torch.broadcast_to(self.weight,(enum,self.prompt_dim))})
        graph.edata.update({'hp':torch.broadcast_to(self.hetero_prompt,(enum,self.hetero_dim))})
        graph.edata.update({'e':e_feat})
        graph.update_all(self.message_func, fn.sum('m', 'ft'))
        res = graph.dstdata['ft']
        return res

class node_prompt_layer_feature_sum(nn.Module):
    def __init__(self):
        super(node_prompt_layer_feature_sum, self).__init__()
    def forward(self, graph,graph_embedding):
        graph.srcdata.update({'ft': graph_embedding})
        graph.update_all(fn.copy_u('ft','m'), fn.sum('m', 'ft'))
        return graph.dstdata['ft']

class hnode_prompt_layer_feature_sum(nn.Module):
    def __init__(self,negative_slope=0.2
                 ):
        super(hnode_prompt_layer_feature_sum, self).__init__()
        self.leaky_relu = nn.LeakyReLU(negative_slope)
    def message_func(self,edges):
        return {'m': torch.cat((edges.src['ft'],edges.data['e']),dim=1)}
    def forward(self, graph, graph_embedding,e_feat):
        e_feat=e_feat.reshape(-1,1).float()
        emb=graph_embedding
        graph.srcdata.update({'ft': emb})
        graph.edata.update({'e': e_feat})
        graph.update_all(self.message_func, fn.sum('m', 'ft'))
        res = graph.dstdata['ft']
        return res

def distance2center(f,center):
    _f=torch.broadcast_to(f,(center.size(0),f.size(0),f.size(1)))
    _center=torch.broadcast_to(center,(f.size(0),center.size(0),center.size(1)))
    _f=_f.permute(1,0,2)
    _center=_center.reshape(-1,_center.size(2))
    _f=_f.reshape(-1,_f.size(2))
    cos=torch.cosine_similarity(_f,_center,dim=1)
    res=cos
    res=res.reshape(f.size(0),center.size(0))
    return res

#calculate the center embedding of each class
def center_embedding(input,index,label_num=0,debug=False):
    device=input.device
    mean = torch.ones(index.size(0)).to(device)
    _mean = torch.zeros(label_num, device=device).scatter_add_(dim=0, index=index, src=mean)
    index=index.reshape(-1,1)
    index = index.expand(input.size())
    c = torch.zeros(label_num, input.size(1)).to(device)
    c = c.scatter_add_(dim=0, index=index, src=input)
    _mean=_mean.reshape(-1,1)
    c = c / _mean
    return c

from dgl.nn.pytorch import GraphConv
class hprompt_gcn(nn.Module):
    def __init__(self,
                 input_dim,
                 negative_slope=0.2
                 ):
        super(hprompt_gcn, self).__init__()
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.layers = nn.ModuleList()
        self.layers.append(GraphConv(input_dim, input_dim, weight=False))
        # hidden layers
        for i in range(2 - 1):
            self.layers.append(GraphConv(input_dim, input_dim))
        self.dropout = nn.Dropout(p=0.2)
        # self.gcn=GraphConv(input_dim, input_dim)
    def message_func(self,edges):
        #all type subgraph
        res=edges.src['ft']
        #type0 subgraph
        _mask = edges.data['e'] == 0
        mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft0'], torch.zeros_like(edges.src['ft']))
        # res0=msg*self.semantic_weight[0,1]
        res0=msg
        #type2 subgraph
        _mask = edges.data['e'] == 2
        mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft2'], torch.zeros_like(edges.src['ft']))
        # res2=msg*self.semantic_weight[0,2]
        res2=msg
        #type4 subgraph
        _mask = edges.data['e'] == 4
        mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft4'], torch.zeros_like(edges.src['ft']))
        # res4=msg*self.semantic_weight[0,3]
        res4=msg

        #type6 subgraph
        _mask = edges.data['e'] == 6
        mask = torch.broadcast_to(_mask.permute(1,0), edges.src['ft'].permute(1, 0).size()).permute(1, 0)
        msg = torch.where(mask, edges.src['ft6'], torch.zeros_like(edges.src['ft']))
        # res6=msg*self.semantic_weight[0,4]
        res6=msg
        return {'m': res+res0+res2+res4+res6}

    def forward(self, graph, graph_embedding,e_feat):
        e_feat=e_feat.reshape(-1,1).float()
        # graph_embedding=self.gcn(graph,graph_embedding)
        for i, layer in enumerate(self.layers):
            graph_embedding = self.dropout(graph_embedding)
            graph_embedding = layer(graph, graph_embedding)
        emb=graph_embedding
        emb0=graph_embedding
        emb2=graph_embedding
        emb4=graph_embedding
        emb6=graph_embedding
        graph.edata.update({'e':e_feat})
        graph.srcdata.update({'ft': emb,'ft0': emb0,'ft2': emb2,'ft4': emb4,'ft6': emb6})
        graph.update_all(self.message_func, fn.sum('m', 'ft'))
        res = graph.dstdata['ft']
        return res

class prompt_gcn(nn.Module):
    def __init__(self,input_dim):
        super(prompt_gcn, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GraphConv(input_dim, input_dim, weight=False))
        # hidden layers
        for i in range(2 - 1):
            self.layers.append(GraphConv(input_dim, input_dim))
        self.dropout = nn.Dropout(p=0.2)
        self.gcn=GraphConv(input_dim, input_dim)
    def forward(self, graph, graph_embedding):
        for i, layer in enumerate(self.layers):
            graph_embedding = self.dropout(graph_embedding)
            graph_embedding = layer(graph, graph_embedding)
        emb=graph_embedding
        # emb=self.gcn(graph,graph_embedding)
        graph.srcdata.update({'ft': emb})
        graph.update_all(fn.copy_u('ft','m'), fn.sum('m', 'ft'))
        return graph.dstdata['ft']


