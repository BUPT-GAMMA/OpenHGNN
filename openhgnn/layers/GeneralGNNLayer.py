import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

from .MetapathConv import MetapathConv
from .macro_layer.SemanticConv import SemanticAttention
from ..models.SimpleHGN import SimpleHGNConv
from ..models.HGT import HGTConv


class MPConv(nn.Module):
    def __init__(self, name, dim_in, dim_out, bias=False, **kwargs):
        super(MPConv, self).__init__()
        macro_func = kwargs['macro_func']
        meta_paths = kwargs['meta_paths']
        if macro_func == 'attention':
            macro_func = SemanticAttention(dim_out)
        elif macro_func == 'sum':
            macro_func = Aggr_sum
        elif macro_func == 'mean':
            macro_func = Aggr_mean
        elif macro_func == 'max':
            macro_func = Aggr_max
        self.model = MetapathConv(
            meta_paths,
            [homo_layer_dict[name](dim_in, dim_out, bias=bias, **kwargs)
            for _ in meta_paths],
            macro_func
        )
        self.meta_paths = meta_paths

    def forward(self, mp_g_list, h):
        h = self.model(mp_g_list, h)
        return h


def Aggr_sum(z):
    z = torch.stack(z, dim=1)
    return z.sum(1)


def Aggr_max(z):
    z = torch.stack(z, dim=1)
    return z.max(1)[0]


def Aggr_mean(z):
    z = torch.stack(z, dim=1)
    return z.mean(1)


class GeneralLayer(nn.Module):
    '''General wrapper for layers'''

    def __init__(self, name, dim_in, dim_out, dropout, act=None, has_bn=True,
                 has_l2norm=False, **kwargs):
        super(GeneralLayer, self).__init__()
        self.has_l2norm = has_l2norm
        if kwargs.get('meta_paths') is not None:
            self.layer = MPConv(name, dim_in, dim_out,
                                          bias=not has_bn, **kwargs)
        else:
            self.layer = homo_layer_dict[name](dim_in, dim_out,
                                          bias=not has_bn, **kwargs)
        layer_wrapper = []
        if has_bn:
            layer_wrapper.append(nn.BatchNorm1d(dim_out))
        if dropout > 0:
            layer_wrapper.append(nn.Dropout(p=dropout))
        if act is not None:
            layer_wrapper.append(act)
        self.post_layer = nn.Sequential(*layer_wrapper)

    def forward(self, homo_g, h):
        h = self.layer(homo_g, h)
        h = self.post_layer(h)
        if self.has_l2norm:
            h = F.normalize(h, p=2, dim=1)
        return h


class MultiLinearLayer(nn.Module):
    def __init__(self, linear_list, dropout, act=None, has_bn=True,
                 has_l2norm=False, **kwargs):
        super(MultiLinearLayer, self).__init__()
        for i in range(len(linear_list) - 1):
            d_in = linear_list[i]
            d_out = linear_list[i+1]
            layer = Linear(d_in, d_out, dropout, act, has_bn, has_l2norm)
            self.add_module('Layer_{}'.format(i), layer)

    def forward(self, h):
        for layer in self.children():
            h = layer(h)
        return h


class Linear(nn.Module):
    def __init__(self, dim_in, dim_out, dropout, act=None, has_bn=True,
                 has_l2norm=False, **kwargs):
        super(Linear, self).__init__()
        self.has_l2norm = has_l2norm
        self.layer = nn.Linear(dim_in, dim_out, bias=not has_bn, **kwargs)
        layer_wrapper = []
        if has_bn:
            layer_wrapper.append(nn.BatchNorm1d(dim_out))
        if dropout > 0:
            layer_wrapper.append(nn.Dropout(p=dropout))
        if act is not None:
            layer_wrapper.append(act)
        self.post_layer = nn.Sequential(*layer_wrapper)

    def forward(self, h):
        h = self.layer(h)
        h = self.post_layer(h)
        if self.has_l2norm:
            h = F.normalize(h, p=2, dim=1)
        return h


class BatchNorm1dNode(nn.Module):
    '''General wrapper for layers'''

    def __init__(self, dim_in):
        super(BatchNorm1dNode, self).__init__()
        self.bn = nn.BatchNorm1d(dim_in)

    def forward(self, h):
        h = self.bn(h)
        return h


# class BatchNorm1dEdge(nn.Module):
#     '''General wrapper for layers'''
#
#     def __init__(self, dim_in):
#         super(BatchNorm1dEdge, self).__init__()
#         self.bn = nn.BatchNorm1d(dim_in, eps=cfg.bn.eps, momentum=cfg.bn.mom)
#
#     def forward(self, batch):
#         batch.edge_feature = self.bn(batch.edge_feature)
#         return batch


class GCNConv(nn.Module):
    def __init__(self, dim_in, dim_out, bias=False, **kwargs):
        super(GCNConv, self).__init__()
        self.model = dgl.nn.pytorch.GraphConv(dim_in, dim_out, norm='both', bias=bias, allow_zero_in_degree=True)

    def forward(self, g, h):
        with g.local_scope():
            # g = dgl.add_reverse_edges(g)
            # g = dgl.remove_self_loop(g)
            # g = dgl.add_self_loop(g)
            h = self.model(g, h)
        return h


class SAGEConv(nn.Module):
    def __init__(self, dim_in, dim_out, bias=False, **kwargs):
        super(SAGEConv, self).__init__()
        self.model = dgl.nn.pytorch.SAGEConv(dim_in, dim_out, aggregator_type='mean', bias=bias)

    def forward(self, g, h):
        h = self.model(g, h)
        return h


class GATConv(nn.Module):
    def __init__(self, dim_in, dim_out, bias=False, **kwargs):
        super(GATConv, self).__init__()
        self.model = dgl.nn.pytorch.GATConv(dim_in, dim_out, num_heads=kwargs['num_heads'], bias=bias, allow_zero_in_degree=True)

    def forward(self, g, h):
        # Note, falatten
        h = self.model(g, h).mean(1)
        return h


class GINConv(nn.Module):
    def __init__(self, dim_in, dim_out, bias=False, **kwargs):
        super(GINConv, self).__init__()
        lin = nn.Sequential(nn.Linear(dim_in, dim_out, bias), nn.ReLU(),
                               nn.Linear(dim_out, dim_out))
        self.model = dgl.nn.pytorch.GINConv(lin, 'max')

    def forward(self, g, h):
        h = self.model(g, h)
        return h


class SimpleConv(nn.Module):
    def __init__(self, dim_in, dim_out, bias=False, **kwargs):
        super(SimpleConv, self).__init__()
        self.model = SimpleHGNConv(dim_in, dim_in, int(dim_out / kwargs['num_heads']), kwargs['num_heads'], kwargs['num_etypes'], beta=0.0)

    def forward(self, g, h):
        h = self.model(g, h)
        return h


class HgtConv(nn.Module):
    def __init__(self, dim_in, dim_out, bias=False, **kwargs):
        super(HgtConv, self).__init__()
        self.model = HGTConv(dim_in, dim_out, n_heads=kwargs['num_heads'], n_etypes=kwargs['num_etypes'],
                             n_ntypes=kwargs['num_ntypes'])
    def forward(self, g, h):
        h = self.model(g,h)
        return h


class APPNPConv(nn.Module):
    def __init__(self, dim_in, dim_out, bias=False, **kwargs):
        super(APPNPConv, self).__init__()
        self.model = dgl.nn.pytorch.APPNPConv(k=3, alpha=0.5)
        self.lin = nn.Linear(dim_in, dim_out, bias, )

    def forward(self, g, h):
        h = self.model(g, h)
        h = self.lin(h)
        return h


homo_layer_dict = {
    'gcnconv': GCNConv,
    'sageconv': SAGEConv,
    'gatconv': GATConv,
    'ginconv': GINConv,
    'simpleconv': SimpleConv,
    'hgtconv': HgtConv,
    'appnpconv': APPNPConv
}