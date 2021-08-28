import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl.nn.pytorch import GraphConv, HeteroGraphConv
from openhgnn.models.MetapathConv import MetapathConv
from openhgnn.models.macro_layer.SemanticConv import SemanticAttention


class GeneralLayer(nn.Module):
    '''General wrapper for layers'''

    def __init__(self, name, meta_paths, macro_func, dim_in, dim_out, dropout, act=None, has_bn=True,
                 has_l2norm=False, **kwargs):
        super(GeneralLayer, self).__init__()
        self.has_l2norm = has_l2norm
        self.layer = layer_dict[name](meta_paths, dim_in, dim_out, macro_func,
                                      bias=not has_bn, **kwargs)
        layer_wrapper = []
        if has_bn:
            layer_wrapper.append(nn.BatchNorm1d(dim_out))
        if dropout > 0:
            layer_wrapper.append(nn.Dropout(p=dropout))
        if act is not None:
            layer_wrapper.append(act)
        self.post_layer = nn.Sequential(*layer_wrapper)

    def forward(self, mp_g_list, h):
        h = self.layer(mp_g_list, h)
        h = self.post_layer(h)
        if self.has_l2norm:
            h = F.normalize(h, p=2, dim=1)
        return h


class HANConv(nn.Module):
    def __init__(self, meta_paths, dim_in, dim_out, macro_func, bias=False, **kwargs):
        super(HANConv, self).__init__()
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
            [dgl.nn.pytorch.GATConv(dim_in, dim_out, num_heads=1, bias=bias)
            #[dgl.nn.pytorch.GraphConv(dim_in, dim_out, bias=bias)
            for _ in meta_paths],
            macro_func
        )
        self.meta_paths = meta_paths

    def forward(self, mp_g_list, h):
        h = self.model(mp_g_list, h)
        return h


def Aggr_sum(z):
    return z.sum(1)


def Aggr_max(z):
    return z.max(1)


def Aggr_mean(z):
    return z.mean(1)

layer_dict = {
    'hanconv': HANConv,
}
