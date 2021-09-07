import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl.nn.pytorch import GraphConv, HeteroGraphConv
from openhgnn.models.macro_layer.SemanticConv import SemanticAttention
from ..models.layers import homo_layer_dict

class HeteroGeneralLayer(nn.Module):
    '''General wrapper for layers'''

    def __init__(self, name, rel_names, dim_in, dim_out, dropout, act=None, has_bn=True,
                 has_l2norm=False, **kwargs):
        super(HeteroGeneralLayer, self).__init__()
        self.has_l2norm = has_l2norm
        has_bn = has_bn
        self.layer = RelationConv(name, rel_names, dim_in, dim_out,
                                      bias=not has_bn, **kwargs)
        layer_wrapper = []
        if has_bn:
            layer_wrapper.append(nn.BatchNorm1d(dim_out))
        if dropout > 0:
            layer_wrapper.append(nn.Dropout(p=dropout))
        if act is not None:
            layer_wrapper.append(act)
        self.post_layer = nn.Sequential(*layer_wrapper)

    def forward(self, g, h_dict):
        h_dict = self.layer(g, h_dict)
        if self.has_l2norm:
            for name, batch_h in h_dict.items():
                h_dict[name] = F.normalize(self.post_layer(batch_h), p=2, dim=-1)
        return h_dict


class BatchNorm1dNode(nn.Module):
    '''General wrapper for layers'''

    def __init__(self, dim_in):
        super(BatchNorm1dNode, self).__init__()
        self.bn = nn.BatchNorm1d(dim_in)

    def forward(self, h):
        h = self.bn(h)
        return h


class RelationConv(nn.Module):
    def __init__(self, name, rel_names, dim_in, dim_out, bias=False, **kwargs):
        super(RelationConv, self).__init__()
        macro_func = kwargs['macro_func']
        if macro_func == 'attention':
            macro_func = SemanticAttention(dim_out)
        self.model = HeteroGraphConv({
            rel: homo_layer_dict[name](dim_in, dim_out, bias=bias)
            for rel in rel_names
        }, aggregate=macro_func)

    def forward(self, g, h_dict):
        h_dict = self.model(g, h_dict)
        return h_dict
