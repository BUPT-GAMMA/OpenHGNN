import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
from ..relation_models import RGCNConv


## General classes
class GeneralLayer(nn.Module):
    '''General wrapper for layers'''

    def __init__(self, name, dim_in, dim_out, dropout, act=None, has_bn=True,
                 has_l2norm=False, **kwargs):
        super(GeneralLayer, self).__init__()
        self.has_l2norm = has_l2norm
        self.layer = layer_dict[name](dim_in, dim_out,
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
        self.model = dgl.nn.pytorch.GraphConv(dim_in, dim_out, bias=bias)

    def forward(self, g, h):
        h = self.model(g, h)
        return h


class SAGEConv(nn.Module):
    def __init__(self, dim_in, dim_out, bias=False, **kwargs):
        super(SAGEConv, self).__init__()
        self.model = dgl.nn.pytorch.SAGEConv(dim_in, dim_out, aggregator_type='mean', bias=bias)

    def forward(self, batch):
        batch.node_feature = self.model(batch.node_feature, batch.edge_index)
        return batch


class GATConv(nn.Module):
    def __init__(self, dim_in, dim_out, bias=False, **kwargs):
        super(GATConv, self).__init__()
        self.model = dgl.nn.pytorch.GATConv(dim_in, dim_out, num_heads=1, bias=bias)

    def forward(self, g, h):
        # Note, falatten
        h = self.model(g, h).flatten(1)
        return h


class GINConv(nn.Module):
    def __init__(self, dim_in, dim_out, bias=False, **kwargs):
        super(GINConv, self).__init__()
        gin_nn = nn.Sequential(nn.Linear(dim_in, dim_out), nn.ReLU(),
                               nn.Linear(dim_out, dim_out))
        self.model = pyg.nn.GINConv(gin_nn)

    def forward(self, batch):
        batch.node_feature = self.model(batch.node_feature, batch.edge_index)
        return batch


class SplineConv(nn.Module):
    def __init__(self, dim_in, dim_out, bias=False, **kwargs):
        super(SplineConv, self).__init__()
        self.model = pyg.nn.SplineConv(dim_in, dim_out,
                                       dim=1, kernel_size=2, bias=bias)

    def forward(self, batch):
        batch.node_feature = self.model(batch.node_feature, batch.edge_index,
                                        batch.edge_feature)
        return batch


class GeneralConv(nn.Module):
    def __init__(self, dim_in, dim_out, bias=False, **kwargs):
        super(GeneralConv, self).__init__()
        self.model = GeneralConvLayer(dim_in, dim_out, bias=bias)

    def forward(self, batch):
        batch.node_feature = self.model(batch.node_feature, batch.edge_index)
        return batch


class GeneralEdgeConv(nn.Module):
    def __init__(self, dim_in, dim_out, bias=False, **kwargs):
        super(GeneralEdgeConv, self).__init__()
        self.model = GeneralEdgeConvLayer(dim_in, dim_out, bias=bias)

    def forward(self, batch):
        batch.node_feature = self.model(batch.node_feature, batch.edge_index,
                                        edge_feature=batch.edge_feature)
        return batch


class GeneralSampleEdgeConv(nn.Module):
    def __init__(self, dim_in, dim_out, bias=False, **kwargs):
        super(GeneralSampleEdgeConv, self).__init__()
        self.model = GeneralEdgeConvLayer(dim_in, dim_out, bias=bias)

    def forward(self, batch):
        edge_mask = torch.rand(batch.edge_index.shape[1]) < cfg.gnn.keep_edge
        edge_index = batch.edge_index[:, edge_mask]
        edge_feature = batch.edge_feature[edge_mask, :]
        batch.node_feature = self.model(batch.node_feature, edge_index,
                                        edge_feature=edge_feature)
        return batch


layer_dict = {
    'gcnconv': GCNConv,
    'sageconv': SAGEConv,
    'gatconv': GATConv,
    'splineconv': SplineConv,
    'ginconv': GINConv,
    'generalconv': GeneralConv,
}
