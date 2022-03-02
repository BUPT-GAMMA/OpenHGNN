import torch
import torch.nn as nn
import torch.nn.functional as F
from ..layers.GeneralHGNNLayer import HeteroGeneralLayer
from ..layers.GeneralGNNLayer import GeneralLayer


def HGNNLayer(gnn_type, rel_names, dim_in, dim_out, dropout, act, has_bn, has_l2norm, **kwargs):
    return HeteroGeneralLayer(gnn_type, rel_names, dim_in, dim_out, dropout, act, has_bn, has_l2norm, **kwargs)


class HGNNSkipBlock(nn.Module):
    '''Skip block for HGNN'''

    def __init__(self, gnn_type, rel_names, dim_in, dim_out, num_layers,
                 stage_type, dropout, act, has_bn, has_l2norm, **kwargs):
        super(HGNNSkipBlock, self).__init__()
        self.stage_type = stage_type
        self.f = nn.ModuleList()
        if num_layers == 1:
            self.f.append(HGNNLayer(gnn_type, rel_names, dim_in, dim_out, dropout, act, has_bn, has_l2norm, **kwargs))
        else:
            self.f = []
            for i in range(num_layers - 1):
                d_in = dim_in if i == 0 else dim_out
                self.f.append(HGNNLayer(gnn_type, rel_names, d_in, dim_out, dropout, act, has_bn, has_l2norm, **kwargs))
            d_in = dim_in if num_layers == 1 else dim_out
            self.f.append(HGNNLayer(gnn_type, rel_names, d_in, dim_out, dropout, act, has_bn, has_l2norm, **kwargs))
        self.act = act
        if stage_type == 'skipsum':
            assert dim_in == dim_out, 'Sum skip must have same dim_in, dim_out'

    def forward(self, g, h):
        h_0 = h
        for layer in self.f:
            h = layer(g, h)
        out_h = {}
        for key, value in h_0.items():

            if self.stage_type == 'skipsum':
                out_h[key] = self.act(h[key] + h_0[key])
            elif self.stage_type == 'skipconcat':
                out_h[key] = self.act(torch.cat((h[key], h_0[key]), 1))
            else:
                raise ValueError('stage_type must in [skipsum, skipconcat]')

        return out_h


########### Stage: NN except start and head ############

class HGNNStackStage(nn.Module):
    '''Simple Stage that stack GNN layers'''

    def __init__(self, gnn_type, rel_names, stage_type, dim_in, dim_out,
                 num_layers, skip_every, dropout, act, has_bn, has_l2norm, **kwargs):
        super(HGNNStackStage, self).__init__()

        for i in range(num_layers):
            d_in = dim_in if i == 0 else dim_out
            layer = HGNNLayer(gnn_type, rel_names, d_in, dim_out, dropout, act, has_bn, has_l2norm, **kwargs)
            self.add_module('layer{}'.format(i), layer)
        self.dim_out = dim_out
        self.has_l2norm = has_l2norm

    def forward(self, g, h_dict):
        for layer in self.children():
            h_dict = layer(g, h_dict)
        if self.has_l2norm:
            for name, batch_h in h_dict.items():
                h_dict[name] = F.normalize(batch_h, p=2, dim=-1)
        return h_dict


class HGNNSkipStage(nn.Module):
    ''' Stage with skip connections'''

    def __init__(self, gnn_type, rel_names, stage_type, dim_in, dim_out,
                 num_layers, skip_every, dropout, act, has_bn, has_l2norm, **kwargs):
        super(HGNNSkipStage, self).__init__()
        assert num_layers % skip_every == 0, \
            'cfg.gnn.skip_every must be multiples of cfg.gnn.layer_mp' \
            '(excluding head layer)'
        for i in range(num_layers // skip_every):
            if stage_type == 'skipsum':
                d_in = dim_in if i == 0 else dim_out
            elif stage_type == 'skipconcat':
                d_in = dim_in if i == 0 else dim_in + i * dim_out
            block = HGNNSkipBlock(gnn_type, rel_names, d_in, dim_out, skip_every, stage_type, dropout, act, has_bn,
                                  has_l2norm, **kwargs)
            self.add_module('block{}'.format(i), block)
        if stage_type == 'skipconcat':
            self.dim_out = d_in + dim_out
        else:
            self.dim_out = dim_out
        self.has_l2norm = has_l2norm

    def forward(self, g, h_dict):
        for layer in self.children():
            h_dict = layer(g, h_dict)
        if self.has_l2norm:
            for name, batch_h in h_dict.items():
                h_dict[name] = F.normalize(batch_h, p=2, dim=-1)
        return h_dict


def GNNLayer(gnn_type, dim_in, dim_out, dropout, act, has_bn, has_l2norm, **kwargs):
    return GeneralLayer(gnn_type, dim_in, dim_out, dropout, act, has_bn, has_l2norm, **kwargs)


class GNNSkipBlock(nn.Module):
    r"""
    Skip block for HGNN
    """

    def __init__(self, gnn_type, dim_in, dim_out, num_layers, stage_type, dropout, act, has_bn, has_l2norm, *args, **kwargs):
        super(GNNSkipBlock, self).__init__()
        self.stage_type = stage_type
        self.f = nn.ModuleList()
        if num_layers == 1:
            self.f.append(GNNLayer(gnn_type, dim_in,
                                   dim_out, dropout, None, has_bn, has_l2norm, **kwargs))
        else:
            for i in range(num_layers - 1):
                d_in = dim_in if i == 0 else dim_out
                self.f.append(GNNLayer(gnn_type, d_in, dim_out, dropout, act, has_bn, has_l2norm, **kwargs))
            d_in = dim_in if num_layers == 1 else dim_out
            self.f.append(GNNLayer(gnn_type, d_in, dim_out, dropout, None, has_bn, has_l2norm, **kwargs))
        self.act = act
        if stage_type == 'skipsum':
            assert dim_in == dim_out, 'Sum skip must have same dim_in, dim_out'

    def forward(self, g, h):
        h_0 = h
        for layer in self.f:
            h = layer(g, h)
        if self.stage_type == 'skipsum':
            h = h + h_0
        elif self.stage_type == 'skipconcat':
            h = torch.cat((h_0, h), 1)
        else:
            raise ValueError('stage_type must in [skipsum, skipconcat]')
        h = self.act(h)
        return h


########### Stage: NN except start and head ############

class GNNStackStage(nn.Module):
    '''Simple Stage that stack GNN layers'''

    def __init__(self, gnn_type, dim_in, dim_out,
                 num_layers, dropout, act, has_bn, has_l2norm, *args, **kwargs):
        super(GNNStackStage, self).__init__()

        for i in range(num_layers):
            d_in = dim_in if i == 0 else dim_out
            # if i == num_layers - 1:
            #     layer = GNNLayer(gnn_type, d_in, dim_out, dropout, None, has_bn, has_l2norm, **kwargs)
            # else:
            layer = GNNLayer(gnn_type, d_in, dim_out, dropout, act, has_bn, has_l2norm, **kwargs)
            self.add_module('layer{}'.format(i), layer)
        self.dim_out = dim_out
        self.has_l2norm = has_l2norm

    def forward(self, g, h):
        for layer in self.children():
            h = layer(g, h)
        if self.has_l2norm:
            h = F.normalize(h, p=2, dim=-1)
        return h


class GNNSkipStage(nn.Module):
    ''' Stage with skip connections'''

    def __init__(self, gnn_type, stage_type,dim_in, dim_out, num_layers,
                 skip_every, dropout, act, has_bn, has_l2norm, *args, **kwargs):
        super(GNNSkipStage, self).__init__()
        assert num_layers % skip_every == 0, \
            'cfg.gnn.skip_every must be multiples of cfg.gnn.layer_mp' \
            '(excluding head layer)'
        for i in range(num_layers // skip_every):
            if stage_type == 'skipsum':
                d_in = dim_in if i == 0 else dim_out
            elif stage_type == 'skipconcat':
                d_in = dim_in if i == 0 else dim_in + i * dim_out
            # if i == num_layers // skip_every - 1:
            #     block = GNNSkipBlock(gnn_type, d_in, dim_out, skip_every, stage_type, dropout, None, has_bn, has_l2norm,
            #                          **kwargs)
            # else:
            block = GNNSkipBlock(gnn_type, d_in, dim_out, skip_every, stage_type, dropout, act, has_bn, has_l2norm, **kwargs)
            self.add_module('block{}'.format(i), block)
        if stage_type == 'skipconcat':
            self.dim_out = d_in + dim_out
        else:
            self.dim_out = dim_out
        self.has_l2norm = has_l2norm

    def forward(self, g, h):
        for layer in self.children():
            h = layer(g, h)
        if self.has_l2norm:
            h = F.normalize(h, p=2, dim=-1)
        return h