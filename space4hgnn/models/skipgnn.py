import torch
import torch.nn as nn
import torch.nn.functional as F
from space4hgnn.models.layers import GeneralLayer


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
                                   dim_out, dropout, act, has_bn, has_l2norm, **kwargs))
        else:
            for i in range(num_layers - 1):
                d_in = dim_in if i == 0 else dim_out
                self.f.append(GNNLayer(gnn_type, d_in, dim_out, dropout, act, has_bn, has_l2norm, **kwargs))
            d_in = dim_in if num_layers == 1 else dim_out
            self.f.append(GNNLayer(gnn_type, d_in, dim_out, dropout, act, has_bn, has_l2norm, **kwargs))
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

stage_dict = {
    'stack': GNNStackStage,
    'skipsum': GNNSkipStage,
    'skipconcat': GNNSkipStage,
}