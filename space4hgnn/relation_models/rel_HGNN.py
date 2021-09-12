import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import HeteroGeneralLayer
from openhgnn.models import BaseModel, register_model
from space4hgnn.models.MLP import HGNNPostMP, HGNNPreMP
########### Layer ############

def HGNNLayer(gnn_type, rel_names, dim_in, dim_out, dropout, act, has_bn, has_l2norm, **kwargs):
    return HeteroGeneralLayer(gnn_type, rel_names, dim_in, dim_out, dropout, act, has_bn, has_l2norm, **kwargs)


########### Block: multiple layers ############

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
            block = HGNNSkipBlock(gnn_type, rel_names, d_in, dim_out, skip_every, stage_type, dropout, act, has_bn, has_l2norm, **kwargs)
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


stage_dict = {
    'stack': HGNNStackStage,
    'skipsum': HGNNSkipStage,
    'skipconcat': HGNNSkipStage,
}


########### Model: start + stage + head ############

@register_model('relation_HGNN')
class relation_HGNN(BaseModel):
    '''General homogeneous GNN model'''
    @classmethod
    def build_model_from_args(cls, args, hg):
        out_node_type = args.out_node_type
        return cls(args, hg, out_node_type)

    def __init__(self, args, hg, out_node_type, **kwargs):
        """
        """
        super(relation_HGNN, self).__init__()
        self.out_node_type = out_node_type
        if args.layers_pre_mp - 1 > 0:
            self.pre_mp = HGNNPreMP(args, hg.ntypes, args.layers_pre_mp, args.hidden_dim, args.hidden_dim)

        if args.layers_gnn > 0:
            HGNNStage = stage_dict[args.stage_type]
            self.hgnn = HGNNStage(gnn_type=args.gnn_type,
                                  rel_names=hg.etypes,
                                  stage_type=args.stage_type,
                                  dim_in=args.hidden_dim,
                                  dim_out=args.hidden_dim,
                                  num_layers=args.layers_gnn,
                                  skip_every=1,
                                  dropout=args.dropout,
                                  act=args.activation,
                                  has_bn=args.has_bn,
                                  has_l2norm=args.has_l2norm,
                                  macro_func=args.macro_func)
        #     d_in = self.mp.dim_out

        gnn_out_dim = self.hgnn.dim_out
        self.post_mp = HGNNPostMP(args, self.out_node_type, args.layers_post_mp, gnn_out_dim, args.out_dim)
        #
        # self.apply(init_weights)

    def forward(self, hg, h_dict):
        with hg.local_scope():
            if hasattr(self, 'pre_mp'):
                h_dict = self.pre_mp(h_dict)
            if hasattr(self, 'hgnn'):
                h_dict = self.hgnn(hg, h_dict)
            if hasattr(self, 'post_mp'):
                out_h = {}
                for key, value in h_dict.items():
                    if key in self.out_node_type:
                        out_h[key] = value
                out_h = self.post_mp(out_h)
        return out_h