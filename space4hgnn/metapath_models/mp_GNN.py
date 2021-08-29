import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

from space4hgnn.metapath_models.layers import GeneralLayer
from space4hgnn.models.layers import MultiLinearLayer
from openhgnn.models import BaseModel, register_model, HeteroMLPLayer
from openhgnn.models import HeteroEmbedLayer
from openhgnn.utils import get_nodes_dict


########### Layer ############
def GNNLayer(gnn_type, meta_paths, macro_func, dim_in, dim_out, dropout, act, has_bn, has_l2norm):
    return GeneralLayer(gnn_type, meta_paths, macro_func, dim_in, dim_out, dropout, act, has_bn, has_l2norm)


########### Block: multiple layers ############


class GNNSkipBlock(nn.Module):
    '''Skip block for HGNN'''

    def __init__(self, gnn_type, meta_paths, macro_func, dim_in, dim_out, num_layers, stage_type, dropout, act, has_bn, has_l2norm):
        super(GNNSkipBlock, self).__init__()
        self.stage_type = stage_type
        self.f = nn.ModuleList()
        if num_layers == 1:
            self.f.append(GNNLayer(gnn_type, meta_paths, macro_func, dim_in, dim_out, dropout, act, has_bn, has_l2norm))
        else:
            self.f = []
            for i in range(num_layers - 1):
                d_in = dim_in if i == 0 else dim_out
                self.f.append(GNNLayer(gnn_type, meta_paths, macro_func, d_in, dim_out, dropout, act, has_bn, has_l2norm))
            d_in = dim_in if num_layers == 1 else dim_out
            self.f.append(GNNLayer(gnn_type, meta_paths, macro_func, d_in, dim_out, dropout, act, has_bn, has_l2norm))
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

    def __init__(self, gnn_type, stage_type, meta_paths, macro_func, dim_in, dim_out, num_layers, skip_every, dropout, act, has_bn, has_l2norm):
        super(GNNStackStage, self).__init__()

        for i in range(num_layers):
            d_in = dim_in if i == 0 else dim_out
            layer = GNNLayer(gnn_type, meta_paths, macro_func, d_in, dim_out, dropout, act, has_bn, has_l2norm)
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

    def __init__(self, gnn_type, stage_type, meta_paths, macro_func, dim_in, dim_out, num_layers, skip_every, dropout, act, has_bn, has_l2norm):
        super(GNNSkipStage, self).__init__()
        assert num_layers % skip_every == 0, \
            'cfg.gnn.skip_every must be multiples of cfg.gnn.layer_mp' \
            '(excluding head layer)'
        for i in range(num_layers // skip_every):
            if stage_type == 'skipsum':
                d_in = dim_in if i == 0 else dim_out
            elif stage_type == 'skipconcat':
                d_in = dim_in if i == 0 else dim_in + i * dim_out
            block = GNNSkipBlock(gnn_type, meta_paths, macro_func, d_in, dim_out, skip_every, stage_type, dropout, act, has_bn, has_l2norm)
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


# stage_dict = {**register.stage_dict, **stage_dict}

def GNNPreMP(args):
    linear_list = [args.in_dim] + args.layers_pre_mp * [args.hidden_dim]
    return MultiLinearLayer(linear_list, dropout=args.dropout, act=args.activation, has_bn=args.has_bn,
                            has_l2norm=args.has_l2norm)


def GNNPostMP(args):
    linear_list = args.layers_pre_mp * [args.hidden_dim] + [args.out_dim]
    return MultiLinearLayer(linear_list, dropout=args.dropout, act=args.activation, has_bn=args.has_bn,
                            has_l2norm=args.has_l2norm)


########### Model: start + stage + head ############

@register_model('mp_GNN')
class mp_GNN(BaseModel):
    '''General homogeneous GNN model'''

    @classmethod
    def build_model_from_args(cls, args, hg):
        return cls(args, hg)

    def __init__(self, args, hg, **kwargs):
        """
            Parameters:
            node_encoding_classes - For integer features, gives the number
            of possible integer features to map.
        """
        super(mp_GNN, self).__init__()
        self.category = args.category
        self.meta_paths = []
        for etype in hg.canonical_etypes:
            if etype[0] == args.category:
                for dst_e in hg.canonical_etypes:
                    if etype[0] == dst_e[2] and etype[2] == dst_e[0]:
                        if etype[0] != etype[2]:
                            self.meta_paths.append((etype, dst_e))
        if args.has_feature == False:
            self.embedding_layer = nn.Embedding(hg.num_nodes(self.category), args.hidden_dim)
            args.in_dim = args.hidden_dim

        if args.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(args)

        if args.layers_gnn > 0:
            GNNStage = stage_dict[args.stage_type]
            self.gnn = GNNStage(gnn_type=args.gnn_type,
                                stage_type=args.stage_type,
                                meta_paths=self.meta_paths,
                                macro_func=args.macro_func,
                                dim_in=args.hidden_dim,
                                dim_out=args.hidden_dim,
                                num_layers=args.layers_gnn,
                                skip_every=1,
                                dropout=args.dropout,
                                act=args.activation,
                                has_bn=args.has_bn,
                                has_l2norm=args.has_l2norm)
        #     d_in = self.mp.dim_out

        self.post_mp = GNNPostMP(args)
        self._cached_graph = None
        self._cached_coalesced_graph = {}
        #
        # self.apply(init_weights)

    def forward(self, hg):
        with hg.local_scope():

            if self._cached_graph is None or self._cached_graph is not hg:
                self._cached_graph = hg
                self._cached_coalesced_graph.clear()
                for meta_path in self.meta_paths:
                    mp_g = dgl.metapath_reachable_graph(hg, meta_path)
                    mp_g = dgl.remove_self_loop(mp_g)
                    mp_g = dgl.add_self_loop(mp_g)
                    self._cached_coalesced_graph[meta_path] = mp_g

            if hasattr(self, 'embedding_layer'):
                h = self.embedding_layer(torch.arange(hg.num_nodes(self.category)).to(hg.device))
            else:
                h = hg.nodes[self.category].data['h']
            if hasattr(self, 'pre_mp'):
                h = self.pre_mp(h)
            if hasattr(self, 'gnn'):
                out_h = self.gnn(self._cached_coalesced_graph, h)
            if hasattr(self, 'post_mp'):
                out_h = self.post_mp(out_h)
        return {self.category: out_h}
