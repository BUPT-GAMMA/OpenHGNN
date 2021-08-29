
import torch
import torch.nn as nn
import dgl
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.pytorch import edge_softmax, GATConv
from openhgnn.models import BaseModel, register_model, hetero_linear


@register_model('GAT')
class GAT(BaseModel):
    @classmethod
    def build_model_from_args(cls, args, hg):
        heads = [args.num_heads] * args.num_layers + [1]
        return cls(hg,
                   args.hidden_dim,
                   args.out_dim,
                   args.num_layers,
                   heads,
                   F.elu,
                   args.dropout,
                   args.dropout,
                   args.slope,
                   False)

    def __init__(self,
                 hg,
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
        self.hg = hg
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation

        # input projection (no residual)
        self.gat_layers.append(GATConv(
            num_hidden, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                num_hidden * heads[l - 1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation))
        # output projection
        self.gat_layers.append(GATConv(
            num_hidden * heads[-2], num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None))


        linear_list1 = []
        for ntype in self.hg.ntypes:
            in_dim = self.hg.nodes[ntype].data['h'].shape[1]
            linear_list1.append((ntype, in_dim, num_hidden))
        # * ================== Project feature Layer==================
        self.hetero_linear = hetero_linear(linear_list1)


    def forward(self, hg):
        with hg.local_scope():
            hg.ndata['h'] = self.hetero_linear(hg.ndata['h'])
            homo_g = dgl.to_homogeneous(hg, ndata=['h'])
            homo_g = dgl.remove_self_loop(homo_g)
            homo_g = dgl.add_self_loop(homo_g)
            h = homo_g.ndata.pop('h')

            for l in range(self.num_layers):
                h = self.gat_layers[l](homo_g, h).flatten(1)
            # output projection
            h = self.gat_layers[-1](homo_g, h).mean(1)
            # homo_g.ndata['out_h'] = h
            # out_h = dgl.to_heterogeneous(homo_g, hg.ntypes, hg.etypes).ndata['out_h']
            out_h = self.h2dict(h, hg.ndata['h'])
        return out_h

    def h2dict(self, h, hdict):
        pre = 0
        for i, value in hdict.items():
            hdict[i] = h[pre:value.shape[0]+pre]
            pre += value.shape[0]
        return hdict