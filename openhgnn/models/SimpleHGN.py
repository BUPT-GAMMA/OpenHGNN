import dgl
import dgl.function as Fn
from dgl.ops import edge_softmax

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import BaseModel, register_model


@register_model('SimpleHGN')
class SimpleHGN(BaseModel):
    @classmethod
    def build_model_from_args(cls, args, hg):
        heads = [args.num_heads] * args.num_layers + [1]
        return cls(args.edge_dim,
                   args.num_edge * 2,
                   [args.hidden_dim],
                   args.h_dim,
                   len(hg.ntypes),
                   args.num_layers,
                   heads,
                   args.feats_drop_rate,
                   args.attn_drop_rate,
                   args.slope,
                   True,
                   args.beta
                   )

    def __init__(self, edge_dim, num_etypes, in_dims, num_hidden, num_classes,
                num_layers, heads, feat_drop, attn_drop, negative_slope,
                residual, beta):
        super(SimpleHGN, self).__init__()
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = F.elu

        # input projection (no residual)
        self.gat_layers.append(
            SimpleHGNConv(
                edge_dim,
                in_dims[0],
                num_hidden,
                heads[0],
                num_etypes,
                feat_drop,
                attn_drop,
                negative_slope,
                False,
                self.activation,
                beta=beta,
            )
        )
        # hidden layers
        for l in range(1, num_layers):  # noqa E741
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(
                SimpleHGNConv(
                    edge_dim,
                    num_hidden * heads[l - 1],
                    num_hidden,
                    heads[l],
                    num_etypes,
                    feat_drop,
                    attn_drop,
                    negative_slope,
                    residual,
                    self.activation,
                    beta=beta,
                )
            )
        # output projection
        self.gat_layers.append(
            SimpleHGNConv(
                edge_dim,
                num_hidden * heads[-2],
                num_classes,
                heads[-1],
                num_etypes,
                feat_drop,
                attn_drop,
                negative_slope,
                residual,
                None,
                beta=beta,
            )
        )

    def forward(self, g, h_dict = None):
        with g.local_scope():
            g.ndata['h'] = h_dict
            graph = dgl.to_homogeneous(g, ndata = ['h'])
            h = graph.ndata['h']
            for l in range(self.num_layers):  # noqa E741
                h = self.gat_layers[l](graph, h)
                h = h.flatten(1)

        graph.ndata['h'] = h
        g_2 = dgl.to_heterogeneous(graph, g.ntypes, g.etypes)
        h_dict = g_2.ndata['h']

        return h_dict

class SimpleHGNConv(nn.Module):
    def __init__(self,
                 edge_feats,
                 in_features,
                 out_features,
                 num_heads,
                 num_etype,
                 feat_drop=0.0,
                 attn_drop=0.5,
                 negative_slope=0.2,
                 residual=True,
                 activation=F.elu,
                 beta=0.0
                 ):
        super(SimpleHGNConv, self).__init__()
        self.edge_feats = edge_feats
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.num_etype = num_etype

        self.edge_emb = nn.Parameter(torch.zeros(size=(num_etype, edge_feats)))

        self.W = nn.Parameter(torch.FloatTensor(
            in_features, out_features * num_heads))
        self.W_e = nn.Parameter(torch.FloatTensor(
            edge_feats, edge_feats * num_heads))

        self.a_l = nn.Parameter(torch.empty(size=(1, num_heads, out_features)))
        self.a_r = nn.Parameter(torch.empty(size=(1, num_heads, out_features)))
        self.a_e = nn.Parameter(torch.empty(size=(1, num_heads, edge_feats)))

        nn.init.xavier_uniform_(self.edge_emb, gain=1.414)
        nn.init.xavier_uniform_(self.W, gain=1.414)
        nn.init.xavier_uniform_(self.W_e, gain=1.414)
        nn.init.xavier_uniform_(self.a_l.data, gain=1.414)
        nn.init.xavier_uniform_(self.a_r.data, gain=1.414)
        nn.init.xavier_uniform_(self.a_e.data, gain=1.414)

        self.feat_drop = nn.Dropout(feat_drop)
        self.dropout = nn.Dropout(attn_drop)
        self.leakyrelu = nn.LeakyReLU(negative_slope)
        self.activation = activation

        if residual:
            self.residual = nn.Linear(in_features, out_features * num_heads)
        else:
            self.register_buffer("residual", None)

        self.beta = beta

    def forward(self, g, h):
        emb = self.feat_drop(h)
        emb = torch.matmul(emb, self.W).view(-1, self.num_heads, self.out_features)
        emb[torch.isnan(emb)] = 0.0

        e = torch.matmul(self.edge_emb, self.W_e).view(-1,
                                                       self.num_heads, self.edge_feats)

        row = g.edges()[0]
        col = g.edges()[1]
        tp = g.edata['_TYPE']
        # tp = g.edge_type

        h_l = (self.a_l * emb).sum(dim=-1)[row]
        h_r = (self.a_r * emb).sum(dim=-1)[col]
        h_e = (self.a_e * e).sum(dim=-1)[tp]

        edge_attention = self.leakyrelu(h_l + h_r + h_e)
        edge_attention = edge_softmax(g, edge_attention)

        if 'alpha' in g.edata.keys():
            res_attn = g.edata['alpha']
            edge_attention = edge_attention * \
                             (1 - self.beta) + res_attn * self.beta

        with g.local_scope():
            h_prime = []
            emb = emb.permute(1, 0, 2).contiguous()
            for i in range(self.num_heads):
                g.edata['alpha'] = edge_attention[:, i]
                g.srcdata.update({'emb': emb[i]})
                g.update_all(Fn.u_mul_e('emb', 'alpha', 'm'),
                             Fn.sum('m', 'emb'))
                h_prime.append(g.ndata['emb'])
            h_output = torch.cat(h_prime, dim=1)

        g.edata['alpha'] = edge_attention
        if self.residual:
            res = self.residual(h)
            h_output += res
        h_output = self.activation(h_output)

        return h_output