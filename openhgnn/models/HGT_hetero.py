import dgl
import math
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.functional import edge_softmax
from . import BaseModel, register_model, HeteroEmbedLayer
from ..utils import get_nodes_dict


@register_model('HGT')
class HGT(BaseModel):
    @classmethod
    def build_model_from_args(cls, args, hg):
        node_dict = {}
        edge_dict = {}
        for ntype in hg.ntypes:
            node_dict[ntype] = len(node_dict)
        for etype in hg.etypes:
            edge_dict[etype] = len(edge_dict)
            hg.edges[etype].data['id'] = th.ones(hg.number_of_edges(etype), dtype=th.long).to(args.device) * edge_dict[etype]
        in_dim = hg.nodes['paper'].data['h'].shape[1]
        return cls(node_dict, edge_dict, get_nodes_dict(hg), in_dim, args.hidden_dim, args.out_dim, args.n_layers, args.num_heads, args.dropout)

    def __init__(self, node_dict, edge_dict, n_nodes, in_dim, hidden_dim, out_dim, n_layers, n_heads, dropout, use_norm=True):
        super(HGT, self).__init__()
        self.node_dict = node_dict
        self.edge_dict = edge_dict
       # self.h_n_dict = HeteroEmbedLayer(n_nodes, in_dim)

        self.gcs = nn.ModuleList()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.n_layers = n_layers
        self.adapt_ws = nn.ModuleList()
        for t in range(len(node_dict)):
            self.adapt_ws.append(nn.Linear(in_dim, hidden_dim))
        for _ in range(n_layers):
            self.gcs.append(HGTLayer(hidden_dim, hidden_dim, node_dict, edge_dict, n_heads, dropout, use_norm = use_norm))
        self.out = nn.Linear(hidden_dim, out_dim)

    def forward(self, G, h_in=None):
        #h_in = self.h_n_dict.forward()
        h = {}
        for ntype in G.ntypes:
            n_id = self.node_dict[ntype]
            h[ntype] = F.gelu(self.adapt_ws[n_id](h_in[ntype]))
        for i in range(self.n_layers):
            h = self.gcs[i](G, h)
        return {'paper': self.out(h['paper'])}


class HGTLayer(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 node_dict,
                 edge_dict,
                 n_heads,
                 dropout = 0.2,
                 use_norm = False):
        super(HGTLayer, self).__init__()

        self.in_dim        = in_dim
        self.out_dim       = out_dim
        self.node_dict     = node_dict
        self.edge_dict     = edge_dict
        self.num_types     = len(node_dict)
        self.num_relations = len(edge_dict)
        self.total_rel     = self.num_types * self.num_relations * self.num_types
        self.n_heads       = n_heads
        self.d_k           = out_dim // n_heads
        self.sqrt_dk       = math.sqrt(self.d_k)
        self.att           = None

        self.k_linears   = nn.ModuleList()
        self.q_linears   = nn.ModuleList()
        self.v_linears   = nn.ModuleList()
        self.a_linears   = nn.ModuleList()
        self.norms       = nn.ModuleList()
        self.use_norm    = use_norm

        for t in range(self.num_types):
            self.k_linears.append(nn.Linear(in_dim,   out_dim))
            self.q_linears.append(nn.Linear(in_dim,   out_dim))
            self.v_linears.append(nn.Linear(in_dim,   out_dim))
            self.a_linears.append(nn.Linear(out_dim,  out_dim))
            if use_norm:
                self.norms.append(nn.LayerNorm(out_dim))

        self.relation_pri   = nn.Parameter(th.ones(self.num_relations, self.n_heads))
        self.relation_att   = nn.Parameter(th.Tensor(self.num_relations, n_heads, self.d_k, self.d_k))
        self.relation_msg   = nn.Parameter(th.Tensor(self.num_relations, n_heads, self.d_k, self.d_k))
        self.skip           = nn.Parameter(th.ones(self.num_types))
        self.drop           = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.relation_att)
        nn.init.xavier_uniform_(self.relation_msg)

    def forward(self, G, h):
        with G.local_scope():
            node_dict, edge_dict = self.node_dict, self.edge_dict
            for srctype, etype, dsttype in G.canonical_etypes:
                sub_graph = G[srctype, etype, dsttype]

                k_linear = self.k_linears[node_dict[srctype]]
                v_linear = self.v_linears[node_dict[srctype]]
                q_linear = self.q_linears[node_dict[dsttype]]

                k = k_linear(h[srctype]).view(-1, self.n_heads, self.d_k)
                v = v_linear(h[srctype]).view(-1, self.n_heads, self.d_k)
                q = q_linear(h[dsttype]).view(-1, self.n_heads, self.d_k)

                e_id = self.edge_dict[etype]

                relation_att = self.relation_att[e_id]
                relation_pri = self.relation_pri[e_id]
                relation_msg = self.relation_msg[e_id]

                k = th.einsum("bij,ijk->bik", k, relation_att)
                v = th.einsum("bij,ijk->bik", v, relation_msg)

                sub_graph.srcdata['k'] = k
                sub_graph.dstdata['q'] = q
                sub_graph.srcdata['v_%d' % e_id] = v

                sub_graph.apply_edges(fn.v_dot_u('q', 'k', 't'))
                attn_score = sub_graph.edata.pop('t').sum(-1) * relation_pri / self.sqrt_dk
                attn_score = edge_softmax(sub_graph, attn_score, norm_by='dst')

                sub_graph.edata['t'] = attn_score.unsqueeze(-1)

            G.multi_update_all({etype : (fn.u_mul_e('v_%d' % e_id, 't', 'm'), fn.sum('m', 't')) \
                                for etype, e_id in edge_dict.items()}, cross_reducer = 'mean')

            new_h = {}
            for ntype in G.ntypes:
                '''
                    Step 3: Target-specific Aggregation
                    x = norm( W[node_type] * gelu( Agg(x) ) + x )
                '''
                n_id = node_dict[ntype]
                alpha = th.sigmoid(self.skip[n_id])
                t = G.nodes[ntype].data['t'].view(-1, self.out_dim)
                trans_out = self.drop(self.a_linears[n_id](t))
                trans_out = trans_out * alpha + h[ntype] * (1-alpha)
                if self.use_norm:
                    new_h[ntype] = self.norms[n_id](trans_out)
                else:
                    new_h[ntype] = trans_out
            return new_h
