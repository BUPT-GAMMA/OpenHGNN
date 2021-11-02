import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl.data.utils import load_graphs
from dgl.sampling import sample_neighbors
from dgl.nn.pytorch import GATConv, GraphConv
from openhgnn.models import BaseModel, register_model
from ..utils.utils import extract_metapaths


def init_drop(dropout):
    if dropout > 0:
        return nn.Dropout(dropout)
    else:
        return lambda x: x


@register_model('HeCo')
class HeCo(BaseModel):
    @classmethod
    def build_model_from_args(cls, args, hg):
        if args.meta_paths is None:
            meta_paths = extract_metapaths(args.category, hg.canonical_etypes)
        else:
            meta_paths = args.meta_paths
        schema = []
        for etype in hg.canonical_etypes:
            if etype[2] == args.category:
                schema.append(etype)
        return cls(meta_paths=meta_paths, network_schema=schema, category=args.category,
                   hidden_size=args.hidden_dim, feat_drop=args.feat_drop,
                   attn_drop=args.attn_drop, sample_rate=args.sample_rate, tau=args.tau, lam=args.lam)

    def __init__(self, meta_paths, network_schema, category, hidden_size, feat_drop, attn_drop
                 , sample_rate, tau, lam):
        super(HeCo, self).__init__()
        self.category = category  # target node type
        self.feat_drop = init_drop(feat_drop)
        self.attn_drop = attn_drop
        self.mp = Mp_encoder(meta_paths, hidden_size, self.attn_drop)
        self.sc = Sc_encoder(network_schema, hidden_size, self.attn_drop, sample_rate, self.category)
        self.contrast = Contrast(hidden_size, tau, lam)

    def forward(self, g, h_dict, pos):
        new_h = {}
        for key, value in h_dict.items():
            new_h[key] = F.elu(self.feat_drop(value))
        z_mp = self.mp(g, new_h[self.category])
        z_sc = self.sc(g, new_h)
        loss = self.contrast(z_mp, z_sc, pos)
        return loss

    def get_embeds(self, g, h_dict):
        z_mp = F.elu(h_dict[self.category])
        z_mp = self.mp(g, z_mp)
        return z_mp.detach()

class SelfAttention(nn.Module):
    def __init__(self, hidden_dim, attn_drop, txt):
        super(SelfAttention, self).__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim, bias=True)
        nn.init.xavier_normal_(self.fc.weight, gain=1.414)

        self.tanh = nn.Tanh()
        self.att = nn.Parameter(torch.empty(size=(1, hidden_dim)), requires_grad=True)
        nn.init.xavier_normal_(self.att.data, gain=1.414)

        self.softmax = nn.Softmax()
        self.attn_drop = init_drop(attn_drop)
        self.txt = txt

    def forward(self, embeds):
        beta = []
        attn_curr = self.attn_drop(self.att)
        for embed in embeds:
            sp = self.tanh(self.fc(embed)).mean(dim=0)
            beta.append(attn_curr.matmul(sp.t()))
        beta = torch.cat(beta, dim=-1).view(-1)
        beta = self.softmax(beta)
        print(self.txt, beta.data.cpu().numpy())  # semantic attention
        z_mp = 0
        for i in range(len(embeds)):
            z_mp += embeds[i]*beta[i]
        return z_mp



class Mp_encoder(nn.Module):
    def __init__(self, meta_paths, hidden_size, attn_drop):
        super(Mp_encoder, self).__init__()
        # One GCN layer for each meta path based adjacency matrix
        self.act = nn.PReLU()
        self.gcn_layers = nn.ModuleList()
        for i in range(len(meta_paths)):
            one_layer = GraphConv(hidden_size, hidden_size, activation=self.act, allow_zero_in_degree=True)
            one_layer.reset_parameters()
            self.gcn_layers.append(one_layer)
        self.meta_paths = list(tuple(meta_path) for meta_path in meta_paths)
        self._cached_graph = None
        self._cached_coalesced_graph = {}
        self.semantic_attention = SelfAttention(hidden_size, attn_drop, "mp")

    def forward(self, g, h):
        semantic_embeddings = []
        if self._cached_graph is None or self._cached_graph is not g:
            self._cached_graph = g
            self._cached_coalesced_graph.clear()
            for meta_path in self.meta_paths:
                self._cached_coalesced_graph[meta_path] = dgl.metapath_reachable_graph(
                        g, meta_path)

        for i, meta_path in enumerate(self.meta_paths):
            new_g = self._cached_coalesced_graph[meta_path]
            one = self.gcn_layers[i](new_g, h)
            semantic_embeddings.append(one)  # node level attention
        return self.semantic_attention(semantic_embeddings)


class Sc_encoder(nn.Module):
    def __init__(self, network_schema, hidden_size, attn_drop, sample_rate, category):
        super(Sc_encoder, self).__init__()
        self.gat_layers = nn.ModuleList()
        for i in range(len(network_schema)):
            one_layer = GATConv((hidden_size, hidden_size), hidden_size, num_heads=1, attn_drop=attn_drop, allow_zero_in_degree=True)
            one_layer.reset_parameters()
            self.gat_layers.append(one_layer)
        self.network_schema = list(tuple(ns) for ns in network_schema)
        self._cached_graph = None
        self._cached_coalesced_graph = {}
        self.inter = SelfAttention(hidden_size, attn_drop, "sc")
        self.sample_rate = sample_rate
        self.category = category

    def forward(self, g, h):
        intra_embeddings = []
        for i, network_schema in enumerate(self.network_schema):
            src_type = network_schema[0]
            one_graph = g[network_schema]
            cate_num = torch.arange(0, g.num_nodes(self.category)).to(g.device)
            sub_graph = sample_neighbors(one_graph, {self.category: cate_num}, {network_schema[1]: self.sample_rate[src_type]}, replace=True)
            one = self.gat_layers[i](sub_graph, (h[src_type], h[self.category]))
            one = one.squeeze(1)
            intra_embeddings.append(one)
        return self.inter(intra_embeddings)


class Contrast(nn.Module):
    def __init__(self, hidden_dim, tau, lam):
        super(Contrast, self).__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.tau = tau
        self.lam = lam
        for model in self.proj:
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight, gain=1.414)

    def sim(self, z1, z2):
        z1_norm = torch.norm(z1, dim=-1, keepdim=True)
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)
        dot_numerator = torch.mm(z1, z2.t())
        dot_denominator = torch.mm(z1_norm, z2_norm.t())
        sim_matrix = torch.exp(dot_numerator / dot_denominator / self.tau)
        return sim_matrix

    def forward(self, z_mp, z_sc, pos):
        z_proj_mp = self.proj(z_mp)
        z_proj_sc = self.proj(z_sc)
        matrix_mp2sc = self.sim(z_proj_mp, z_proj_sc)
        matrix_sc2mp = matrix_mp2sc.t()
        
        matrix_mp2sc = matrix_mp2sc/(torch.sum(matrix_mp2sc, dim=1).view(-1, 1) + 1e-8)
        lori_mp = -torch.log(matrix_mp2sc.mul(pos.to_dense()).sum(dim=-1)).mean()

        matrix_sc2mp = matrix_sc2mp / (torch.sum(matrix_sc2mp, dim=1).view(-1, 1) + 1e-8)
        lori_sc = -torch.log(matrix_sc2mp.mul(pos.to_dense()).sum(dim=-1)).mean()
        return self.lam * lori_mp + (1 - self.lam) * lori_sc

'''logreg'''
class LogReg(nn.Module):
    r"""
    Parameters
    ----------
    ft_in : int
        Size of hid_units
    nb_class : int
        The number of category's types
    """
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret

