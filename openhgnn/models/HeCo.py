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
    r"""

    Description
    -----------
    **Title:** Self-supervised Heterogeneous Graph Neural Network with Co-contrastive Learning

    **Authors:** Xiao Wang, Nian Liu, Hui Han, Chuan Shi

    HeCo was introduced in `[paper] <http://shichuan.org/doc/112.pdf>`_
    and parameters are defined as follows:

    Parameters
    ----------
        meta_paths : dict
            Extract metapaths from graph
        network_schema : dict
            Directed edges from other types to target type
        category : string
            The category of the nodes to be classificated
        hidden_size : int
            Hidden units size
        feat_drop : float
            Dropout rate for projected feature
        attn_drop : float
            Dropout rate for attentions used in two view guided encoders
        sample_rate : dict
            The nuber of neighbors of each type sampled for network schema view
        tau : float
            Temperature parameter used for contrastive loss
        lam : float
            Balance parameter for two contrastive losses

    """

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
        r"""
        Description
        -----------
        This is the forward part of model HeCo.

        Parameters
        ----------
        g : DGLGraph
            A DGLGraph
        h_dict: dict
            Projected features after linear projection
        pos: matrix
            A matrix to indicate the postives for each node

        Returns
        -------
        loss : float
            The optimize objective

        Note
        -----------
        Pos matrix is pre-defined by users. The relative tool is given in original code.
        """
        new_h = {}
        for key, value in h_dict.items():
            new_h[key] = F.elu(self.feat_drop(value))
        z_mp = self.mp(g, new_h[self.category])
        z_sc = self.sc(g, new_h)
        loss = self.contrast(z_mp, z_sc, pos)
        return loss

    def get_embeds(self, g, h_dict):
        r"""
        Description
        -----------
        This is to get final embeddings of target nodes

        """
        z_mp = F.elu(h_dict[self.category])
        z_mp = self.mp(g, z_mp)
        return z_mp.detach()


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim, attn_drop, txt):
        r"""
        Description
        -----------
        This part is used to calculate type-level attention and semantic-level attention, and utilize them to generate :math:`z^{sc}` and :math:`z^{mp}`.

        .. math::
           w_{n}&=\frac{1}{|V|}\sum\limits_{i\in V} \textbf{a}^\top \cdot \tanh\left(\textbf{W}h_i^{n}+\textbf{b}\right) \\
           \beta_{n}&=\frac{\exp\left(w_{n}\right)}{\sum_{i=1}^M\exp\left(w_{i}\right)} \\
           z &= \sum_{n=1}^M \beta_{n}\cdot h^{n}

        Parameters
        ----------
        txt : str
            A str to identify view, MP or SC

        Returns
        -------
        z : matrix
            The fused embedding matrix

        """
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
        z = 0
        for i in range(len(embeds)):
            z += embeds[i] * beta[i]
        return z


class Mp_encoder(nn.Module):
    def __init__(self, meta_paths, hidden_size, attn_drop):
        r"""
        Description
        -----------
        This part is to encode meta-path view.

        Returns
        -------
        z_mp : matrix
            The embedding matrix under meta-path view.

        """
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
        z_mp = self.semantic_attention(semantic_embeddings)
        return z_mp


class Sc_encoder(nn.Module):
    def __init__(self, network_schema, hidden_size, attn_drop, sample_rate, category):
        r"""
        Description
        -----------
        This part is to encode network schema view.

        Returns
        -------
        z_mp : matrix
            The embedding matrix under network schema view.

        Note
        -----------
        There is a different sampling strategy between original code and this code. In original code, the authors implement sampling without replacement if the number of neighbors exceeds a threshold,
        and with replacement if not. In this version, we simply use the API dgl.sampling.sample_neighbors to implement this operation, and set replacement as True.

        """
        super(Sc_encoder, self).__init__()
        self.gat_layers = nn.ModuleList()
        for i in range(len(network_schema)):
            one_layer = GATConv((hidden_size, hidden_size), hidden_size, num_heads=1, attn_drop=attn_drop,
                                allow_zero_in_degree=True)
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
            sub_graph = sample_neighbors(one_graph, {self.category: cate_num},
                                         {network_schema[1]: self.sample_rate[src_type]}, replace=True)
            one = self.gat_layers[i](sub_graph, (h[src_type], h[self.category]))
            one = one.squeeze(1)
            intra_embeddings.append(one)
        z_sc = self.inter(intra_embeddings)
        return z_sc


class Contrast(nn.Module):
    def __init__(self, hidden_dim, tau, lam):
        r"""
        Description
        -----------
        This part is used to calculate the contrastive loss.

        Returns
        -------
        contra_loss : float
            The calculated loss

        """
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
        r"""
        Description
        -----------
        This part is used to calculate the cosine similarity of each pair of nodes from different views.

        """
        z1_norm = torch.norm(z1, dim=-1, keepdim=True)
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)
        dot_numerator = torch.mm(z1, z2.t())
        dot_denominator = torch.mm(z1_norm, z2_norm.t())
        sim_matrix = torch.exp(dot_numerator / dot_denominator / self.tau)
        return sim_matrix

    def forward(self, z_mp, z_sc, pos):
        r"""
        Description
        -----------
        This is the forward part of contrast part.

        We firstly project the embeddings under two views into the space where contrastive loss is calculated. Then, we calculate the contrastive loss with projected embeddings in a cross-view way.

        .. math::
           \mathcal{L}_i^{sc}=-\log\frac{\sum_{j\in\mathbb{P}_i}exp\left(sim\left(z_i^{sc}\_proj,z_j^{mp}\_proj\right)/\tau\right)}{\sum_{k\in\{\mathbb{P}_i\bigcup\mathbb{N}_i\}}exp\left(sim\left(z_i^{sc}\_proj,z_k^{mp}\_proj\right)/\tau\right)}

        where we show the contrastive loss :math:`\mathcal{L}_i^{sc}` under network schema view, and :math:`\mathbb{P}_i` and :math:`\mathbb{N}_i` are positives and negatives for node :math:`i`.

        In a similar way, we can get the contrastive loss :math:`\mathcal{L}_i^{mp}` under meta-path view. Finally, we utilize combination parameter :math:`\lambda` to add this two losses.

        Note
        -----------
        In implementation, each row of 'matrix_mp2sc' means the similarity with exponential between one node in meta-path view and all nodes in network schema view. Then, we conduct normalization for this row,
        and pick the results where the pair of nodes are positives. Finally, we sum these results for each row, and give a log to get the final loss.

        """
        z_proj_mp = self.proj(z_mp)
        z_proj_sc = self.proj(z_sc)
        matrix_mp2sc = self.sim(z_proj_mp, z_proj_sc)
        matrix_sc2mp = matrix_mp2sc.t()

        matrix_mp2sc = matrix_mp2sc / (torch.sum(matrix_mp2sc, dim=1).view(-1, 1) + 1e-8)
        lori_mp = -torch.log(matrix_mp2sc.mul(pos.to_dense()).sum(dim=-1)).mean()

        matrix_sc2mp = matrix_sc2mp / (torch.sum(matrix_sc2mp, dim=1).view(-1, 1) + 1e-8)
        lori_sc = -torch.log(matrix_sc2mp.mul(pos.to_dense()).sum(dim=-1)).mean()
        contra_loss = self.lam * lori_mp + (1 - self.lam) * lori_sc
        return contra_loss


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

