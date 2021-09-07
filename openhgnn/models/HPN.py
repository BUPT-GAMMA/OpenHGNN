"""
This model shows an example of using dgl.metapath_reachable_graph on the original heterogeneous
graph.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import dgl
from scipy import sparse as sp
from . import BaseModel, register_model
from dgl.nn.pytorch.conv import APPNPConv


@register_model('HPN')
class HPN(BaseModel):
    r"""
    Description
    ------------
    This model shows an example of using dgl.metapath_reachable_graph on the original heterogeneous
    graph.HPN from paper 'Heterogeneous Graph Propagation Network <https://ieeexplore.ieee.org/abstract/document/9428609>'
    The author did not provide codes. So, we complete it according to the implementation of HAN
    .. math::
        \bold Z^\Phi=\mathcal{P}_\Phi(\bold X)=g_\Phi(f_\Phi(\bold X))

        \bold H^\Phi=f_\Phi(\bold X)=\sigma(\bold X · \bold W^\Phi+\bold b^\Phi)

        \mathbf{Z}^{\Phi, k}=g_{\Phi}\left(\mathbf{Z}^{\Phi, k-1}\right)=(1-\gamma) \cdot \mathbf{M}^{\Phi} \cdot \mathbf{Z}^{\Phi, k-1}+\gamma \cdot \mathbf{H}^{\Phi}

        w_{\Phi_{p}}=\frac{1}{|\mathcal{V}|} \sum_{i \in \mathcal{V}} \mathbf{q}^{\mathrm{T}} \cdot \tanh \left(\mathbf{W} \cdot \mathbf{z}_{i}^{\Phi_{p}}+\mathbf{b}\right)

        \mathbf{Z}=\sum_{p=1}^{P} \beta_{\Phi_{p}} \cdot \mathbf{Z}^{\Phi_{p}}


    Parameters
    ------------
    meta_paths : list
        contain multiple meta-paths.
    category : str
        The category means the head and tail node of metapaths.
    in_size : int
        input feature dimension.
    out_size : int
        out dimension.
    dropout : float
        Dropout probability.
    out_embedsizes : int
        Dimension of the final embedding Z
    k_layer : int
        propagation times :math:'K'.
    alpha : float
        Value of restart probability :math:'\alpha'.
    edge_drop : float, optional
        The dropout rate on edges that controls the
        messages received by each node. Default: ``0``.


    """
    @classmethod
    def build_model_from_args(cls, args, hg):
        etypes = hg.canonical_etypes
        mps = []
        for etype in etypes:
            if etype[0] == args.category:
                for dst_e in etypes:
                    if etype[0] == dst_e[2] and etype[2] == dst_e[0]:
                        mps.append([etype, dst_e])

        return cls(meta_paths=mps, category=args.category,
                    in_size=args.hidden_dim,
                    out_size=args.out_dim,
                    dropout=args.dropout,
                    out_embedsize=args.out_embedsize,
                    k_layer=args.k_layer,
                    alpha=args.alpha,
                    edge_drop=args.edge_drop
                   )

    def __init__(self, meta_paths, category, in_size,  out_size,  dropout, out_embedsize, k_layer, alpha, edge_drop):
        super(HPN, self).__init__()
        self.category = category
        self.layers = nn.ModuleList()
        self.layers.append(HPNLayer(meta_paths, in_size,  dropout, k_layer, alpha, edge_drop, out_embedsize))
        self.linear = nn.Linear(out_embedsize, out_size)


    def forward(self, g, h_dict):

        h = h_dict[self.category]
        for gnn in self.layers:
            h = gnn(g, h)

        return {self.category: self.linear(h)}

class SemanticFusion(nn.Module):
    def __init__(self, in_size=64, hidden_size=128):
        super(SemanticFusion, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z).mean(0)
        beta = torch.softmax(w, dim=0)
        beta = beta.expand((z.shape[0],) + beta.shape)

        return (beta * z).sum(1)


class HPNLayer(nn.Module):
    """
    HPN layer.
    Arguments
    ---------
    meta_paths : list of metapaths, each as a list of edge types
    in_size : input feature dimension
    dropout : Dropout probability
    k_layer : propagation times
    alpha   : Value of restart probability
    edge_drop : the dropout rate on edges that controls the messages received by each node
    out_embedsize : Dimension of the final embedding Z

    Inputs
    ------
    g : DGLHeteroGraph
        The heterogeneous graph
    h : tensor
        Input features
    Outputs
    -------
    tensor
        The output feature
    """
    def __init__(self, meta_paths, in_size, dropout,k_layer, alpha, edge_drop, out_embedsize):
        super(HPNLayer, self).__init__()

        # semantic projection function fΦ projects node into semantic space

        self.hidden = nn.Sequential(
            #nn.Linear(in_features=in_size, out_features=out_embedsize, bias=True),
            nn.ReLU()
        )


        # One Propagation layer for each meta path
        self.propagation_layers = nn.ModuleList()
        for i in range(len(meta_paths)):
            self.propagation_layers.append(APPNPConv(k_layer, alpha, edge_drop))
        self.semantic_fusion = SemanticFusion()
        self.meta_paths = list(tuple(meta_path) for meta_path in meta_paths)
        self._cached_graph = None
        self._cached_coalesced_graph = {}

    def forward(self, g, h):
        r"""
        Parameters
        -----------
        g : DGLHeteroGraph
            The heterogeneous graph
        h : tensor
            The input features

        Returns
        --------
        h : tensor
            The output features
        """
        semantic_embeddings = []
        h = self.hidden(h)
        if self._cached_graph is None or self._cached_graph is not g:
            self._cached_graph = g
            self._cached_coalesced_graph.clear()
            for meta_path in self.meta_paths:
                self._cached_coalesced_graph[meta_path] = dgl.metapath_reachable_graph(
                        g, meta_path)

        for i, meta_path in enumerate(self.meta_paths):

            new_g = self._cached_coalesced_graph[meta_path]
            semantic_embeddings.append(self.propagation_layers[i](new_g, h).flatten(1))
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)

        return self.semantic_fusion(semantic_embeddings)

