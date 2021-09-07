import dgl
import torch
import torch.nn as nn
from . import BaseModel, register_model
from dgl.nn.pytorch.conv import APPNPConv


@register_model('HPN')
class HPN(BaseModel):
    r"""
    Description
    ------------
    This model shows an example of using dgl.metapath_reachable_graph on the original heterogeneous
    graph.HPN from paper `Heterogeneous Graph Propagation Network
    <https://ieeexplore.ieee.org/abstract/document/9428609>`__.
    The author did not provide codes. So, we complete it according to the implementation of HAN


    .. math::
        \mathbf{Z}^{\Phi}=\mathcal{P}_{\Phi}(\mathbf{X})=g_\Phi(f_\Phi(\mathbf{X}))

    where :math:`\mathbf{X}` denotes initial feature matrix and :math:`\mathbf{Z^\Phi}` denotes semantic-specific node embedding.

    .. math::
        \mathbf{H}^{\Phi}=f_\Phi(\mathbf{X})=\sigma(\mathbf{X} \cdot \mathbf{W}^\Phi+\mathbf{b}^{\Phi})

    where :math:`\mathbf{H}^{\Phi}` is projected node feature matrix

    .. math::
        \mathbf{Z}^{\Phi, k}=g_{\Phi}\left(\mathbf{Z}^{\Phi, k-1}\right)=(1-\gamma) \cdot \mathbf{M}^{\Phi} \cdot \mathbf{Z}^{\Phi, k-1}+\gamma \cdot \mathbf{H}^{\Phi}

    where :math:`\mathbf{Z}^{\Phi,k}` denotes node embedding learned by k-th layer semantic propagation mechanism. :math:`\gamma` is a weight scalar which indicates the
    importance of characteristic of node in aggregating process



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
        propagation times
    alpha : float
        Value of restart probability
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

