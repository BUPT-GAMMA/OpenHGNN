import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GATConv
from . import BaseModel, register_model
from ..layers.macro_layer.SemanticConv import SemanticAttention
from ..layers.MetapathConv import MetapathConv
from ..utils.utils import extract_metapaths, get_ntypes_from_canonical_etypes


@register_model('HAN')
class HAN(BaseModel):
    r"""
    This model shows an example of using dgl.metapath_reachable_graph on the original heterogeneous
    graph HAN from paper `Heterogeneous Graph Attention Network <https://arxiv.org/pdf/1903.07293.pdf>`__..
    Because the original HAN implementation only gives the preprocessed homogeneous graph, this model
    could not reproduce the result in HAN as they did not provide the preprocessing code, and we
    constructed another dataset from ACM with a different set of papers, connections, features and
    labels.


    .. math::
        \mathbf{h}_{i}^{\prime}=\mathbf{M}_{\phi_{i}} \cdot \mathbf{h}_{i}

    where :math:`h_i` and :math:`h'_i` are the original and projected feature of node :math:`i`

    .. math::
        e_{i j}^{\Phi}=a t t_{\text {node }}\left(\mathbf{h}_{i}^{\prime}, \mathbf{h}_{j}^{\prime} ; \Phi\right)

    where :math:`{att}_{node}` denotes the deep neural network.

    .. math::
        \alpha_{i j}^{\Phi}=\operatorname{softmax}_{j}\left(e_{i j}^{\Phi}\right)=\frac{\exp \left(\sigma\left(\mathbf{a}_{\Phi}^{\mathrm{T}} \cdot\left[\mathbf{h}_{i}^{\prime} \| \mathbf{h}_{j}^{\prime}\right]\right)\right)}{\sum_{k \in \mathcal{N}_{i}^{\Phi}} \exp \left(\sigma\left(\mathbf{a}_{\Phi}^{\mathrm{T}} \cdot\left[\mathbf{h}_{i}^{\prime} \| \mathbf{h}_{k}^{\prime}\right]\right)\right)}

    where :math:`\sigma` denotes the activation function, || denotes the concatenate
    operation and :math:`a_{\Phi}` is the node-level attention vector for meta-path :math:`\Phi`.

    .. math::
        \mathbf{z}_{i}^{\Phi}=\prod_{k=1}^{K} \sigma\left(\sum_{j \in \mathcal{N}_{i}^{\Phi}} \alpha_{i j}^{\Phi} \cdot \mathbf{h}_{j}^{\prime}\right)

    where :math:`z^{\Phi}_i` is the learned embedding of node i for the meta-path :math:`\Phi`.
    Given the meta-path set {:math:`\Phi_0 ,\Phi_1,...,\Phi_P`},after feeding node features into node-level attentionwe can obtain P groups of
    semantic-specific node embeddings, denotes as {:math:`Z_0 ,Z_1,...,Z_P`}.
    We use MetapathConv to finish Node-level Attention and Semantic-level Attention.


    Parameters
    ------------
    ntype_meta_paths_dict : dict[str, dict[str, list[etype]]]
        Dict from node type to dict from meta path name to meta path. For node classification, there is only one node type.
        For link prediction, there can be multiple node types which are source and destination node types of target links.
    in_dim : int
        Input feature dimension.
    hidden_dim : int
        Hidden layer dimension.
    out_dim : int
        Output feature dimension.
    num_heads : list[int]
        Number of attention heads.
    dropout : float
        Dropout probability.
    """

    @classmethod
    def build_model_from_args(cls, args, hg):

        # build ntype_meta_paths_dict

        ntypes = set()
        if hasattr(args, 'target_link'):
            ntypes = get_ntypes_from_canonical_etypes(args.target_link)
        elif hasattr(args, 'category'):
            ntypes.add(args.category)
        else:
            raise ValueError

        ntype_meta_paths_dict = {}
        for ntype in ntypes:
            ntype_meta_paths_dict[ntype] = {}
            for meta_path_name, meta_path in args.meta_paths_dict.items():
                # a meta path starts with this node type
                if meta_path[0][0] == ntype:
                    ntype_meta_paths_dict[ntype][meta_path_name] = meta_path
        for ntype, meta_paths_dict in ntype_meta_paths_dict.items():
            if len(meta_paths_dict) == 0:
                ntype_meta_paths_dict[ntype] = extract_metapaths(ntype, hg.canonical_etypes)

        return cls(ntype_meta_paths_dict=ntype_meta_paths_dict,
                   in_dim=args.hidden_dim,
                   hidden_dim=args.hidden_dim,
                   out_dim=args.out_dim,
                   num_heads=args.num_heads,
                   dropout=args.dropout)

    def __init__(self, ntype_meta_paths_dict, in_dim, hidden_dim, out_dim, num_heads, dropout):
        super(HAN, self).__init__()
        self.out_dim = out_dim
        self.mod_dict = nn.ModuleDict()
        for ntype, meta_paths_dict in ntype_meta_paths_dict.items():
            self.mod_dict[ntype] = _HAN(meta_paths_dict, in_dim, hidden_dim, out_dim, num_heads, dropout)

    def forward(self, g, h_dict):
        r"""
        Parameters
        -----------
        g : DGLHeteroGraph or dict[str, dict[str, DGLBlock]]
            For full batch, it is a heterogeneous graph. For mini batch, it is a dict from node type to dict from
            mata path name to DGLBlock.
        h_dict : dict[str, Tensor] or dict[str, dict[str, dict[str, Tensor]]]
            The input features. For full batch, it is a dict from node type to node features. For mini batch, it is
            a dict from node type to dict from meta path name to dict from node type to node features.

        Returns
        --------
        out_dict : dict[str, Tensor]
            The output features. Dict from node type to node features.
        """
        out_dict = {}
        for ntype, han in self.mod_dict.items():
            if isinstance(g, dict):
                # mini batch
                if ntype not in g:
                    continue
                _g = g[ntype]
                _in_h = h_dict[ntype]
            else:
                # full batch
                _g = g
                _in_h = h_dict
            _out_h = han(_g, _in_h)
            for ntype, h in _out_h.items():
                out_dict[ntype] = h

        return out_dict


class _HAN(nn.Module):

    def __init__(self, meta_paths_dict, in_dim, hidden_dim, out_dim, num_heads, dropout):
        super(_HAN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(HANLayer(meta_paths_dict, in_dim, hidden_dim, num_heads[0], dropout))
        for l in range(1, len(num_heads)):
            self.layers.append(HANLayer(meta_paths_dict, hidden_dim * num_heads[l - 1],
                                        hidden_dim, num_heads[l], dropout))
        self.linear = nn.Linear(hidden_dim * num_heads[-1], out_dim)

    def forward(self, g, h_dict):
        for gnn in self.layers:
            h_dict = gnn(g, h_dict)
        out_dict = {}
        for ntype, h in h_dict.items():  # only one ntype here
            out_dict[ntype] = self.linear(h_dict[ntype])
        return out_dict

    def get_emb(self, g, h_dict):
        h = h_dict[self.category]
        for gnn in self.layers:
            h = gnn(g, h)

        return {self.category: h.detach().cpu().numpy()}


class HANLayer(nn.Module):
    """
    HAN layer.

    Parameters
    ------------
    meta_paths_dict : dict[str, list[etype]]
        Dict from meta path name to meta path.
    in_dim : int
        Input feature dimension.
    hidden_dim : int
        Hidden layer dimension.
    out_dim : int
        Output feature dimension.
    layer_num_heads : int
        Number of attention heads.
    dropout : float
        Dropout probability.

    Attributes
    ------------
    _cached_graph : dgl.DGLHeteroGraph
        a cached graph
    _cached_coalesced_graph : list
        _cached_coalesced_graph list generated by *dgl.metapath_reachable_graph()*
    """

    def __init__(self, meta_paths_dict, in_dim, out_dim, layer_num_heads, dropout):
        super(HANLayer, self).__init__()
        self.meta_paths_dict = meta_paths_dict
        semantic_attention = SemanticAttention(in_size=out_dim * layer_num_heads)
        mods = nn.ModuleDict({mp: GATConv(in_dim, out_dim, layer_num_heads,
                                          dropout, dropout, activation=F.elu,
                                          allow_zero_in_degree=True) for mp in meta_paths_dict})
        self.model = MetapathConv(meta_paths_dict, mods, semantic_attention)
        self._cached_graph = None
        self._cached_coalesced_graph = {}

    def forward(self, g, h):
        r"""
        Parameters
        -----------
        g : DGLHeteroGraph or dict[str, DGLBlock]
            For full batch, it is a heterogeneous graph. For mini batch, it is a dict from mata path name to DGLBlock.
        h : dict[str, Tensor] or dict[str, dict[str, Tensor]]
            The input features. For full batch, it is a dict from node type to node features. For mini batch, it is
            a  dict from meta path name to dict from node type to node features.

        Returns
        --------
        h : dict[str, Tensor]
            The output features. Dict from node type to node features. Only one node type.
        """
        # mini batch
        if isinstance(g, dict):
            h = self.model(g, h)

        # full batch
        else:
            if self._cached_graph is None or self._cached_graph is not g:
                self._cached_graph = g
                self._cached_coalesced_graph.clear()
                for mp, mp_value in self.meta_paths_dict.items():
                    self._cached_coalesced_graph[mp] = dgl.metapath_reachable_graph(
                        g, mp_value)
            h = self.model(self._cached_coalesced_graph, h)

        return h
