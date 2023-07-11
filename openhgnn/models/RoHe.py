import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl import function as fn
from dgl.utils import expand_as_pair
from dgl.nn.pytorch.softmax import edge_softmax
from dgl.nn.pytorch.utils import Identity

from . import BaseModel, register_model
from ..layers.macro_layer.SemanticConv import SemanticAttention
from ..layers.MetapathConv import MetapathConv
from ..utils.utils import extract_metapaths, get_ntypes_from_canonical_etypes

@register_model('RoHe')
class RoHe(BaseModel):
    r"""
    RoHe model:  ”Robust Heterogeneous Graph Neural Networks against Adversarial Attacks“ (AAAI2022)
    RoHe model shows an example of using HAN, called RoHe-HAN. Most of the settings remain consistent with HAN,
    with partial modifications made in forward function, specifically replacing a portion of GATConv with RoHeGATConv.
    HAN model shows an example of using dgl.metapath_reachable_graph on the original heterogeneous
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
                   dropout=args.dropout,
                   settings=args.settings)

    def __init__(self, ntype_meta_paths_dict, in_dim, hidden_dim, out_dim, num_heads, dropout, settings):
        super(RoHe, self).__init__()
        self.mod_dict = nn.ModuleDict()
        for ntype, meta_paths_dict in ntype_meta_paths_dict.items():
            self.mod_dict[ntype] = _HAN(meta_paths_dict, in_dim, hidden_dim, out_dim, num_heads, dropout, settings)

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

    def __init__(self, meta_paths_dict, in_dim, hidden_dim, out_dim, num_heads, dropout, settings):
        super(_HAN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(HANLayer(meta_paths_dict, in_dim, hidden_dim, num_heads[0], dropout, settings))
        for l in range(1, len(num_heads)):
            self.layers.append(HANLayer(meta_paths_dict, hidden_dim * num_heads[l - 1],
                                        hidden_dim, num_heads[l], dropout, settings))
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

    def __init__(self, meta_paths_dict, in_dim, out_dim, layer_num_heads, dropout, settings):
        super(HANLayer, self).__init__()
        self.meta_paths_dict = meta_paths_dict
        semantic_attention = SemanticAttention(in_size=out_dim * layer_num_heads)
        mods = nn.ModuleDict({mp: RoHeGATConv(in_dim, out_dim, layer_num_heads,
                                          dropout, dropout, activation=F.elu,
                                          settings=settings[i]) for i, mp in enumerate(meta_paths_dict.keys())})
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


class RoHeGATConv(nn.Module):
    r"""like Graph attention layer from `Graph Attention Network
        <https://arxiv.org/pdf/1710.10903.pdf>`, but modifying the computation of \alpha_{ij}.

        .. math::
            h_i^{(l+1)} = \sum_{j\in \mathcal{N}(i)} \alpha_{i,j} W^{(l)} h_j^{(l)}

        where :math:`\alpha_{ij}` is the attention score bewteen node :math:`i` and
        node :math:`j`:

        .. math::
            \alpha_{ij}^{l} &= \mathrm{softmax_i} (e_{ij}^{l} + m_{ij}^{l})

            e_{ij}^{l} &= \mathrm{LeakyReLU}\left(\vec{a}^T [W h_{i} \| W h_{j}]\right)

        Parameters
        ----------
        in_feats : int, or pair of ints
            Input feature size; i.e, the number of dimensions of :math:`h_i^{(l)}`.
            GATConv can be applied on homogeneous graph and unidirectional
            `bipartite graph <https://docs.dgl.ai/generated/dgl.bipartite.html?highlight=bipartite>`__.
            If the layer is to be applied to a unidirectional bipartite graph, ``in_feats``
            specifies the input feature size on both the source and destination nodes.  If
            a scalar is given, the source and destination node feature size would take the
            same value.
        out_feats : int
            Output feature size; i.e, the number of dimensions of :math:`h_i^{(l+1)}`.
        num_heads : int
            Number of heads in Multi-Head Attention.
        feat_drop : float, optional
            Dropout rate on feature. Defaults: ``0``.
        attn_drop : float, optional
            Dropout rate on attention weight. Defaults: ``0``.
        negative_slope : float, optional
            LeakyReLU angle of negative slope. Defaults: ``0.2``.
        residual : bool, optional
            If True, use residual connection. Defaults: ``False``.
        activation : callable activation function/layer or None, optional.
            If not None, applies an activation function to the updated node features.
            Default: ``None``.
    """
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None,
                 settings={'K': 10, 'P': 0.6, 'tau': 0.1, 'Flag': "None"}):

        super(RoHeGATConv, self).__init__()
        self._num_heads = num_heads
        self.settings = settings
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(0.0)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def mask(self, attM):
        T = self.settings['T']
        indices_to_remove = attM < torch.clamp(torch.topk(attM, T)[0][..., -1, None], min=0)
        attM[indices_to_remove] = -9e15
        return attM

    def forward(self, graph, feat):
        graph = graph.local_var()
        if isinstance(feat, tuple):
            h_src = self.feat_drop(feat[0])
            h_dst = self.feat_drop(feat[1])
            feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
            feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
        else:
            h_src = h_dst = self.feat_drop(feat)
            feat_src = feat_dst = self.fc(h_src).view(
                -1, self._num_heads, self._out_feats)
        N = graph.nodes().shape[0]
        N_e = graph.edges()[0].shape[0]
        graph.srcdata.update({'ft': feat_src})

        # introduce transiting prior
        e_trans = torch.FloatTensor(self.settings['TransM'].data).view(N_e, 1)
        e_trans = e_trans.repeat(1, 8).resize_(N_e, 8, 1)

        # feature-based similarity
        e = torch.cat([torch.matmul(feat_src[:, i, :].view(N, self._out_feats),
                                    feat_src[:, i, :].t().view(self._out_feats, N))[
                           graph.edges()[0], graph.edges()[1]].view(N_e, 1) \
                       for i in range(self._num_heads)], dim=1).view(N_e, 8, 1)

        total_edge = torch.cat((graph.edges()[0].view(1, N_e), graph.edges()[1].view(1, N_e)), 0)
        # confidence score in Eq(7)   change here device
        attn = torch.sparse.FloatTensor(total_edge.to(self.settings['device']),
                                        torch.squeeze(
                                            (e.to(self.settings['device']) * e_trans.to(self.settings['device'])).sum(
                                                -2)),
                                        torch.Size([N, N])).to(self.settings['device'])

        # purification mask in Eq(8)
        attn = self.mask(attn.to_dense()).t()
        e[attn[graph.edges()[0], graph.edges()[1]].view(N_e, 1).repeat(1, 8).view(N_e, 8, 1) < -100] = -9e15

        # obtain purified final attention in Eq(9)
        graph.edata['a'] = edge_softmax(graph, e)

        # message passing
        graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                         fn.sum('m', 'ft'))
        rst = graph.dstdata['ft']

        # residual
        if self.res_fc is not None:
            resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
            rst = rst + resval

        # activation
        if self.activation:
            rst = self.activation(rst)
        return rst