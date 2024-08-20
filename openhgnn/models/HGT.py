import math
import dgl
import torch
import torch.nn as nn

from dgl import function as fn
from dgl.nn.pytorch.linear import TypedLinear
from dgl.nn.pytorch.softmax import edge_softmax
from . import BaseModel, register_model
from ..utils import to_hetero_feat


@register_model('HGT')
class HGT(BaseModel):
    r"""Heterogeneous graph transformer convolution from `Heterogeneous Graph Transformer
    <https://arxiv.org/abs/2003.01332>`__

    For more details, you may refer to `HGT<https://docs.dgl.ai/en/0.8.x/generated/dgl.nn.pytorch.conv.HGTConv.html>`__
    
    Parameters
    ----------
    in_dim: int
        the input dimension
    out_dim: int
        the output dimension
    num_heads: list
        the list of the number of heads in each layer
    num_etypes: int
        the number of the edge type
    num_ntypes: int
        the number of the node type
    num_layers: int
        the number of layers we used in the computing
    dropout: float
        the feature drop rate
    norm: boolean
        if we need the norm operation
    ntypes: list
        the list of node type
    """

    @classmethod
    def build_model_from_args(cls, args, hg):

        return cls(args.hidden_dim,
                   args.out_dim,
                   args.num_heads,
                   len(hg.etypes),
                   hg.ntypes,
                   args.num_layers,
                   args.dropout,
                   args.norm
                   )

    def __init__(self, in_dim, out_dim, num_heads, num_etypes, ntypes,
                 num_layers, dropout=0.2, norm=False):
        super(HGT, self).__init__()
        self.num_layers = num_layers
        self.ntypes = ntypes
        self.hgt_layers = nn.ModuleList()
        self.hgt_layers.append(
            HGTConv(in_dim,
                    in_dim // num_heads,
                    num_heads,
                    len(ntypes),
                    num_etypes,
                    dropout,
                    norm)
        )

        for _ in range(1, num_layers - 1):
            self.hgt_layers.append(
                HGTConv(in_dim,
                        in_dim // num_heads,
                        num_heads,
                        len(ntypes),
                        num_etypes,
                        dropout,
                        norm)
            )

        self.hgt_layers.append(
            HGTConv(in_dim,
                    out_dim,
                    1,
                    len(ntypes),
                    num_etypes,
                    dropout,
                    norm)
        )

    def forward(self, hg, h_dict):
        """
        The forward part of the HGT.
        
        Parameters
        ----------
        hg : object
            the dgl heterogeneous graph
        h_dict: dict
            the feature dict of different node types
            
        Returns
        -------
        dict
            The embeddings after the output projection.
        """

        if hasattr(hg, 'ntypes'):
            # full graph training,
            with hg.local_scope():
                hg.ndata['h'] = h_dict
                g = dgl.to_homogeneous(hg, ndata='h')
                h = g.ndata['h']
                for l in range(self.num_layers):
                    h = self.hgt_layers[l](g, h, g.ndata['_TYPE'], g.edata['_TYPE'], presorted=True)
            h_dict = to_hetero_feat(h, g.ndata['_TYPE'], self.ntypes)

        else:
            # for minibatch training, input h_dict is a tensor
            h = h_dict
            for layer, block in zip(self.hgt_layers, hg):
                h = layer(block, h, block.ndata['_TYPE']['_N'], block.edata['_TYPE'], presorted=False)
            h_dict = to_hetero_feat(h, block.ndata['_TYPE']['_N'][:block.num_dst_nodes()], self.ntypes)

        return h_dict

    @property
    def to_homo_flag(self):
        return True

class HGTConv(nn.Module):
    r"""Heterogeneous graph transformer convolution from `Heterogeneous Graph Transformer
    <https://arxiv.org/abs/2003.01332>`__

    Given a graph :math:`G(V, E)` and input node features :math:`H^{(l-1)}`,
    it computes the new node features as follows:

    Compute a multi-head attention score for each edge :math:`(s, e, t)` in the graph:

    .. math::

      Attention(s, e, t) = \text{Softmax}\left(||_{i\in[1,h]}ATT-head^i(s, e, t)\right) \\
      ATT-head^i(s, e, t) = \left(K^i(s)W^{ATT}_{\phi(e)}Q^i(t)^{\top}\right)\cdot
        \frac{\mu_{(\tau(s),\phi(e),\tau(t)}}{\sqrt{d}} \\
      K^i(s) = \text{K-Linear}^i_{\tau(s)}(H^{(l-1)}[s]) \\
      Q^i(t) = \text{Q-Linear}^i_{\tau(t)}(H^{(l-1)}[t]) \\

    Compute the message to send on each edge :math:`(s, e, t)`:

    .. math::

      Message(s, e, t) = ||_{i\in[1, h]} MSG-head^i(s, e, t) \\
      MSG-head^i(s, e, t) = \text{M-Linear}^i_{\tau(s)}(H^{(l-1)}[s])W^{MSG}_{\phi(e)} \\

    Send messages to target nodes :math:`t` and aggregate:

    .. math::

      \tilde{H}^{(l)}[t] = \sum_{\forall s\in \mathcal{N}(t)}\left( Attention(s,e,t)
      \cdot Message(s,e,t)\right)

    Compute new node features:

    .. math::

      H^{(l)}[t]=\text{A-Linear}_{\tau(t)}(\sigma(\tilde(H)^{(l)}[t])) + H^{(l-1)}[t]

    Parameters
    ----------
    in_size : int
        Input node feature size.
    head_size : int
        Output head size. The output node feature size is ``head_size * num_heads``.
    num_heads : int
        Number of heads. The output node feature size is ``head_size * num_heads``.
    num_ntypes : int
        Number of node types.
    num_etypes : int
        Number of edge types.
    dropout : optional, float
        Dropout rate.
    use_norm : optiona, bool
        If true, apply a layer norm on the output node feature.

    Examples
    --------
    """
    def __init__(self,
                 in_size,
                 head_size,
                 num_heads,
                 num_ntypes,
                 num_etypes,
                 dropout=0.2,
                 use_norm=False):
        super().__init__()
        self.in_size = in_size
        self.head_size = head_size
        self.num_heads = num_heads
        self.sqrt_d = math.sqrt(head_size)
        self.use_norm = use_norm

        self.linear_k = TypedLinear(in_size, head_size * num_heads, num_ntypes)
        self.linear_q = TypedLinear(in_size, head_size * num_heads, num_ntypes)
        self.linear_v = TypedLinear(in_size, head_size * num_heads, num_ntypes)
        self.linear_a = TypedLinear(head_size * num_heads, head_size * num_heads, num_ntypes)

        self.relation_pri = nn.ParameterList([nn.Parameter(torch.ones(num_etypes))
                                              for i in range(num_heads)])
        self.relation_att = nn.ModuleList([TypedLinear(head_size, head_size, num_etypes)
                                           for i in range(num_heads)])
        self.relation_msg = nn.ModuleList([TypedLinear(head_size, head_size, num_etypes)
                                           for i in range(num_heads)])
        self.skip = nn.Parameter(torch.ones(num_ntypes))
        self.drop = nn.Dropout(dropout)
        if use_norm:
            self.norm = nn.LayerNorm(head_size * num_heads)
        if in_size != head_size * num_heads:
            self.residual_w = nn.Parameter(torch.Tensor(in_size, head_size * num_heads))
            nn.init.xavier_uniform_(self.residual_w)

    def forward(self, g, x, ntype, etype, *, presorted=False):
        """Forward computation.

        Parameters
        ----------
        g : DGLGraph
            The input graph.
        x : torch.Tensor
            A 2D tensor of node features. Shape: :math:`(|V|, D_{in})`.
        ntype : torch.Tensor
            An 1D integer tensor of node types. Shape: :math:`(|V|,)`.
        etype : torch.Tensor
            An 1D integer tensor of edge types. Shape: :math:`(|E|,)`.
        presorted : bool, optional
            Whether *both* the nodes and the edges of the input graph have been sorted by
            their types. Forward on pre-sorted graph may be faster. Graphs created by
            :func:`~dgl.to_homogeneous` automatically satisfy the condition.
            Also see :func:`~dgl.reorder_graph` for manually reordering the nodes and edges.

        Returns
        -------
        torch.Tensor
            New node features. Shape: :math:`(|V|, D_{head} * N_{head})`.
        """
        self.presorted = presorted
        if g.is_block:
            x_src = x
            x_dst = x[:g.num_dst_nodes()]
            srcntype = ntype
            dstntype = ntype[:g.num_dst_nodes()]
        else:
            x_src = x
            x_dst = x
            srcntype = ntype
            dstntype = ntype
        with g.local_scope():
            k = self.linear_k(x_src, srcntype, presorted).view(-1, self.num_heads, self.head_size)
            q = self.linear_q(x_dst, dstntype, presorted).view(-1, self.num_heads, self.head_size)
            v = self.linear_v(x_src, srcntype, presorted).view(-1, self.num_heads, self.head_size)
            g.srcdata['k'] = k
            g.dstdata['q'] = q
            g.srcdata['v'] = v
            g.edata['etype'] = etype
            g.apply_edges(self.message)
            g.edata['m'] = g.edata['m'] * edge_softmax(g, g.edata['a']).unsqueeze(-1)
            g.update_all(fn.copy_e('m', 'm'), fn.sum('m', 'h'))
            h = g.dstdata['h'].view(-1, self.num_heads * self.head_size)
            # target-specific aggregation
            h = self.drop(self.linear_a(h, dstntype, presorted))
            alpha = torch.sigmoid(self.skip[dstntype]).unsqueeze(-1)
            if x_dst.shape != h.shape:
                h = h * alpha + (x_dst @ self.residual_w) * (1 - alpha)
            else:
                h = h * alpha + x_dst * (1 - alpha)
            if self.use_norm:
                h = self.norm(h)
            return h

    def message(self, edges):
        """Message function."""
        a, m = [], []
        etype = edges.data['etype']
        k = torch.unbind(edges.src['k'], dim=1)
        q = torch.unbind(edges.dst['q'], dim=1)
        v = torch.unbind(edges.src['v'], dim=1)
        for i in range(self.num_heads):
            kw = self.relation_att[i](k[i], etype, self.presorted)  # (E, O)
            a.append((kw * q[i]).sum(-1) * self.relation_pri[i][etype] / self.sqrt_d)   # (E,)
            m.append(self.relation_msg[i](v[i], etype, self.presorted))  # (E, O)
        return {'a' : torch.stack(a, dim=1), 'm' : torch.stack(m, dim=1)}
