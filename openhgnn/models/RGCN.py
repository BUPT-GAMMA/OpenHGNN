import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn as dglnn
from . import BaseModel, register_model


@register_model('RGCN')
class RGCN(BaseModel):
    """
    **Title:** `Modeling Relational Data with Graph Convolutional Networks <https://arxiv.org/abs/1703.06103>`_

    **Authors:** Michael Schlichtkrull, Thomas N. Kipf, Peter Bloem, Rianne van den Berg, Ivan Titov, Max Welling

    Parameters
    ----------
    in_dim : int
        Input feature size.
    hidden_dim : int
        Hidden dimension .
    out_dim : int
        Output feature size.
    etypes : list[str]
        Relation names.
    num_bases : int, optional
        Number of bases. If is none, use number of relations. Default: None.
    num_hidden_layers: int
        Number of RelGraphConvLayer
    dropout : float, optional
        Dropout rate. Default: 0.0
    use_self_loop : bool, optional
        True to include self loop message. Default: False

    Attributes
    -----------
    RelGraphConvLayer: RelGraphConvLayer

    """
    @classmethod
    def build_model_from_args(cls, args, hg):
        return cls(args.hidden_dim,
                   args.hidden_dim,
                   args.out_dim,
                   hg.etypes,
                   args.n_bases,
                   args.num_layers - 2,
                   dropout=args.dropout)

    def __init__(self, in_dim,
                 hidden_dim,
                 out_dim,
                 etypes,
                 num_bases,
                 num_hidden_layers=1,
                 dropout=0,
                 use_self_loop=False):
        super(RGCN, self).__init__()
        self.in_dim = in_dim
        self.h_dim = hidden_dim
        self.out_dim = out_dim
        self.rel_names = list(set(etypes))
        self.rel_names.sort()
        if num_bases < 0 or num_bases > len(self.rel_names):
            self.num_bases = len(self.rel_names)
        else:
            self.num_bases = num_bases
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.use_self_loop = use_self_loop

        self.layers = nn.ModuleList()
        # input 2 hidden
        self.layers.append(RelGraphConvLayer(
            self.in_dim, self.h_dim, self.rel_names,
            self.num_bases, activation=F.relu, self_loop=self.use_self_loop,
            dropout=self.dropout, weight=True))
        # hidden 2 hidden
        for i in range(self.num_hidden_layers):
            self.layers.append(RelGraphConvLayer(
                self.h_dim, self.h_dim, self.rel_names,
                self.num_bases, activation=F.relu, self_loop=self.use_self_loop,
                dropout=self.dropout))
        # hidden 2 output
        self.layers.append(RelGraphConvLayer(
            self.h_dim, self.out_dim, self.rel_names,
            self.num_bases, activation=None,
            self_loop=self.use_self_loop))

    def forward(self, hg, h_dict):
        r"""
        Support full-batch and mini-batch training.

        Parameters
        ----------
        hg: dgl.HeteroGraph or dgl.blocks
            Input graph
        h_dict: dict[str, th.Tensor]
            Input feature
        Returns
        -------
        h: dict[str, th.Tensor]
            output feature
        """
        if hasattr(hg, 'ntypes'):
            # full graph training,
            for layer in self.layers:
                h_dict = layer(hg, h_dict)
        else:
            # minibatch training, block
            for layer, block in zip(self.layers, hg):
                h_dict = layer(block, h_dict)
        return h_dict

    def l2_penalty(self):
        loss = 0.0005 * th.norm(self.layers[0].weight, p=2, dim=1)
        return loss


class RelGraphConvLayer(nn.Module):
    r"""Relational graph convolution layer.

    We use `HeteroGraphConv <https://docs.dgl.ai/api/python/nn.pytorch.html#heterographconv>`_ to implement the model.

    Parameters
    ----------
    in_feat : int
        Input feature size.
    out_feat : int
        Output feature size.
    rel_names : list[str]
        Relation names.
    num_bases : int, optional
        Number of bases. If is none, use number of relations. Default: None.
    weight : bool, optional
        True if a linear layer is applied after message passing. Default: True
    bias : bool, optional
        True if bias is added. Default: True
    activation : callable, optional
        Activation function. Default: None
    self_loop : bool, optional
        True to include self loop message. Default: False
    dropout : float, optional
        Dropout rate. Default: 0.0
    """

    def __init__(self,
                 in_feat,
                 out_feat,
                 rel_names,
                 num_bases,
                 *,
                 weight=True,
                 bias=True,
                 activation=None,
                 self_loop=False,
                 dropout=0.0):
        super(RelGraphConvLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.rel_names = rel_names
        self.num_bases = num_bases
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop
        self.batchnorm = False

        self.conv = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feat, out_feat, norm='right', weight=False, bias=False)
            for rel in rel_names
        })

        self.use_weight = weight
        self.use_basis = num_bases < len(self.rel_names) and weight
        if self.use_weight:
            if self.use_basis:
                self.basis = dglnn.WeightBasis((in_feat, out_feat), num_bases, len(self.rel_names))
            else:
                self.weight = nn.Parameter(th.Tensor(len(self.rel_names), in_feat, out_feat))
                nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

        # bias
        if bias:
            self.h_bias = nn.Parameter(th.Tensor(out_feat))
            nn.init.zeros_(self.h_bias)

        # weight for self loop
        if self.self_loop:
            self.loop_weight = nn.Parameter(th.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight,
                                    gain=nn.init.calculate_gain('relu'))
        # define batch norm layer
        if self.batchnorm:
            self.bn = nn.BatchNorm1d(out_feat)

        self.dropout = nn.Dropout(dropout)

    def forward(self, g, inputs):
        """Forward computation
        Parameters
        ----------
        g : DGLHeteroGraph
            Input graph.
        inputs : dict[str, torch.Tensor]
            Node feature for each node type.
        Returns
        -------
        dict[str, torch.Tensor]
            New node features for each node type.
        """
        g = g.local_var()
        if self.use_weight:
            weight = self.basis() if self.use_basis else self.weight
            wdict = {self.rel_names[i]: {'weight': w.squeeze(0)}
                     for i, w in enumerate(th.split(weight, 1, dim=0))}
        else:
            wdict = {}

        if g.is_block:
            inputs_src = inputs
            inputs_dst = {k: v[:g.number_of_dst_nodes(k)] for k, v in inputs.items()}
        else:
            inputs_src = inputs_dst = inputs

        hs = self.conv(g, inputs_src, mod_kwargs=wdict)

        def _apply(ntype, h):
            if self.self_loop:
                h = h + th.matmul(inputs_dst[ntype], self.loop_weight)
            if self.bias:
                h = h + self.h_bias
            if self.activation:
                h = self.activation(h)
            if self.batchnorm:
                h = self.bn(h)
            return self.dropout(h)

        return {ntype: _apply(ntype, h) for ntype, h in hs.items()}