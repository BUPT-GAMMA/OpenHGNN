import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn as dglnn
from . import BaseModel, register_model


@register_model('RGAT')
class RGAT(BaseModel):
    @classmethod
    def build_model_from_args(cls, args, hg):
        return cls(in_dim=args.hidden_dim,
                   out_dim=args.hidden_dim,
                   h_dim=args.out_dim,
                   etypes=hg.etypes,
                   num_heads=args.num_heads,
                   dropout=args.dropout)

    def __init__(self, in_dim, out_dim, h_dim, etypes, num_heads, dropout):
        super(RGAT, self).__init__()
        self.rel_names = list(set(etypes))
        self.layers = nn.ModuleList()
        # input 2 hidden
        self.layers.append(RGATLayer(
            in_dim, h_dim, num_heads, self.rel_names, activation=F.relu, dropout=dropout))
        self.layers.append(RGATLayer(
            h_dim, out_dim, num_heads, self.rel_names, activation=None))
        return

    def forward(self, hg, h_dict=None):
        if hasattr(hg, 'ntypes'):
            # full graph training,
            for layer in self.layers:
                h_dict = layer(hg, h_dict)
        else:
            # minibatch training, block
            for layer, block in zip(self.layers, hg):
                h_dict = layer(block, h_dict)
        return h_dict


class RGATLayer(nn.Module):

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
        super(RGATLayer, self).__init__()
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