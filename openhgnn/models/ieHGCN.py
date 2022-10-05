from audioop import bias
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn.pytorch as dglnn

from . import BaseModel, register_model


@register_model('ieHGCN')
class ieHGCN(BaseModel):
    r"""
    ie-HGCN from paper `Interpretable and Efficient Heterogeneous Graph Convolutional Network
    <https://arxiv.org/pdf/2005.13183.pdf>`__.

    `Source Code Link <https://github.com/kepsail/ie-HGCN>`_
    
    The core part of ie-HGCN, the calculating flow of projection, object-level aggregation and type-level aggregation in
    a specific type block.

    Projection
    
    .. math::
        Y^{Self-\Omega }=H^{\Omega} \cdot W^{Self-\Omega} \quad (1)-1

        Y^{\Gamma - \Omega}=H^{\Gamma} \cdot W^{\Gamma - \Omega} , \Gamma \in N_{\Omega} \quad (1)-2

    Object-level Aggregation
    
    .. math::
        Z^{ Self - \Omega } = Y^{ Self - \Omega}=H^{\Omega} \cdot W^{Self - \Omega} \quad (2)-1

        Z^{\Gamma - \Omega}=\hat{A}^{\Omega-\Gamma} \cdot Y^{\Gamma - \Omega} = \hat{A}^{\Omega-\Gamma} \cdot H^{\Gamma} \cdot W^{\Gamma - \Omega} \quad (2)-2

    Type-level Aggregation
    
    .. math::
        Q^{\Omega}=Z^{Self-\Omega} \cdot W_q^{\Omega} \quad (3)-1

        K^{Self-\Omega}=Z^{Self -\Omega} \cdot W_{k}^{\Omega} \quad (3)-2

        K^{\Gamma - \Omega}=Z^{\Gamma - \Omega} \cdot W_{k}^{\Omega}, \quad \Gamma \in N_{\Omega} \quad (3)-3

    .. math::
        e^{Self-\Omega}={ELU} ([K^{ Self-\Omega} \| Q^{\Omega}] \cdot w_{a}^{\Omega}) \quad (4)-1

        e^{\Gamma - \Omega}={ELU} ([K^{\Gamma - \Omega} \| Q^{\Omega}] \cdot w_{a}^{\Omega}), \Gamma \in N_{\Omega} \quad (4)-2

    .. math::
        [a^{Self-\Omega}\|a^{1 - \Omega}\| \ldots . a^{\Gamma - \Omega}\|\ldots\| a^{|N_{\Omega}| - \Omega}] \\
        = {softmax}([e^{Self - \Omega}\|e^{1 - \Omega}\| \ldots\|e^{\Gamma - \Omega}\| \ldots \| e^{|N_{\Omega}| - \Omega}]) \quad (5)

    .. math::
        H_{i,:}^{\Omega \prime}=\sigma(a_{i}^{Self-\Omega} \cdot Z_{i,:}^{Self-\Omega}+\sum_{\Gamma \in N_{\Omega}} a_{i}^{\Gamma - \Omega} \cdot Z_{i,:}^{\Gamma - \Omega}) \quad (6)
    
    Parameters
    ----------
    num_layers: int
        the number of layers
    in_dim: int
        the input dimension
    hidden_dim: int
        the hidden dimension
    out_dim: int
        the output dimension
    attn_dim: int
        the dimension of attention vector
    ntypes: list
        the node type of a heterogeneous graph
    etypes: list
        the edge type of a heterogeneous graph
    bias: boolean
        whether we need bias vector
    batchnorm: boolean
        whether we need batchnorm
    dropout: float
        the drop out rate
    """
    @classmethod
    def build_model_from_args(cls, args, hg:dgl.DGLGraph):
        return cls(args.num_layers,
                   args.hidden_dim,
                   args.out_dim,
                   args.attn_dim,
                   hg.ntypes,
                   hg.etypes,
                   args.bias,
                   args.batchnorm,
                   args.dropout
                   )

    def __init__(self, num_layers, hidden_dim, out_dim, attn_dim, ntypes, etypes, bias, batchnorm, dropout):
        super(ieHGCN, self).__init__()
        self.num_layers = num_layers
        self.activation = F.elu
        self.hgcn_layers = nn.ModuleList()

        for i in range(0, num_layers - 1):
            self.hgcn_layers.append(
                ieHGCNConv(
                    hidden_dim,
                    hidden_dim,
                    attn_dim,
                    ntypes,
                    etypes,
                    self.activation,
                    bias,
                    batchnorm,
                    dropout
                )
            )
        
        self.hgcn_layers.append(
            ieHGCNConv(
                hidden_dim,
                out_dim,
                attn_dim,
                ntypes,
                etypes,
                None,
                False,
                False,
                0.0
            )
        )

    def forward(self, hg, h_dict):
        """
        The forward part of the ieHGCN.
        
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
        if hasattr(hg, "ntypes"):
            for l in range(self.num_layers):
                h_dict = self.hgcn_layers[l](hg, h_dict)
        else:
            for layer, block in zip(self.hgcn_layers, hg):
                h_dict = layer(block, h_dict)
        return h_dict

class ieHGCNConv(nn.Module):
    r"""
    The ieHGCN convolution layer.

    Parameters
    ----------
    in_size: int
        the input dimension
    out_size: int
        the output dimension
    attn_size: int
        the dimension of attention vector
    ntypes: list
        the node type list of a heterogeneous graph
    etypes: list
        the edge type list of a heterogeneous graph
    activation: str
        the activation function
    bias: boolean
        whether we need bias vector
    batchnorm: boolean
        whether we need batchnorm
    dropout: float
        the drop out rate
    """
    def __init__(self, in_size, out_size, attn_size, ntypes, etypes, activation = F.elu, 
                 bias = False, batchnorm = False, dropout = 0.0):
        super(ieHGCNConv, self).__init__()
        self.bias = bias
        self.batchnorm = batchnorm
        self.dropout = dropout
        node_size = {}
        for ntype in ntypes:
            node_size[ntype] = in_size
        attn_vector = {}
        for ntype in ntypes:
            attn_vector[ntype] =  attn_size
        self.W_self = dglnn.HeteroLinear(node_size, out_size)
        self.W_al = dglnn.HeteroLinear(attn_vector, 1)
        self.W_ar = dglnn.HeteroLinear(attn_vector, 1)
        
        # self.conv = dglnn.HeteroGraphConv({
        #     etype: dglnn.GraphConv(in_size, out_size, norm = 'right', weight = True, bias = True)
        #     for etype in etypes
        # })
        self.in_size = in_size
        self.out_size = out_size
        self.attn_size = attn_size
        mods = {
            etype: dglnn.GraphConv(in_size, out_size, norm = 'right', 
                                   weight = True, bias = True, allow_zero_in_degree = True)
            for etype in etypes
            }
        self.mods = nn.ModuleDict(mods)
        
        self.linear_q = nn.ModuleDict({ntype: nn.Linear(out_size, attn_size) for ntype in ntypes})
        self.linear_k = nn.ModuleDict({ntype: nn.Linear(out_size, attn_size) for ntype in ntypes})
        
        self.activation = activation
        if batchnorm:
            self.bn = nn.BatchNorm1d(out_size)
        if bias:
            self.h_bias = nn.Parameter(torch.Tensor(out_size))
            nn.init.zeros_(self.h_bias)      
        self.dropout = nn.Dropout(dropout)

    def forward(self, hg, h_dict):
        """
        The forward part of the ieHGCNConv.
        
        Parameters
        ----------
        hg : object or list[block]
            the dgl heterogeneous graph or the list of blocks
        h_dict: dict
            the feature dict of different node types
            
        Returns
        -------
        dict
            The embeddings after final aggregation.
        """
        outputs = {ntype: [] for ntype in hg.dsttypes}
        if hg.is_block:
            src_inputs = h_dict
            dst_inputs = {k: v[:hg.number_of_dst_nodes(k)] for k, v in h_dict.items()}
        else:
            src_inputs = h_dict
            dst_inputs = h_dict
        with hg.local_scope():
            hg.ndata['h'] = h_dict
            # formulas (2)-1
            dst_inputs = self.W_self(dst_inputs)
            query = {}
            key = {}
            attn = {}
            attention = {}
            
            # formulas (3)-1 and (3)-2
            for ntype in hg.dsttypes:
                query[ntype] = self.linear_q[ntype](dst_inputs[ntype])
                key[ntype] = self.linear_k[ntype](dst_inputs[ntype])
            # formulas (4)-1
            h_l = self.W_al(key)
            h_r = self.W_ar(query)
            for ntype in hg.dsttypes:
                attention[ntype] = F.elu(h_l[ntype] + h_r[ntype])
                attention[ntype] = attention[ntype].unsqueeze(0)
            
            for srctype, etype, dsttype in hg.canonical_etypes:
                rel_graph = hg[srctype, etype, dsttype]
                if srctype not in h_dict:
                    continue
                # formulas (2)-2
                dstdata = self.mods[etype](
                    rel_graph,
                    (src_inputs[srctype], dst_inputs[dsttype])
                )
                outputs[dsttype].append(dstdata)
                # formulas (3)-3
                attn[dsttype] = self.linear_k[dsttype](dstdata)
                # formulas (4)-2
                h_attn = self.W_al(attn)
                attn.clear()
                edge_attention = F.elu(h_attn[dsttype] + h_r[dsttype])
                attention[dsttype] = torch.cat((attention[dsttype], edge_attention.unsqueeze(0)))

            # formulas (5)
            for ntype in hg.dsttypes:
                attention[ntype] = F.softmax(attention[ntype], dim = 0)

            # formulas (6)
            rst = {ntype: 0 for ntype in hg.dsttypes}
            for ntype, data in outputs.items():
                data = [dst_inputs[ntype]] + data
                if len(data) != 0:
                    for i in range(len(data)):
                        aggregation = torch.mul(data[i], attention[ntype][i])
                        rst[ntype] = aggregation + rst[ntype]
                
            # h = self.conv(hg, hg.ndata['h'], aggregate = self.my_agg_func)
        def _apply(ntype, h):
            if self.bias:
                h = h + self.h_bias
            if self.activation:
                h = self.activation(h)
            if self.batchnorm:
                h = self.bn(h)
            return self.dropout(h)
            
        return {ntype: _apply(ntype, feat) for ntype, feat in rst.items()}