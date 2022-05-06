import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn.pytorch as dglnn
import numpy as np

from . import BaseModel, register_model
from ..utils import to_hetero_feat

import sys


@register_model('ieHGCN')
class ieHGCN(BaseModel):
    r"""
    Description
    -----------
    ie-HGCN from paper `Interpretable and Efficient Heterogeneous Graph Convolutional Network
    <https://arxiv.org/pdf/2005.13183.pdf>`__.

    `Source Code Link <https://github.com/kepsail/ie-HGCN>`_
    
    Description
    -----------
    The core part of ie-HGCN, the calculating flow of projection, object-level aggregation and type-level aggregation in
    a specific type block.

    Projection
        .. math::
            Y^{Self-\Omega }=H^{\Omega} \cdot W^{Self-\Omega} (1)-1

            Y^{\Gamma - \Omega}=H^{\Gamma} \cdot W^{\Gamma - \Omega} , \Gamma \in N_{\Omega} (1)-2

    Object-level Aggregation
        .. math::
            Z^{ Self - \Omega } = Y^{ Self - \Omega}=H^{\Omega} \cdot W^{Self - \Omega} (2)-1

            Z^{\Gamma - \Omega}=\hat{A}^{\Omega-\Gamma} \cdot Y^{\Gamma - \Omega} = \hat{A}^{\Omega-\Gamma} \cdot H^{\Gamma} \cdot W^{\Gamma - \Omega} (2)-2

    Type-level Aggregation
        .. math::
            Q^{\Omega}=Z^{Self-\Omega} \cdot W_q^{\Omega} (3)-1

            K^{Self-\Omega}=Z^{Self -\Omega} \cdot W_{k}^{\Omega} (3)-2

            K^{\Gamma - \Omega}=Z^{\Gamma - \Omega} \cdot W_{k}^{\Omega}, \quad \Gamma \in N_{\Omega} (3)-3

        .. math::
            e^{Self-\Omega}={ELU} ([K^{ Self-\Omega} \| Q^{\Omega}] \cdot w_{a}^{\Omega}) (4)-1

            e^{\Gamma - \Omega}={ELU} ([K^{\Gamma - \Omega} \| Q^{\Omega}] \cdot w_{a}^{\Omega}), \Gamma \in N_{\Omega} (4)-2

        .. math::
            [a^{Self-\Omega}\|a^{1 - \Omega}\| \ldots . a^{\Gamma - \Omega}\|\ldots\| a^{|N_{\Omega}| - \Omega}]=
            {softmax}([e^{Self - \Omega}\|e^{1 - \Omega}\| \ldots\|e^{\Gamma - \Omega}\| \ldots \| e^{|\N_{\Omega}| - \Omega}]) (5)

        .. math::
            H_{i,:}^{\Omega \prime}=\sigma(a_{i}^{Self-\Omega} \cdot Z_{i,:}^{Self-\Omega}+\sum_{\Gamma \in N_{\Omega}} a_{i}^{\Gamma - \Omega} \cdot Z_{i,:}^{\Gamma - \Omega}) (6)
    
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
    """
    @classmethod
    def build_model_from_args(cls, args, hg:dgl.DGLGraph):
        return cls(args.num_layers,
                   args.in_dim,
                   args.hidden_dim,
                   args.out_dim,
                   args.attn_dim,
                   hg.ntypes,
                   hg.etypes
                   )

    def __init__(self, num_layers, in_dim, hidden_dim, out_dim, attn_dim, ntypes, etypes):
        super(ieHGCN, self).__init__()
        self.num_layers = num_layers
        self.activation = F.elu
        self.hgcn_layers = nn.ModuleList()
        
        self.hgcn_layers.append(
            ieHGCNConv(
                in_dim,
                hidden_dim,
                attn_dim,
                ntypes,
                etypes,
                self.activation,
            )
        )

        for i in range(1, num_layers - 1):
            self.hgcn_layers.append(
                ieHGCNConv(
                    hidden_dim,
                    hidden_dim,
                    attn_dim,
                    ntypes,
                    etypes,
                    self.activation
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
        with hg.local_scope():
            hg.ndata['h'] = h_dict
            for l in range(self.num_layers):
                h_dict = self.hgcn_layers[l](hg, h_dict)
            
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
        the feature drop rate
    activation: str
        the activation function
    """
    def __init__(self, in_size, out_size, attn_size, ntypes, etypes, activation = F.elu):
        super(ieHGCNConv, self).__init__()
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
        

    def forward(self, hg, h_dict):
        """
        The forward part of the ieHGCNConv.
        
        Parameters
        ----------
        hg : object
            the dgl heterogeneous graph
        h_dict: dict
            the feature dict of different node types
            
        Returns
        -------
        dict
            The embeddings after final aggregation.
        """
        outputs = {ntype: [] for ntype in hg.dsttypes}
        with hg.local_scope():
            hg.ndata['h'] = h_dict
            # formulas (2)-1
            hg.ndata['z'] = self.W_self(hg.ndata['h'])
            query = {}
            key = {}
            attn = {}
            attention = {}
            
            # formulas (3)-1 and (3)-2
            for ntype in hg.dsttypes:
                query[ntype] = self.linear_q[ntype](hg.ndata['z'][ntype])
                key[ntype] = self.linear_k[ntype](hg.ndata['z'][ntype])
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
                    (h_dict[srctype], h_dict[dsttype])
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
                data = [hg.ndata['z'][ntype]] + data
                if len(data) != 0:
                    for i in range(len(data)):
                        aggregation = torch.mul(data[i], attention[ntype][i])
                        rst[ntype] = aggregation + rst[ntype]
                
            # h = self.conv(hg, hg.ndata['h'], aggregate = self.my_agg_func)
        if self.activation is not None:
            for ntype in rst.keys():
                rst[ntype] = self.activation(rst[ntype])
            
        return rst