import dgl
import torch
import torch.nn as nn
import dgl.function as Fn
import torch.nn.functional as F

from dgl.ops import edge_softmax
from dgl.nn.pytorch import HeteroLinear
from . import BaseModel, register_model
from ..utils import to_hetero_feat

@register_model('HGAT')
class HGAT(BaseModel):
    r"""
    This is a model HGAT from `Heterogeneous Graph Attention Networks for Semi-supervised Short Text Classification
    <https://dl.acm.org/doi/abs/10.1145/3450352>`__

    It contains the following parts:

    Type-level Attention: Given a specific node :math:`v`, we need to calculate the type-level attention scores based on the current node 
    embedding and the type embedding.
    
    .. math::
       a_{\tau} = \sigma(\mu_{\tau}^T \cdot [h_v \parallel h_{\tau}]) \quad (1)
    
    The type embedding is :math:`h_{\tau}=\sum_{v^{'}}\widetilde{A}_{vv^{'}}h_{v^{'}}`, 
    which is the sum of the neighboring node features :math:`h_{v^{'}}` 
    where the nodes :math:`v^{'} \in \mathcal{N}_v` and are with the type :math:`h_{\tau}`.
    :math:`\mu_{\tau}` is the attention vector for the type :math:`\tau`.
    
    And the type-level attention weights is:
    
    .. math::
       \alpha_{\tau} = \frac{exp(a_{\tau})}{\sum_{\tau^{'}\in \mathcal{T}} exp(a_{\tau^{'}})} \quad (2)

    Node-level Attention: Given a specific node :math:`v` and its neightoring node :math:`v^{'}\in \mathcal{N}_v`, 
    we need to calculate the node-level attention scores based on the node embeddings :math:`h_v` and :math:`h_{v^{'}}`
    and with the type-level attention weight :math:`\alpha_{\tau^{'}}` for the node :math:`v^{'}`:
    
    .. math::
       b_{vv^{'}} = \sigma(\nu^T \cdot \alpha_{\tau^{'}}[h_v \parallel h_{v^{'}}]) \quad (3)
    
    where :math:`\nu` is the attention vector.
    
    And the node-level attention weights is:
    
    .. math::
       \beta_{vv^{'}} = \frac{exp(b_{vv^{'}})}{\sum_{i\in \mathcal{N}_v} exp(b_{vi})} \quad (4)
    
    The final output is:
    
    .. math::
       H^{(l+1)} = \sigma(\sum_{\tau \in \mathcal{T}}B_{\tau}\cdot H_{\tau}^{(l)}\cdot W_{\tau}^{(l)}) \quad (5)
    
    Parameters
    ----------
    num_layers: int
        the number of layers we used in the computing
    in_dim: int
        the input dimension
    hidden_dim: int
        the hidden dimension
    num_classes: int
        the number of the output classes
    ntypes: list
        the list of the node type in the graph
    negative_slope: float
        the negative slope used in the LeakyReLU
    """
    @classmethod
    def build_model_from_args(cls, args, hg):
        return cls(args.num_layers,
                   args.hidden_dim,
                   args.num_classes,
                   hg.ntypes,
                   args.negative_slope)
    
    def __init__(self, num_layers, hidden_dim,
                 num_classes, ntypes, negative_slope):
        super(HGAT, self).__init__()
        self.num_layers = num_layers
        self.activation = F.elu
        
        
        self.hgat_layers = nn.ModuleList()
        self.hgat_layers.append(
            TypeAttention(hidden_dim,
                          ntypes,
                          negative_slope))
        self.hgat_layers.append(
            NodeAttention(hidden_dim,
                          hidden_dim,
                          negative_slope)
        )
        for l in range(num_layers - 1):
            self.hgat_layers.append(
                TypeAttention(hidden_dim,
                            ntypes,
                            negative_slope))
            self.hgat_layers.append(
                NodeAttention(hidden_dim,
                            hidden_dim,
                            negative_slope)
            )
        
        self.hgat_layers.append(
            TypeAttention(hidden_dim,
                          ntypes,
                          negative_slope))
        self.hgat_layers.append(
            NodeAttention(hidden_dim,
                          num_classes,
                          negative_slope)
        )
        
        
    def forward(self, hg, h_dict):
        """
        The forward part of the HGAT.
        
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
                attention = self.hgat_layers[2 * l](hg, hg.ndata['h'])
                hg.edata['alpha'] = attention
                g = dgl.to_homogeneous(hg, ndata = 'h', edata = ['alpha'])
                h = self.hgat_layers[2 * l + 1](g, g.ndata['h'], g.ndata['_TYPE'], g.ndata['_TYPE'], presorted = True)
                h_dict = to_hetero_feat(h, g.ndata['_TYPE'], hg.ntypes)
                hg.ndata['h'] = h_dict

        return h_dict

class TypeAttention(nn.Module):
    """
    The type-level attention layer

    Parameters
    ----------
    in_dim: int
        the input dimension of the feature
    ntypes: list
        the list of the node type in the graph
    slope: float
        the negative slope used in the LeakyReLU
    """
    def __init__(self, in_dim, ntypes, slope):
        super(TypeAttention, self).__init__()
        attn_vector = {}
        for ntype in ntypes:
            attn_vector[ntype] = in_dim
        self.mu_l = HeteroLinear(attn_vector, in_dim)
        self.mu_r = HeteroLinear(attn_vector, in_dim)
        self.leakyrelu = nn.LeakyReLU(slope)
        
    def forward(self, hg, h_dict):
        """
        The forward part of the TypeAttention.
        
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
        h_t = {}
        attention = {}
        with hg.local_scope():
            hg.ndata['h'] = h_dict
            for srctype, etype, dsttype in hg.canonical_etypes:
                rel_graph = hg[srctype, etype, dsttype]
                if srctype not in h_dict:
                    continue
                with rel_graph.local_scope():
                    degs = rel_graph.out_degrees().float().clamp(min = 1)
                    norm = torch.pow(degs, -0.5)
                    feat_src = h_dict[srctype]
                    shp = norm.shape + (1,) * (feat_src.dim() - 1)
                    norm = torch.reshape(norm, shp)
                    feat_src = feat_src * norm
                    rel_graph.srcdata['h'] = feat_src
                    rel_graph.update_all(Fn.copy_u('h', 'm'), Fn.sum(msg='m', out='h'))
                    rst = rel_graph.dstdata['h']
                    degs = rel_graph.in_degrees().float().clamp(min=1)
                    norm = torch.pow(degs, -0.5)
                    shp = norm.shape + (1,) * (feat_src.dim() - 1)
                    norm = torch.reshape(norm, shp)
                    rst = rst * norm
                    h_t[srctype] = rst
                    h_l = self.mu_l(h_dict)[dsttype]
                    h_r = self.mu_r(h_t)[srctype]
                    edge_attention = F.elu(h_l + h_r)
                    # edge_attention = F.elu(h_l + h_r).unsqueeze(0)
                    rel_graph.ndata['m'] = {dsttype: edge_attention,
                                    srctype: torch.zeros((rel_graph.num_nodes(ntype = srctype),)).to(edge_attention.device)}
                    # print(rel_graph.ndata)
                    reverse_graph = dgl.reverse(rel_graph)
                    reverse_graph.apply_edges(Fn.copy_u('m', 'alpha'))
                
                hg.edata['alpha'] = {(srctype, etype, dsttype): reverse_graph.edata['alpha']}
                
                # if dsttype not in attention.keys():
                #     attention[dsttype] = edge_attention
                # else:
                #     attention[dsttype] = torch.cat((attention[dsttype], edge_attention))
            attention = edge_softmax(hg, hg.edata['alpha'])
            # for ntype in hg.dsttypes:
            #     attention[ntype] = F.softmax(attention[ntype], dim = 0)

        return attention
    
class NodeAttention(nn.Module):
    """
    The node-level attention layer

    Parameters
    ----------
    in_dim: int
        the input dimension of the feature
    out_dim: int
        the output dimension
    slope: float
        the negative slope used in the LeakyReLU
    """
    def __init__(self, in_dim, out_dim, slope):
        super(NodeAttention, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.Mu_l = nn.Linear(in_dim, in_dim)
        self.Mu_r = nn.Linear(in_dim, in_dim)
        self.leakyrelu = nn.LeakyReLU(slope)
        
    def forward(self, g, x, ntype, etype, presorted = False):
        """
        The forward part of the NodeAttention.

        Parameters
        ----------
        g : object
            the dgl homogeneous graph
        x: tensor
            the original features of the graph
        ntype: tensor
            the node type of the graph
        etype: tensor
            the edge type of the graph
        presorted: boolean
            if the ntype and etype are preordered, default: ``False``
            
        Returns
        -------
        tensor
            The embeddings after aggregation.
        """
        with g.local_scope():
            src = g.edges()[0]
            dst = g.edges()[1]
            h_l = self.Mu_l(x)[src]
            h_r = self.Mu_r(x)[dst]
            edge_attention = self.leakyrelu((h_l + h_r) * g.edata['alpha'])
            edge_attention = edge_softmax(g, edge_attention)
            g.edata['alpha'] = edge_attention
            g.srcdata['x'] = x
            g.update_all(Fn.u_mul_e('x', 'alpha', 'm'),
                         Fn.sum('m', 'x'))
            h = g.ndata['x']
        return h