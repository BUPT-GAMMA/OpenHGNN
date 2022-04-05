import dgl
import torch
import torch.nn as nn
import dgl.function as Fn
import torch.nn.functional as F

from dgl.ops import edge_softmax
from dgl.nn import TypedLinear
from ..utils import to_hetero_feat
from . import BaseModel, register_model

@register_model('HetSANN')
class HetSANN(BaseModel):
    @classmethod
    def build_model_from_args(cls, args, hg):
        return cls(
            args.num_heads,
            args.num_layers,
            args.hidden_dim,
            args.h_dim,
            args.out_dim,
            len(hg.etypes),
            args.dropout,
            args.slope,
            args.residual,
            )
    
    def __init__(self, num_heads, num_layers, in_dim, hidden_dim, 
                 num_classes, num_etypes, dropout, negative_slope, residual):
        """
        This is a model HetSANN from `An Attention-Based Graph Neural Network for Heterogeneous Structural Learning
        <https://arxiv.org/abs/1912.10832>`__

        It contains the following part:

        Apply a linear transformation:
        
        ..math::
            h^{(l+1, m)}_{\phi(j),i} = W^{(l+1, m)}_{\phi(j),\phi(i)} h^{(l)}_i  (1)
        
        And return the new embeddings.
        
        You may refer to the paper HetSANN-Section 2.1-Type-aware Attention Layer-(1)

        Aggregation of Neighborhood:
        
        Computing the attention coefficient:
        
        ..math::
            o^{(l+1,m)}_e = \sigma(f^{(l+1,m)}_r(h^{(l+1, m)}_{\phi(j),j}, h^{(l+1, m)}_{\phi(j),i}))  (2)
            
        ..math::
            f^{(l+1,m)}_r(e) = [h^{(l+1, m)^T}_{\phi(j),j}||h^{(l+1, m)^T}_{\phi(j),i}]a^{(l+1, m)}_r ] (3)
        
        ..math::
            \alpha^{(l+1,m)}_e = exp(o^{(l+1,m)}_e) / \sum_{k\in \varepsilon_j} exp(o^{(l+1,m)}_k)  (4)
        
        Getting new embeddings with multi-head and residual
        
        ..math::
            h^{(l + 1, m)}_j = \sigma(\sum_{e = (i,j,r)\in \varepsilon_j} \alpha^{(l+1,m)}_e h^{(l+1, m)}_{\phi(j),i})  (5)
        
        Multi-heads:
        
        ..math::
            h^{(l+1)}_j = \parallel^M_{m = 1}h^{(l + 1, m)}_j  (6)
        
        Residual:
        
        ..math::
            h^{(l+1)}_j = h^{(l)}_j + \parallel^M_{m = 1}h^{(l + 1, m)}_j  (7)
        
        Parameters
        ----------
        num_heads: int
            the number of heads in the attention computing
        num_layers: int
            the number of layers we used in the computing
        in_dim: int
            the input dimension
        hidden_dim: int
            the hidden dimension
        num_classes: int
            the number of the output classes
        num_etypes: int
            the number of the edge types
        dropout: float
            the dropout rate
        negative_slope: float
            the negative slope used in the LeakyReLU
        residual: boolean
            if we need the residual operation

        """
        super(HetSANN, self).__init__()
        self.num_heads = num_heads
        self.num_layers = num_layers
        # self.dropout = nn.Dropout(dropout)
        self.residual = residual
        self.activation = F.elu
        
        self.het_layers = nn.ModuleList()
        
        # input projection
        self.het_layers.append(
            HetSANNConv(
                num_heads,
                in_dim,
                hidden_dim,
                num_etypes,
                dropout,
                negative_slope,
                False,
                self.activation,
            )
        )

        # hidden layer
        for i in range(1, num_layers - 1):
            self.het_layers.append(
                HetSANNConv(
                    num_heads,
                    hidden_dim * num_heads,
                    hidden_dim,
                    num_etypes,
                    dropout,
                    negative_slope,
                    residual,
                    self.activation
                )
            )

        # output projection
        self.het_layers.append(
            HetSANNConv(
                1,
                hidden_dim * num_heads,
                num_classes,
                num_etypes,
                dropout,
                negative_slope,
                residual,
                None,
            )
        )

    def forward(self, hg, h_dict):
        """
        The forward part of the HetSANN.
        
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
            # input layer and hidden layers
            hg.ndata['h'] = h_dict
            g = dgl.to_homogeneous(hg, ndata = 'h')
            h = g.ndata['h']
            for i in range(self.num_layers - 1):
                h = self.het_layers[i](g, h, g.ndata['_TYPE'], g.edata['_TYPE'], True)

            # output layer
            h = self.het_layers[-1](g, h, g.ndata['_TYPE'], g.edata['_TYPE'], True)

            h_dict = to_hetero_feat(h, g.ndata['_TYPE'], hg.ntypes)
            # g.ndata['h'] = h
            # hg = dgl.to_heterogeneous(g, hg.ntypes, hg.etypes)
            # h_dict = hg.ndata['h']

            # for etype in hg.etypes:
            #     source = etype.split('-')[0]
            #     h[source] = self.W_out[etype](h_dict[source])
            # pre_h = dgl.to_homogeneous(hg, ndata = 'h').ndata['h']
            # hg.ndata['h'] = h
            # g = dgl.to_homogeneous(hg, ndata = 'h')
            # h = self.het_layers[-1](g, pre_h)
            # hg = dgl.to_heterogeneous(g, hg.ntypes, hg.etypes)
            # h_dict = hg.ndata['h']
            
        return h_dict

class HetSANNConv(nn.Module):
    def __init__(self, num_heads, in_dim, hidden_dim, num_etypes,
                 dropout, negative_slope, residual, activation):
        """
        The HetSANN convolution layer.

        Parameters
        ----------
        num_heads: int
            the number of heads in the attention computing
        in_dim: int
            the input dimension of the feature
        hidden_dim: int
            the hidden dimension
        num_etypes: int
            the number of the edge types
        dropout: float
            the dropout rate
        negative_slope: float
            the negative slope used in the LeakyReLU
        residual: boolean
            if we need the residual operation
        activation: str
            the activation function
        """
        super(HetSANNConv, self).__init__()
        self.num_heads = num_heads
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

        self.W = TypedLinear(in_dim, hidden_dim * num_heads, num_etypes)
        # self.W_out = TypedLinear(hidden_dim * num_heads, num_classes, num_etypes)

        # self.W_hidden = nn.ModuleDict()
        # self.W_out = nn.ModuleDict()
        
        # for etype in etypes:
        #     self.W_hidden[etype] = nn.Linear(in_dim, hidden_dim * num_heads)
        
        # for etype in etypes:
        #     self.W_out[etype] = nn.Linear(hidden_dim * num_heads, num_classes)
        
        self.a_l = TypedLinear(self.hidden_dim, self.hidden_dim, num_etypes)
        self.a_r = TypedLinear(self.hidden_dim, self.hidden_dim, num_etypes)
        
        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(negative_slope)
        if residual:
            self.residual = nn.Linear(in_dim, hidden_dim * num_heads)
        else:
            self.register_buffer("residual", None)
            
        self.activation = activation
        
        
    def forward(self, g, x, ntype, etype, presorted = False):
        """
        The forward part of the HetSANNConv.

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
        # formula (1)
        feat = self.W(x, ntype, presorted)
        h = self.dropout(feat)
        h = feat.view(-1, self.num_heads, self.hidden_dim)
        
        src = g.edges()[0]
        dst = g.edges()[1]
        
        # formula (2) (3) (4)
        h_l = self.a_l(h.view(-1, self.hidden_dim), ntype, presorted) \
        .view(-1, self.num_heads, self.hidden_dim).sum(dim = -1)[src]

        h_r = self.a_r(h.view(-1, self.hidden_dim), ntype, presorted) \
        .view(-1, self.num_heads, self.hidden_dim).sum(dim = -1)[dst]
        
        attention = self.leakyrelu(h_l + h_r)
        attention = edge_softmax(g, attention)
        
        # formula (5) (6)
        with g.local_scope():
            h = h.permute(0, 2, 1).contiguous()
            g.edata['alpha'] = attention
            g.srcdata['emb'] = h
            g.update_all(Fn.u_mul_e('emb', 'alpha', 'm'),
                         Fn.sum('m', 'emb'))
            h_output = g.ndata['emb'].view(-1, self.hidden_dim * self.num_heads)

            # h_prime = []
            # h = h.permute(1, 0, 2).contiguous()
            # for i in range(self.num_heads):
            #     g.edata['alpha'] = attention[:, i]
            #     g.srcdata.update({'emb': h[i]})
            #     g.update_all(Fn.u_mul_e('emb', 'alpha', 'm'),
            #                  Fn.sum('m', 'emb'))
            #     h_prime.append(g.ndata['emb'])
            # h_output = torch.cat(h_prime, dim=1)

        # formula (7)
        if self.residual:
            res = self.residual(x)
            h_output += res
        
        if self.activation is not None:
            h_output = self.activation(h_output)       
        
        return h_output