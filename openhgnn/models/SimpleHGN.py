import dgl
import torch
import torch.nn as nn
import dgl.function as Fn
import torch.nn.functional as F

from dgl.ops import edge_softmax
from dgl.nn.pytorch import TypedLinear
from ..utils import to_hetero_feat
from . import BaseModel, register_model

@register_model('SimpleHGN')
class SimpleHGN(BaseModel):
    r"""
    This is a model SimpleHGN from `Are we really making much progress? Revisiting, benchmarking, and
    refining heterogeneous graph neural networks
    <https://dl.acm.org/doi/pdf/10.1145/3447548.3467350>`__

    The model extend the original graph attention mechanism in GAT by including edge type information into attention calculation.

    Calculating the coefficient:
    
    .. math::
        \alpha_{ij} = \frac{exp(LeakyReLU(a^T[Wh_i||Wh_j||W_r r_{\psi(<i,j>)}]))}{\Sigma_{k\in\mathcal{E}}{exp(LeakyReLU(a^T[Wh_i||Wh_k||W_r r_{\psi(<i,k>)}]))}} \quad (1)
    
    Residual connection including Node residual:
    
    .. math::
        h_i^{(l)} = \sigma(\Sigma_{j\in \mathcal{N}_i} {\alpha_{ij}^{(l)}W^{(l)}h_j^{(l-1)}} + h_i^{(l-1)}) \quad (2)
    
    and Edge residual:
        
    .. math::
        \alpha_{ij}^{(l)} = (1-\beta)\alpha_{ij}^{(l)}+\beta\alpha_{ij}^{(l-1)} \quad (3)
        
    Multi-heads:
    
    .. math::
        h^{(l+1)}_j = \parallel^M_{m = 1}h^{(l + 1, m)}_j \quad (4)
    
    Residual:
    
        .. math::
            h^{(l+1)}_j = h^{(l)}_j + \parallel^M_{m = 1}h^{(l + 1, m)}_j \quad (5)
    
    Parameters
    ----------
    edge_dim: int
        the edge dimension
    num_etypes: int
        the number of the edge type
    in_dim: int
        the input dimension
    hidden_dim: int
        the output dimension
    num_classes: int
        the number of the output classes
    num_layers: int
        the number of layers we used in the computing
    heads: list
        the list of the number of heads in each layer
    feat_drop: float
        the feature drop rate
    negative_slope: float
        the negative slope used in the LeakyReLU
    residual: boolean
        if we need the residual operation
    beta: float
        the hyperparameter used in edge residual
    ntypes: list
        the list of node type
    """
    @classmethod
    def build_model_from_args(cls, args, hg):
        heads = [args.num_heads] * args.num_layers + [1]
        return cls(args.edge_dim,
                   len(hg.etypes),
                   [args.hidden_dim],
                   args.hidden_dim // args.num_heads,
                   args.out_dim,
                   args.num_layers,
                   heads,
                   args.feats_drop_rate,
                   args.slope,
                   True,
                   args.beta,
                   hg.ntypes
                   )

    def __init__(self, edge_dim, num_etypes, in_dim, hidden_dim, num_classes,
                num_layers, heads, feat_drop, negative_slope,
                residual, beta, ntypes):
        super(SimpleHGN, self).__init__()
        self.ntypes = ntypes
        self.num_layers = num_layers
        self.hgn_layers = nn.ModuleList()
        self.activation = F.elu

        # input projection (no residual)
        self.hgn_layers.append(
            SimpleHGNConv(
                edge_dim,
                in_dim[0],
                hidden_dim,
                heads[0],
                num_etypes,
                feat_drop,
                negative_slope,
                False,
                self.activation,
                beta=beta,
            )
        )
        # hidden layers
        for l in range(1, num_layers - 1):  # noqa E741
            # due to multi-head, the in_dim = hidden_dim * num_heads
            self.hgn_layers.append(
                SimpleHGNConv(
                    edge_dim,
                    hidden_dim * heads[l - 1],
                    hidden_dim,
                    heads[l],
                    num_etypes,
                    feat_drop,
                    negative_slope,
                    residual,
                    self.activation,
                    beta=beta,
                )
            )
        # output projection
        self.hgn_layers.append(
            SimpleHGNConv(
                edge_dim,
                hidden_dim * heads[-2],
                num_classes,
                heads[-1],
                num_etypes,
                feat_drop,
                negative_slope,
                residual,
                None,
                beta=beta,
            )
        )

    def forward(self, hg, h_dict):
        """
        The forward part of the SimpleHGN.
        
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
                g = dgl.to_homogeneous(hg, ndata = 'h')
                h = g.ndata['h']
                for l in range(self.num_layers):  # noqa E741
                    h = self.hgn_layers[l](g, h, g.ndata['_TYPE'], g.edata['_TYPE'], True)
                    h = h.flatten(1)

            h_dict = to_hetero_feat(h, g.ndata['_TYPE'], hg.ntypes)
        else:
            # for minibatch training, input h_dict is a tensor
            h = h_dict
            for layer, block in zip(self.hgn_layers, hg):
                h = layer(block, h, block.ndata['_TYPE']['_N'], block.edata['_TYPE'], presorted=False)
            h_dict = to_hetero_feat(h, block.ndata['_TYPE']['_N'][:block.num_dst_nodes()], self.ntypes)

        return h_dict
    
    @property
    def to_homo_flag(self):
        return True

class SimpleHGNConv(nn.Module):
    r"""
    The SimpleHGN convolution layer.

    Parameters
    ----------
    edge_dim: int
        the edge dimension
    num_etypes: int
        the number of the edge type
    in_dim: int
        the input dimension
    out_dim: int
        the output dimension
    num_heads: int
        the number of heads
    num_etypes: int
        the number of edge type
    feat_drop: float
        the feature drop rate
    negative_slope: float
        the negative slope used in the LeakyReLU
    residual: boolean
        if we need the residual operation
    activation: str
        the activation function
    beta: float
        the hyperparameter used in edge residual
    """
    def __init__(self, edge_dim, in_dim, out_dim, num_heads, num_etypes, feat_drop=0.0,
                 negative_slope=0.2, residual=True, activation=F.elu, beta=0.0):
        super(SimpleHGNConv, self).__init__()
        self.edge_dim = edge_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_etypes = num_etypes

        self.edge_emb = nn.Parameter(torch.empty(size=(num_etypes, edge_dim)))

        self.W = nn.Parameter(torch.FloatTensor(
            in_dim, out_dim * num_heads))
        self.W_r = TypedLinear(edge_dim, edge_dim * num_heads, num_etypes)

        self.a_l = nn.Parameter(torch.empty(size=(1, num_heads, out_dim)))
        self.a_r = nn.Parameter(torch.empty(size=(1, num_heads, out_dim)))
        self.a_e = nn.Parameter(torch.empty(size=(1, num_heads, edge_dim)))

        nn.init.xavier_uniform_(self.edge_emb, gain=1.414)
        nn.init.xavier_uniform_(self.W, gain=1.414)
        nn.init.xavier_uniform_(self.a_l.data, gain=1.414)
        nn.init.xavier_uniform_(self.a_r.data, gain=1.414)
        nn.init.xavier_uniform_(self.a_e.data, gain=1.414)

        self.feat_drop = nn.Dropout(feat_drop)
        self.leakyrelu = nn.LeakyReLU(negative_slope)
        self.activation = activation

        if residual:
            self.residual = nn.Linear(in_dim, out_dim * num_heads)
        else:
            self.register_buffer("residual", None)

        self.beta = beta

    def forward(self, g, h, ntype, etype, presorted = False):
        """
        The forward part of the SimpleHGNConv.

        Parameters
        ----------
        g : object
            the dgl homogeneous graph
        h: tensor
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
        emb = self.feat_drop(h)
        emb = torch.matmul(emb, self.W).view(-1, self.num_heads, self.out_dim)
        emb[torch.isnan(emb)] = 0.0

        edge_emb = self.W_r(self.edge_emb[etype], etype, presorted).view(-1,
                                                       self.num_heads, self.edge_dim)

        row = g.edges()[0]
        col = g.edges()[1]

        h_l = (self.a_l * emb).sum(dim=-1)[row]
        h_r = (self.a_r * emb).sum(dim=-1)[col]
        h_e = (self.a_e * edge_emb).sum(dim=-1)

        edge_attention = self.leakyrelu(h_l + h_r + h_e)
        edge_attention = edge_softmax(g, edge_attention)

        if 'alpha' in g.edata.keys():
            res_attn = g.edata['alpha']
            edge_attention = edge_attention * \
                             (1 - self.beta) + res_attn * self.beta
        if self.num_heads == 1:
            edge_attention = edge_attention[:, 0]
            edge_attention = edge_attention.unsqueeze(1)

        with g.local_scope():
            emb = emb.permute(0, 2, 1).contiguous()
            g.edata['alpha'] = edge_attention
            g.srcdata['emb'] = emb
            g.update_all(Fn.u_mul_e('emb', 'alpha', 'm'),
                         Fn.sum('m', 'emb'))
            h_output = g.dstdata['emb'].view(-1, self.out_dim * self.num_heads)
            # h_prime = []
            # for i in range(self.num_heads):
            #     g.edata['alpha'] = edge_attention[:, i]
            #     g.srcdata.update({'emb': emb[i]})
            #     g.update_all(Fn.u_mul_e('emb', 'alpha', 'm'),
            #                  Fn.sum('m', 'emb'))
            #     h_prime.append(g.ndata['emb'])
            # h_output = torch.cat(h_prime, dim=1)

        g.edata['alpha'] = edge_attention
        if g.is_block:
            h = h[:g.num_dst_nodes()]
        if self.residual:
            res = self.residual(h)
            h_output += res
        if self.activation is not None:
            h_output = self.activation(h_output)

        return h_output
