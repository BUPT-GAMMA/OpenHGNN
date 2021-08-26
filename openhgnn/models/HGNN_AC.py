import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from . import BaseModel, register_model

@register_model('HGNN_AC')
class HGNN_AC(BaseModel):
    @classmethod
    def build_model_from_args(cls, args, hg):
        return cls(in_dim = hg.nodes[hg.ntypes[0]].data['emb'].shape[1], 
                                hidden_dim = args.attn_vec_dim, 
                                dropout = args.dropout, activation = F.elu, 
                                num_heads = args.num_heads,
                                cuda = False if args.device == torch.device('cpu') else True)
    def __init__(self, in_dim, hidden_dim, dropout, activation, num_heads, cuda):
        r"""

        Description
        -----------
        This is the main method of model HGNN_AC

        Parameters
        ----------
        in_dim: int
            nodes' topological embedding dimension
        hidden_dim: int
            hidden dimension 
        dropout: float
            the dropout rate of neighbor nodes dropout
        activation: callable activation function
            the activation function used in HGNN_AC.  default: F.elu
        num_heads: int
            the number of heads in attribute completion with attention mechanism
        """
        super(HGNN_AC, self).__init__()
        self.dropout = dropout
        self.attentions = [AttentionLayer(in_dim, hidden_dim, dropout, activation, cuda) for _ in range(num_heads)]

        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

    def forward(self, bias, emb_dest, emb_src, feature_src):
        r"""

        Description
        -----------
        This is the forward part of model HGNN_AC

        Parameters
        ----------
        bias: matrix
            adjacency matrix related to the source node type
        emb_dest: matrix
            embeddings of the destination node
        emb_src: matrix
            embeddings of the source node
        feature_src: matrix
            features of the source node
            
        Returns
        -------
        matrix:
        the new features of the type of node
        """
        
        #Attribute Completion with Attention Mechanism
        adj = F.dropout(bias, self.dropout, training=self.training)
        #x = sum_k(x_v)
        x = torch.cat([att(adj, emb_dest, emb_src, feature_src).unsqueeze(0) for att in self.attentions], dim=0)

        #X_{v}^{C} = mean(x)
        return torch.mean(x, dim=0, keepdim=False)


class AttentionLayer(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout, activation, cuda=False):
        super(AttentionLayer, self).__init__()
        self.dropout = dropout
        self.activation = activation
        self.is_cuda = cuda

        self.W = nn.Parameter(nn.init.xavier_normal_(
            torch.Tensor(in_dim, hidden_dim).type(torch.cuda.FloatTensor if cuda else torch.FloatTensor),
            gain=np.sqrt(2.0)), requires_grad=True)
        self.W2 = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(hidden_dim, hidden_dim).type(
            torch.cuda.FloatTensor if cuda else torch.FloatTensor), gain=np.sqrt(2.0)),
            requires_grad=True)

        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, bias, emb_dest, emb_src, feature_src):
        h_1 = torch.mm(emb_src, self.W)
        h_2 = torch.mm(emb_dest, self.W)

        #contribution of the neighbor nodes using a masked attention
        #e_{vu} = activation(h_v * W * h_u)
        e = self.leakyrelu(torch.mm(torch.mm(h_2, self.W2), h_1.t()))
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(bias > 0, e, zero_vec)
        
        #get normalized weighted coefficient
        #a_{vu} = softmax(e_{vu})
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        #x_v = sum(a_{vu} * x_u)
        h_prime = torch.matmul(attention, feature_src)

        #return a new attribute
        return self.activation(h_prime)
