import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from . import BaseModel, register_model

@register_model('HGNN_AC')
class HGNN_AC(BaseModel):
    r"""
    HGNN_AC was introduced in `HGNN_AC <https://dl.acm.org/doi/10.1145/3442381.3449914>`__.
        
    It included four parts:

    - Pre-learning of Topological Embedding
        HGNN-AC first obtains more comprehensive node sequences by random walk according to the frequently used multiple meta-paths, 
        and then feeds these sequences to the skip-gram model to learn node embeddings :math:`H`.
        
    - Attribute Completion with Attention Mechanism
        HGNN-AC adopts a masked attention mechanism which means we only calculate :math:`e_{vu}` for nodes :math:`u\in{N_v^+}`, 
        where :math:`u\in{N_v^+}` denotes the first-order neighbors of node :math:`v` 
        in set :math:`V^+`, where :math:`V^+` is the set of nodes with attributes.
        
        .. math::
           e_{vu}=\sigma(h_v^{T}Wh_u)
        
        where :math:`W` is the parametric matrix, and :math:`\sigma` an activation function.
    
        Then, softmax function is applied to get normalized weighted coefficient :math:`a_{vu}`

        .. math::
           a_{vu}=softmax(e_{vu})=\frac{exp(e_{vu})}{\sum_{s\in{N_v^+}}{exp(e_{vs})}}

        HGNN-AC can perform weighted aggregation of attributes
        for node :math:`v`  according to weighted coefficient :math:`a_{vu}`  :

        .. math::
           X_v^C=\sum_{u\in{N_v^+}}{a_{vu}x_u}

        where :math:`N_v^+` denotes the set of neighbors of node :math:`v\in{V^+}`,
        and :math:`x_u` denotes the attributes of nodes :math:`u`.

        .. _here:
        
        Specially, the attention process is extended to a multi-head attention
        to stabilize the learning process and reduce the high variance

        .. math::
           X_v^C=mean(\sum_k^K {\sum_{u\in{N_v^+}}{a_{vu}x_u}})

        where :math:`K` means that we perform :math:`K` independent attention process.

    - Dropping some Attributes
        To be specific, for nodes in :math:`V^+`, HGNN-AC randomly divides them into two parts
        :math:`V_{drop}^+` and :math:`V_{keep}^+` according to a small ratio :math:`\alpha`, i.e. :math:`|V_{drop}^+|=\alpha|V^+|`.
        HGNN-AC first drops attributes of nodes in :math:`V_{drop}^+` and then 
        reconstructs these attributes via attributes of nodes :math:`V_{drop}^+` by conducting
        attribute completion.
        
        .. math::
           X_v^C=mean(\sum_k^K {\sum_{u\in{V_{keep}^+ \cap V_i^+}}{a_{vu}x_u}})

        It introduced a weakly supervised loss to optimize the parameters of attribute completion 
        and use euclidean distance as the metric to design the loss function as:
    
        .. math::
           L_{completion}=\frac{1}{|V_{drop}^+|}\sum_{i \in V_{drop}^+} \sqrt{(X_i^C-X_i)^2}
    
    - Combination with HIN Model
        Now, we have completed attributes nodes in :math:`V^-`(the set of nodes without attribute), and the raw attributes nodes in :math:`V+`, 
        Wthen the new attributes of all nodes are defined as:

        .. math::
           X^{new}=\{X_i^C,X_j|\forall i \in V^-, \forall j \in V^+\}

        the new attributes :math:`X^{new}`, together with network topology :math:`A`, as
        a new graph, are sent to the HIN model:

        .. math::
           \overline{Y}=\Phi(A,X^{new})
           L_{prediction}=f(\overline{Y},Y)
        
        where :math:`\Phi` denotes an arbitrary HINs model.

        the overall model can be optimized via back propagation in an end-to-end
        manner:

        .. math::
           L=\lambda L_{completion}+L_{prediction}
    
        where :math:`\lambda` is a weighted coefficient to balance these two parts.
        
    Parameters
    ----------
    in_dim: int
        nodes' topological embedding dimension
    hidden_dim: int
        hidden dimension 
    dropout: float
        the dropout rate of neighbor nodes dropout
    activation: callable activation function
        the activation function used in HGNN_AC.  default: ``F.elu``
    num_heads: int
        the number of heads in attribute completion with attention mechanism
    """
    @classmethod
    def build_model_from_args(cls, args, hg):
        return cls(in_dim = hg.nodes[hg.ntypes[0]].data['h'].shape[1], 
                                hidden_dim = args.attn_vec_dim, 
                                dropout = args.dropout, activation = F.elu, 
                                num_heads = args.num_heads,
                                cuda = False if args.device == torch.device('cpu') else True)
    def __init__(self, in_dim, hidden_dim, dropout, activation, num_heads, cuda):
        super(HGNN_AC, self).__init__()
        self.dropout = dropout
        self.attentions = [AttentionLayer(in_dim, hidden_dim, dropout, activation, cuda) for _ in range(num_heads)]

        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

    def forward(self, bias, emb_dest, emb_src, feature_src):
        r"""
        This is the forward part of model HGNN_AC

        Parameters
        ----------
        bias: matrix
            adjacency matrix related to the source nodes
        emb_dest: matrix
            embeddings of the destination node
        emb_src: matrix
            embeddings of the source node
        feature_src: matrix
            features of the source node
            
        Returns
        -------
        features: matrix
            the new features of the type of node
        """
        
        #Attribute Completion with Attention Mechanism
        adj = F.dropout(bias, self.dropout, training=self.training)
        #x = sum_k(x_v)
        x = torch.cat([att(adj, emb_dest, emb_src, feature_src).unsqueeze(0) for att in self.attentions], dim=0)

        #X_{v}^{C} = mean(x)
        return torch.mean(x, dim=0, keepdim=False)


class AttentionLayer(nn.Module):
    r"""
    This is the attention process used in HGNN\_AC. For more details, you can check here_.
    
    Parameters
    -------------------
    in_dim: int
        nodes' topological embedding dimension
    hidden_dim: int
        hidden dimension
    dropout: float
        the drop rate used in the attention
    activation: callable activation function
        the activation function used in HGNN_AC.  default: ``F.elu``
    """
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
        r"""
        This is the forward part of the attention process.
        
        Parameters
        --------------
        bias: matrix
            the processed adjacency matrix related to the source nodes
        emb_dest: matrix
            the embeddings of the destination nodes
        emb_src: matrix
            the embeddings of the source nodes
        feature_src: matrix
            the features of the source nodes
        
        Returns
        ------------
        features: matrix
            the new features of the nodes
        """
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