import numpy as np
import torch as th
import torch.nn as nn
import dgl.function as fn
import torch.nn.functional as F
from . import BaseModel, register_model
import torch.nn.functional as F

@register_model('KGCN')
class KGCN(BaseModel):
    r"""
    This module KGCN was introduced in `KGCN <https://dl.acm.org/doi/10.1145/3308558.3313417>`__.

    It included two parts:

    Aggregate the entity representation and its neighborhood representation into the entity's embedding.
    The message function is defined as follow:

    :math:`\mathrm{v}_{\mathcal{N}(v)}^{u}=\sum_{e \in \mathcal{N}(v)} \tilde{\pi}_{r_{v, e}}^{u} \mathrm{e}`

    where :math:`\mathrm{e}` is the representation of entity,
    :math:`\tilde{\pi}_{r_{v, e}}^{u}` is the scalar weight on the edge from entity to entity,
    the result :math:`\mathrm{v}_{\mathcal{N}(v)}^{u}` saves message which is passed from neighbor nodes

    There are three types of aggregators.
    Sum aggregator takes the summation of two representation vectors,
    Concat aggregator concatenates the two representation vectors and
    Neighbor aggregator directly takes the neighborhood representation of entity as the output representation

    :math:`a g g_{s u m}=\sigma\left(\mathbf{W} \cdot\left(\mathrm{v}+\mathrm{v}_{\mathcal{S}(v)}^{u}\right)+\mathbf{b}\right)`

    :math:`agg $_{\text {concat }}=\sigma\left(\mathbf{W} \cdot \text{concat}\left(\mathrm{v}, \mathrm{v}_{\mathcal{S}(v)}^{u}\right)+\mathbf{b}\right)$`

    :math:`\text { agg }_{\text {neighbor }}=\sigma\left(\mathrm{W} \cdot \mathrm{v}_{\mathcal{S}(v)}^{u}+\mathrm{b}\right)`

    In the above equations, :math:`\sigma` is the nonlinear function and
    :math:`\mathrm{W}` and :math:`\mathrm{b}` are transformation weight and bias.
    the representation of an item is bound up with its neighbors by aggregation

    Obtain scores using final entity representation and user representation
    The final entity representation is denoted as :math:`\mathrm{v}^{u}`,
    :math:`\mathrm{v}^{u}` do dot product with user representation :math:`\mathrm{u}`
    can obtain the probability. The math formula for the above function is:

    :math:`$\hat{y}_{u v}=f\left(\mathbf{u}, \mathrm{v}^{u}\right)$`

    Parameters
    ----------
    g : DGLGraph
        A knowledge Graph preserves relationships between entities
    args : Config
        Model's config
    """
    @classmethod
    def build_model_from_args(cls, args, g):
        return cls(g, args)

    def __init__(self, g, args):
        super(KGCN, self).__init__()
        self.g = g
        self.args = args
        self.in_dim = args.in_dim
        self.out_dim = args.out_dim
        self.entity_emb_matrix = nn.Parameter(th.FloatTensor(self.g.num_nodes(), self.in_dim))
        self.relation_emb_matrix = nn.Parameter(th.FloatTensor(args.n_relation, self.in_dim))
        self.user_emb_matrix = nn.Parameter(th.FloatTensor(args.n_user, self.in_dim))

        self.Aggregate = KGCN_Aggregate(args)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.entity_emb_matrix, -1, 1)
        nn.init.uniform_(self.relation_emb_matrix, -1, 1)
        nn.init.uniform_(self.user_emb_matrix, -1, 1)

    
    def get_score(self):
        r"""
        Obtain scores using final entity representation and user representation
        
        Returns
        -------

        """
        self.user_embeddings = self.user_emb_matrix[np.array(self.userList)]
        self.scores = th.sum(self.user_embeddings * self.item_embeddings, dim=1)
        self.scores_normalized = th.sigmoid(self.scores)


    def get_embeddings(self):
        return self.user_emb_matrix, self.entity_emb_matrix, self.relation_emb_matrix

    def forward(self, blocks, inputdata):
        r"""
        Predict the probability between user and entity

        Parameters
        ----------
        blocks : list
            Blocks saves the information of neighbor nodes in each layer
        inputdata : numpy.ndarray
            Inputdata contains the relationship between the user and the entity

        Returns
        -------
        labels : torch.Tensor
            the label between users and entities
        scores : torch.Tensor
            Probability of users clicking on entitys
        """
        self.data = inputdata
        self.blocks = blocks
        self.user_indices = self.data[:,0]
        self.itemlist = self.data[:,1]
        self.labels = self.data[:,2]
        self.item_embeddings, self.userList,self.labelList = self.Aggregate(blocks, inputdata)
        self.get_score()
        self.labels = th.tensor(self.labelList).to(self.args.device)

        return self.labels, self.scores


class KGCN_Aggregate(nn.Module):
    def __init__(self, args):
        super(KGCN_Aggregate, self).__init__()
        self.args = args
        self.in_dim = args.in_dim
        self.out_dim = args.out_dim
        if self.args.aggregate == 'CONCAT':
            self.agg = nn.Linear(self.in_dim*2, self.out_dim)
        else:
            self.agg = nn.Linear(self.in_dim, self.out_dim)

    def aggregate(self):
        self.sub_g.update_all(fn.u_mul_e('embedding', 'weight', 'm'),fn.sum('m', 'ft'))
        self.userList = []
        self.labelList = []
        embeddingList = []
        for i in range(len(self.data)):
            weightIndex = np.where(self.itemlist==int(self.sub_g.dstdata['_ID'][i]))
            if self.args.aggregate == 'SUM':
                embeddingList.append(self.sub_g.dstdata['embedding'][i] + self.sub_g.dstdata['ft'][i][weightIndex]) 
            elif self.args.aggregate == 'CONCAT':
                embeddingList.append(th.cat([self.sub_g.dstdata['embedding'][i], self.sub_g.dstdata['ft'][i][weightIndex].squeeze(0)],dim=-1)) 
            elif self.args.aggregate == 'NEIGHBOR':
                embeddingList.append(self.sub_g.dstdata['embedding'][i])
            self.userList.append(int(self.user_indices[weightIndex]))
            self.labelList.append(int(self.labels[weightIndex]))

        self.sub_g.dstdata['embedding'] = th.stack(embeddingList).squeeze(1)
        output = F.dropout(self.sub_g.dstdata['embedding'],p=0)
        if self.layer+1 == len(self.blocks):
            self.item_embeddings = th.tanh(self.agg(output))
        else:
            self.item_embeddings = th.relu(self.agg(output))
        
    def forward(self,blocks,inputdata):
        r"""
        Aggregate the entity representation and its neighborhood representation

        Parameters
        ----------
        blocks : list
            Blocks saves the information of neighbor nodes in each layer
        inputdata : numpy.ndarray
            Inputdata contains the relationship between the user and the entity

        Returns
        -------
        item_embeddings : torch.Tensor
            items' embeddings after aggregated
        userList : list
            Users corresponding to items
        labelList : list
            Labels corresponding to items
        """
        self.data = inputdata
        self.blocks = blocks
        self.user_indices = self.data[:,0]
        self.itemlist = self.data[:,1]
        self.labels = self.data[:,2]
        for self.layer in range(len(blocks)):
            self.sub_g = blocks[self.layer]
            self.aggregate()

        return self.item_embeddings, self.userList, self.labelList

