import torch as th
import torch.nn as nn
import dgl.function as fn
from . import BaseModel, register_model
from torch import Tensor
from torch.nn import Embedding, ModuleList
from dgl.utils import expand_as_pair

@register_model('lightGCN')
class lightGCN(BaseModel):

    r"""
        This module lightGCN was introduced in `lightGCN <https://dl.acm.org/doi/abs/10.1145/3397271.3401063>`__.

        The difference with GCN is that aggregate the entity representation and its neighborhood representation into the entity's embedding, but don't use feature transformation and nonlinear
        activation.
        The message function is defined as follow:

        :math:`\mathbf{e}_u^{(k+1)}=\operatorname{AGG}\left(\mathbf{e}_u^{(k)},\left\{\mathbf{e}_i^{(k)}: i \in \mathcal{N}_u\right\}\right)`

        The AGG is an aggregation function — the core of graph convolution — that considers the k-th layer’s representation of the target node and its neighbor nodes.


        In LightGCN, we adopt the simple weighted sum aggregator and abandon the use of feature transformation and nonlinear activation.
        :math:`\mathbf{e}_u^{(k+1)}=\sum_{i \in \mathcal{N}_u} \frac{1}{\sqrt{\left|\mathcal{N}_u\right|} \sqrt{\left|\mathcal{N}_i\right|}}`
        :math:`\mathbf{e}_i^{(k)}, \\ & \mathbf{e}_i^{(k+1)}=\sum_{u \in \mathcal{N}_i} \frac{1}{\sqrt{\left|\mathcal{N}_i\right|} \sqrt{\left|\mathcal{N}_u\right|}} \mathbf{e}_u^{(k)}`

        In the above equations, :math:`\sigma` is the nonlinear function and
        :math:`\mathrm{W}` and :math:`\mathrm{b}` are transformation weight and bias.
        the representation of an item is bound up with its neighbors by aggregation

        The model prediction is defined as the inner product of user and
        item final representations:

        :math:`\hat{y}_{u i}=\mathbf{e}_u^T \mathbf{e}_i`

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

    def __init__(self, g, args, **kwargs):
        super(lightGCN, self).__init__()

        self.g = g['g']
        self.num_nodes = self.g.shape[0]
        self.num_user = g['user_num']
        self.num_item = g['item_num']
        self.embedding_dim = args.embedding_size
        self.num_layers = args.num_layers
        # if args.alpha is None:
        #     self.alpha = 1. / (self.num_layers + 1)
        self.alpha = 1. / (self.num_layers + 1)
        if isinstance(self.alpha, Tensor):
            assert self.alpha.size(0) == self.num_layers + 1
        else:
            self.alpha = th.tensor([self.alpha] * (self.num_layers + 1))

        self.embedding = Embedding(self.num_nodes, self.embedding_dim)
        self.embedding_user = th.nn.Embedding(
            num_embeddings=self.num_user, embedding_dim=self.embedding_dim)
        self.embedding_item = th.nn.Embedding(
            num_embeddings=self.num_item, embedding_dim=self.embedding_dim)

        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)
        self.f = nn.Sigmoid()

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        # th.nn.init.xavier_uniform_(self.embedding.weight)
        th.nn.init.normal_(self.embedding.weight, std=0.1)

    def computer(self):
        """
        propagate methods for lightGCN
        """
        all_emb = self.embedding.weight
        embs = [all_emb]

        g_droped = self.g

        for layer in range(self.num_layers):

            all_emb = th.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = th.stack(embs, dim=1)

        # print(embs.size())
        light_out = th.mean(embs, dim=1)
        users, items = th.split(light_out, [self.num_user, self.num_item])
        return users, items

    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(th.matmul(users_emb, items_emb.t()))
        return rating

    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb,
         userEmb0, posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) +
                              posEmb0.norm(2).pow(2) +
                              negEmb0.norm(2).pow(2)) / float(len(users))
        pos_scores = th.mul(users_emb, pos_emb)
        pos_scores = th.sum(pos_scores, dim=1)
        neg_scores = th.mul(users_emb, neg_emb)
        neg_scores = th.sum(neg_scores, dim=1)

        loss = th.mean(th.nn.functional.softplus(neg_scores - pos_scores))

        return loss, reg_loss
