import torch as th
import torch.nn as nn
import dgl.function as fn
from . import BaseModel, register_model
from torch import Tensor
from torch.nn import Embedding, ModuleList
from dgl.utils import expand_as_pair

@register_model('lightGCN')
class lightGCN(BaseModel):

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
