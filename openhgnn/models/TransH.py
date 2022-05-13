import numpy
import os
import json
import torch as th
import torch.sparse as sparse
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from . import BaseModel, register_model
import dgl

@register_model('TransH')
class TransH(BaseModel):
    @classmethod
    def build_model_from_args(cls, args, hg):
        return cls(args, hg)

    def __init__(self, args, hg):
        super(TransH, self).__init__()
        self.device = args.device
        self.ent_num = hg.num_nodes()
        self.rel_num = len(hg.canonical_etypes)
        self.dim = args.hidden_dim
        self.margin = args.margin

        self.n_emb = nn.Embedding(self.ent_num, self.dim)
        self.r_emb = nn.Embedding(self.rel_num, self.dim)
        self.norm_vector = nn.Embedding(self.rel_num, self.dim)

        nn.init.xavier_uniform_(self.n_emb.weight.data)
        self.r_emb.weight.data = F.normalize(nn.init.xavier_uniform_(self.r_emb.weight.data), p=2.0, dim=-1)
        nn.init.xavier_uniform_(self.norm_vector.weight.data)

    def _transfer(self, e, norm):
        if e.shape[0] != norm.shape[0]:
            e = e.view(-1, norm.shape[0], e.shape[-1])
            norm = norm.view(-1, norm.shape[0], norm.shape[-1])
            e = e - th.sum(e * norm, -1, True) * norm
            return e.view(-1, e.shape[-1])
        else:
            return e - th.sum(e * norm, -1, True) * norm

    def forward(self, h, r, t):
        self.n_emb.weight.data = F.normalize(self.n_emb.weight.data, p=2.0, dim=-1)
        self.norm_vector.weight.data = F.normalize(self.norm_vector.weight.data, p=2.0, dim=-1)
        h_emb = self.n_emb(h.to(self.device))
        r_emb = self.r_emb(r.to(self.device))
        t_emb = self.n_emb(t.to(self.device))
        r_norm = self.norm_vector(r.to(self.device))
        h_emb = self._transfer(h_emb, r_norm)
        t_emb = self._transfer(t_emb, r_norm)
        return h_emb, r_emb, t_emb