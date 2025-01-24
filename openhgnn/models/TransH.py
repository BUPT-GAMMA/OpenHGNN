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
        self.dis_norm = args.dis_norm

        self.n_emb = nn.Embedding(self.ent_num, self.dim)
        self.r_emb = nn.Embedding(self.rel_num, self.dim)
        self.norm_vector = nn.Embedding(self.rel_num, self.dim)

        nn.init.xavier_uniform_(self.n_emb.weight.data)
        nn.init.xavier_uniform_(self.r_emb.weight.data)
        nn.init.xavier_uniform_(self.norm_vector.weight.data)

    def _transfer(self, n_emb, r_norm):
        if n_emb.shape[0] != r_norm.shape[0]:
            n_emb = n_emb.view(-1, r_norm.shape[0], n_emb.shape[-1])
            r_norm = r_norm.view(-1, r_norm.shape[0], r_norm.shape[-1])
            n_emb = n_emb - th.sum(n_emb * r_norm, -1, True) * r_norm
            return n_emb.view(-1, n_emb.shape[-1])
        else:
            return n_emb - th.sum(n_emb * r_norm, -1, True) * r_norm

    def forward(self, h, r, t):
        if self.training:
            self.n_emb.weight.data = F.normalize(self.n_emb.weight.data, p=2.0, dim=-1)
            self.r_emb.weight.data = F.normalize(self.r_emb.weight.data, p=2.0, dim=-1)
            self.norm_vector.weight.data = F.normalize(self.norm_vector.weight.data, p=2.0, dim=-1)
        if h.shape == th.Size([]):
            h = h.view(1)
        if r.shape == th.Size([]):
            r = r.view(1)
        if t.shape == th.Size([]):
            t = t.view(1)
        h_emb = self.n_emb(h.to(self.device))
        r_emb = self.r_emb(r.to(self.device))
        t_emb = self.n_emb(t.to(self.device))
        r_norm = self.norm_vector(r.to(self.device))
        h_emb = self._transfer(h_emb, r_norm)
        t_emb = self._transfer(t_emb, r_norm)
        h_emb = F.normalize(h_emb, 2.0, -1)
        r_emb = F.normalize(r_emb, 2.0, -1)
        t_emb = F.normalize(t_emb, 2.0, -1)
        score = th.norm(h_emb+r_emb-t_emb, self.dis_norm, dim=-1)
        return score