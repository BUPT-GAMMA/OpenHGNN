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

@register_model('TransR')
class TransR(BaseModel):
    @classmethod
    def build_model_from_args(cls, args, hg):
        return cls(args, hg)

    def __init__(self, args, hg):
        super(TransR, self).__init__()
        self.device = args.device
        self.ent_num = hg.num_nodes()
        self.rel_num = len(hg.canonical_etypes)
        self.ent_dim = args.ent_dim
        self.rel_dim = args.rel_dim
        self.margin = args.margin
        self.dis_norm = args.dis_norm

        self.n_emb = nn.Embedding(self.ent_num, self.ent_dim)
        self.r_emb = nn.Embedding(self.rel_num, self.rel_dim)
        self.r_emb_p = nn.Embedding(self.rel_num, self.ent_dim * self.rel_dim)
        
        nn.init.xavier_uniform_(self.n_emb.weight.data)
        nn.init.xavier_uniform_(self.r_emb.weight.data)
        nn.init.xavier_uniform_(self.r_emb_p.weight.data)

    def _transfer(self, n, r_emb_p):
        r_emb_p = r_emb_p.view(-1, self.ent_dim, self.rel_dim)
        if n.shape[0] != r_emb_p.shape[0]:
            n = n.view(-1, r_emb_p.shape[0], self.ent_dim).permute(1, 0, 2)
            n = th.matmul(n, r_emb_p).permute(1, 0, 2)
        else:
            n = n.view(-1, 1, self.ent_dim)
            n = th.matmul(n, r_emb_p)
        return n.view(-1, self.rel_dim)

    def forward(self, h, r, t):
        if self.training:
            self.n_emb.weight.data = F.normalize(self.n_emb.weight.data, p=2.0, dim=-1)
            self.r_emb.weight.data = F.normalize(self.r_emb.weight.data, p=2.0, dim=-1)
            self.r_emb_p.weight.data = F.normalize(self.r_emb_p.weight.data, p=2.0, dim=-1)
        if h.shape == th.Size([]):
            h = h.view(1)
        if r.shape == th.Size([]):
            r = r.view(1)
        if t.shape == th.Size([]):
            t = t.view(1)
        r = r.to(self.device)
        h_emb = self.n_emb(h.to(self.device))
        r_emb = self.r_emb(r)
        t_emb = self.n_emb(t.to(self.device))
        r_emb_p = self.r_emb_p(r)
        h_emb = self._transfer(h_emb, r_emb_p)
        t_emb = self._transfer(t_emb, r_emb_p)
        h_emb = F.normalize(h_emb, 2.0, -1)
        r_emb = F.normalize(r_emb, 2.0, -1)
        t_emb = F.normalize(t_emb, 2.0, -1)
        score = th.norm(h_emb+r_emb-t_emb, self.dis_norm, dim=-1)
        return score