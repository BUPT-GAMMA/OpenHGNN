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

        self.n_emb = nn.Embedding(self.ent_num, self.ent_dim)
        self.r_emb = nn.Embedding(self.rel_num, self.rel_dim)
        self.rel_transfer = nn.Embedding(self.rel_num, self.ent_dim * self.rel_dim)
        self.n_emb.weight.data = F.normalize(nn.init.xavier_uniform_(self.n_emb.weight.data), p=2.0, dim=-1)
        self.r_emb.weight.data = F.normalize(nn.init.xavier_uniform_(self.r_emb.weight.data), p=2.0, dim=-1)
        self.rel_transfer.weight.data = F.normalize(nn.init.xavier_uniform_(self.rel_transfer.weight.data), p=2.0, dim=-1)

    def _transfer(self, e, r_transfer):
        r_transfer = r_transfer.view(-1, self.ent_dim, self.rel_dim)
        if e.shape[0] != r_transfer.shape[0]:
            e = e.view(-1, r_transfer.shape[0], self.ent_dim).permute(1, 0, 2)
            e = th.matmul(e, r_transfer).permute(1, 0, 2)
        else:
            e = e.view(-1, 1, self.ent_dim)
            e = th.matmul(e, r_transfer)
        return e.view(-1, self.rel_dim)

    def forward(self, h, r, t):
        self.n_emb.weight.data = F.normalize(self.n_emb.weight.data, p=2.0, dim=-1)
        self.r_emb.weight.data = F.normalize(self.r_emb.weight.data, p=2.0, dim=-1)
        self.rel_transfer.weight.data = F.normalize(self.rel_transfer.weight.data, p=2.0, dim=-1)
        r = r.to(self.device)
        h_emb = self.n_emb(h.to(self.device))
        r_emb = self.r_emb(r)
        t_emb = self.n_emb(t.to(self.device))
        r_transfer = self.rel_transfer(r)
        h_emb = self._transfer(h_emb, r_transfer)
        t_emb = self._transfer(t_emb, r_transfer)
        return h_emb, r_emb, t_emb