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

@register_model('TransE')
class TransE(BaseModel):
    @classmethod
    def build_model_from_args(cls, args, hg):
        return cls(args, hg)

    def __init__(self, args, hg):
        super(TransE, self).__init__()
        self.device = args.device
        self.ent_num = hg.num_nodes()
        self.rel_num = len(hg.canonical_etypes)
        self.dim = args.hidden_dim
        self.margin = args.margin
        self.dis_norm = args.dis_norm

        self.n_emb = nn.Embedding(self.ent_num, self.dim)
        self.r_emb = nn.Embedding(self.rel_num, self.dim)
        nn.init.xavier_uniform_(self.n_emb.weight.data)
        nn.init.xavier_uniform_(self.r_emb.weight.data)

    def forward(self, h, r, t):
        if self.training:
            self.r_emb.weight.data = F.normalize(self.r_emb.weight.data, p=2.0, dim=-1)
            self.n_emb.weight.data = F.normalize(self.n_emb.weight.data, p=2.0, dim=-1)
        h_emb = self.n_emb(h.to(self.device))
        r_emb = self.r_emb(r.to(self.device))
        t_emb = self.n_emb(t.to(self.device))
        h_emb = F.normalize(h_emb, 2.0, -1)
        r_emb = F.normalize(r_emb, 2.0, -1)
        t_emb = F.normalize(t_emb, 2.0, -1)
        score = th.norm(h_emb+r_emb-t_emb, self.dis_norm, dim=-1)
        return score