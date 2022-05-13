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

@register_model('TransD')
class TransD(BaseModel):
    @classmethod
    def build_model_from_args(cls, args, hg):
        return cls(args, hg)

    def __init__(self, args, hg):
        super(TransD, self).__init__()
        self.device = args.device
        self.ent_num = hg.num_nodes()
        self.rel_num = len(hg.canonical_etypes)
        self.ent_dim = args.ent_dim
        self.rel_dim = args.rel_dim
        self.margin = args.margin

        self.n_emb = nn.Embedding(self.ent_num, self.ent_dim)
        self.r_emb = nn.Embedding(self.rel_num, self.rel_dim)
        self.ent_transfer = nn.Embedding(self.ent_num, self.ent_dim)
        self.rel_transfer = nn.Embedding(self.rel_num, self.rel_dim)
        self.n_emb.weight.data = F.normalize(nn.init.xavier_uniform_(self.n_emb.weight.data), p=2.0, dim=-1)
        self.r_emb.weight.data = F.normalize(nn.init.xavier_uniform_(self.r_emb.weight.data), p=2.0, dim=-1)
        self.ent_transfer.weight.data = F.normalize(nn.init.xavier_uniform_(self.ent_transfer.weight.data), p=2.0, dim=-1)
        self.rel_transfer.weight.data = F.normalize(nn.init.xavier_uniform_(self.rel_transfer.weight.data), p=2.0, dim=-1)

    def _resize(self, tensor, axis, size):
        shape = tensor.size()
        osize = shape[axis]
        if osize == size:
            return tensor
        if (osize > size):
            return th.narrow(tensor, axis, 0, size)
        paddings = []
        for i in range(len(shape)):
            if i == axis:
                paddings = [0, size - osize] + paddings
            else:
                paddings = [0, 0] + paddings
        print (paddings)
        return F.pad(tensor, paddings = paddings, mode = "constant", value = 0)

    def _transfer(self, e, e_transfer, r_transfer):
        if e.shape[0] != r_transfer.shape[0]:
            e = e.view(-1, r_transfer.shape[0], e.shape[-1])
            e_transfer = e_transfer.view(-1, r_transfer.shape[0], e_transfer.shape[-1])
            r_transfer = r_transfer.view(-1, r_transfer.shape[0], r_transfer.shape[-1])
            e = F.normalize(
                self._resize(e, -1, r_transfer.size()[-1]) + th.sum(e * e_transfer, -1, True) * r_transfer,
                p = 2, 
                dim = -1
            )            
            return e.view(-1, e.shape[-1])
        else:
            return F.normalize(
                self._resize(e, -1, r_transfer.size()[-1]) + th.sum(e * e_transfer, -1, True) * r_transfer,
                p = 2, 
                dim = -1
            )

    def forward(self, h, r, t):
        self.n_emb.weight.data = F.normalize(self.n_emb.weight.data, p=2.0, dim=-1)
        self.r_emb.weight.data = F.normalize(self.r_emb.weight.data, p=2.0, dim=-1)
        self.ent_transfer.weight.data = F.normalize(self.ent_transfer.weight.data, p=2.0, dim=-1)
        self.rel_transfer.weight.data = F.normalize(self.rel_transfer.weight.data, p=2.0, dim=-1)

        h = h.to(self.device)
        r = r.to(self.device)
        t = t.to(self.device)
        h_emb = self.n_emb(h)
        r_emb = self.r_emb(r)
        t_emb = self.n_emb(t)
        h_transfer = self.ent_transfer(h)
        r_transfer = self.rel_transfer(r)
        t_transfer = self.ent_transfer(t)
        h_emb = self._transfer(h_emb, h_transfer, r_transfer)
        t_emb = self._transfer(t_emb, t_transfer, r_transfer)
        return h_emb, r_emb, t_emb