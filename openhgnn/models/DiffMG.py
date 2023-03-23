import torch
import torch.nn as nn
import torch.nn.functional as F
from openhgnn.models import BaseModel, register_model
import numpy as np
@register_model('DiffMG')
class DiffMG(BaseModel):

    @classmethod

    def build_model_from_args(cls, args):
        return cls

    def __init__(self, in_dims, n_hid, n_steps, dropout=None, attn_dim=64, use_norm=True, out_nl=True):
        super(DiffMG, self).__init__()
        self.n_hid = n_hid
        self.ws = nn.ModuleList()
        assert (isinstance(in_dims, list))
        for i in range(len(in_dims)):
            self.ws.append(nn.Linear(in_dims[i], n_hid))
        assert (isinstance(n_steps, list))
        self.metas = nn.ModuleList()
        for i in range(len(n_steps)):
            self.metas.append(Cell(n_steps[i], n_hid, n_hid, use_norm=use_norm, use_nl=out_nl))

        # * [Optional] Combine more than one meta graph?
        self.attn_fc1 = nn.Linear(n_hid, attn_dim)
        self.attn_fc2 = nn.Linear(attn_dim, 1)

        self.feats_drop = nn.Dropout(dropout) if dropout is not None else lambda x: x

    def forward(self, node_feats, node_types, adjs, idxes_seq, idxes_res, gpu):
        if gpu > -1:
            hid = torch.zeros((node_types.size(0), self.n_hid)).cuda()
        else:
            hid = torch.zeros((node_types.size(0), self.n_hid))
        for i in range(len(node_feats)):
            hid[node_types == i] = self.ws[i](node_feats[i])
        hid = self.feats_drop(hid)
        temps = [];
        attns = []
        for i, meta in enumerate(self.metas):
            hidi = meta(hid, adjs, idxes_seq[i], idxes_res[i])
            temps.append(hidi)
            attni = self.attn_fc2(torch.tanh(self.attn_fc1(temps[-1])))
            attns.append(attni)

        hids = torch.stack(temps, dim=0).transpose(0, 1)
        attns = F.softmax(torch.cat(attns, dim=-1), dim=-1)
        out = (attns.unsqueeze(dim=-1) * hids).sum(dim=1)
        return out
class Op(nn.Module):
    def __init__(self):
        super(Op, self).__init__()

    def forward(self, x, adjs, idx):
        return torch.spmm(adjs[idx], x)


class Cell(nn.Module):

    def __init__(self, n_step, n_hid_prev, n_hid, use_norm=True, use_nl=True):
        super(Cell, self).__init__()

        self.affine = nn.Linear(n_hid_prev, n_hid)
        self.n_step = n_step
        self.norm = nn.LayerNorm(n_hid) if use_norm is True else lambda x: x
        self.use_nl = use_nl
        self.ops_seq = nn.ModuleList()
        self.ops_res = nn.ModuleList()
        for i in range(self.n_step):
            self.ops_seq.append(Op())
        for i in range(1, self.n_step):
            for j in range(i):
                self.ops_res.append(Op())

    def forward(self, x, adjs, idxes_seq, idxes_res):

        x = self.affine(x)
        states = [x]
        offset = 0
        for i in range(self.n_step):
            seqi = self.ops_seq[i](states[i], adjs[:-1], idxes_seq[i])  # ! exclude zero Op
            resi = sum(self.ops_res[offset + j](h, adjs, idxes_res[offset + j]) for j, h in enumerate(states[:i]))
            offset += i
            states.append(seqi + resi)
        # assert(offset == len(self.ops_res))

        output = self.norm(states[-1])
        if self.use_nl:
            output = F.gelu(output)
        return output


