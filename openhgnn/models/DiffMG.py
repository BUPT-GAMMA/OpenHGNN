import torch
import torch.nn as nn
import torch.nn.functional as F
from openhgnn.models import BaseModel, register_model
import numpy as np

@register_model('DiffMG')
class DiffMG(BaseModel):

    @classmethod

    def build_model_from_args(cls, args,hg):
        args.search_model = search_model
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
            hid = torch.zeros((len(node_types), self.n_hid)).cuda()
        else:
            hid = torch.zeros((len(node_types), self.n_hid))
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

    # 定义图卷积的操作，其中adjs是邻接矩阵的列表，idx是当前需要使用的邻接矩阵的索引。
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

class Op1(nn.Module):
    '''
    operation for one link in the DAG search space
    '''

    def __init__(self):
        super(Op1, self).__init__()

    def forward(self, x, adjs, ws, idx):
        # assert(ws.size(0) == len(adjs))
        return ws[idx] * torch.spmm(adjs[idx], x)


class Cell1(nn.Module):
    '''
    the DAG search space
    '''

    def __init__(self, n_step, n_hid_prev, n_hid, cstr, use_norm=True, use_nl=True):
        super(Cell1, self).__init__()

        self.affine = nn.Linear(n_hid_prev, n_hid)
        self.n_step = n_step  # * number of intermediate states (i.e., K)
        self.norm = nn.LayerNorm(n_hid, elementwise_affine=False) if use_norm is True else lambda x: x
        self.use_nl = use_nl
        assert (isinstance(cstr, list))
        self.cstr = cstr  # * type constraint

        self.ops_seq = nn.ModuleList()  # * state (i - 1) -> state i, 1 <= i < K
        for i in range(1, self.n_step):
            self.ops_seq.append(Op1())
        self.ops_res = nn.ModuleList()  # * state j -> state i, 0 <= j < i - 1, 2 <= i < K
        for i in range(2, self.n_step):
            for j in range(i - 1):
                self.ops_res.append(Op1())

        self.last_seq = Op1()  # * state (K - 1) -> state K
        self.last_res = nn.ModuleList()  # * state i -> state K, 0 <= i < K - 1
        for i in range(self.n_step - 1):
            self.last_res.append(Op1())

    def forward(self, x, adjs, ws_seq, idxes_seq, ws_res, idxes_res):
        # assert(isinstance(ws_seq, list))
        # assert(len(ws_seq) == 2)
        x = self.affine(x)
        states = [x]
        offset = 0
        for i in range(self.n_step - 1):
            seqi = self.ops_seq[i](states[i], adjs[:-1], ws_seq[0][i], idxes_seq[0][i])  # ! exclude zero Op
            resi = sum(self.ops_res[offset + j](h, adjs, ws_res[0][offset + j], idxes_res[0][offset + j]) for j, h in
                       enumerate(states[:i]))
            offset += i
            states.append(seqi + resi)
        # assert(offset == len(self.ops_res))

        adjs_cstr = [adjs[i] for i in self.cstr]
        out_seq = self.last_seq(states[-1], adjs_cstr, ws_seq[1], idxes_seq[1])
        adjs_cstr.append(adjs[-1])
        out_res = sum(self.last_res[i](h, adjs_cstr, ws_res[1][i], idxes_res[1][i]) for i, h in enumerate(states[:-1]))
        output = self.norm(out_seq + out_res)
        if self.use_nl:
            output = F.gelu(output)
        return output


class search_model(nn.Module):

    def __init__(self, in_dims, n_hid, n_adjs, n_steps, cstr, attn_dim=64, use_norm=True, out_nl=True):
        super(search_model, self).__init__()
        self.cstr = cstr
        self.n_adjs = n_adjs
        self.n_hid = n_hid
        self.ws = nn.ModuleList()  # * node type-specific transformation
        assert (isinstance(in_dims, list))
        for i in range(len(in_dims)):
            self.ws.append(nn.Linear(in_dims[i], n_hid))
        assert (isinstance(n_steps, list))  # * [optional] combine more than one meta graph?
        self.metas = nn.ModuleList()
        for i in range(len(n_steps)):
            self.metas.append(Cell1(n_steps[i], n_hid, n_hid, cstr, use_norm=use_norm, use_nl=out_nl))

        self.as_seq = []  # * arch parameters for ops_seq
        self.as_last_seq = []  # * arch parameters for last_seq
        for i in range(len(n_steps)):
            if n_steps[i] > 1:
                ai = 1e-3 * torch.randn(n_steps[i] - 1, n_adjs - 1)  # ! exclude zero Op
                ai = ai
                ai.requires_grad_(True)
                self.as_seq.append(ai)
            else:
                self.as_seq.append(None)
            ai_last = 1e-3 * torch.randn(len(cstr))
            ai_last = ai_last
            ai_last.requires_grad_(True)
            self.as_last_seq.append(ai_last)

        ks = [sum(1 for i in range(2, n_steps[k]) for j in range(i - 1)) for k in range(len(n_steps))]
        self.as_res = []  # * arch parameters for ops_res
        self.as_last_res = []  # * arch parameters for last_res
        for i in range(len(n_steps)):
            if ks[i] > 0:
                ai = 1e-3 * torch.randn(ks[i], n_adjs)
                ai = ai
                ai.requires_grad_(True)
                self.as_res.append(ai)
            else:
                self.as_res.append(None)

            if n_steps[i] > 1:
                ai_last = 1e-3 * torch.randn(n_steps[i] - 1, len(cstr) + 1)
                ai_last = ai_last
                ai_last.requires_grad_(True)
                self.as_last_res.append(ai_last)
            else:
                self.as_last_res.append(None)

        assert (ks[0] + n_steps[0] + (0 if self.as_last_res[0] is None else self.as_last_res[0].size(0)) == (
                    1 + n_steps[0]) * n_steps[0] // 2)

        # * [optional] combine more than one meta graph?
        self.attn_fc1 = nn.Linear(n_hid, attn_dim)
        self.attn_fc2 = nn.Linear(attn_dim, 1)

    def alphas(self):
        alphas = []
        for each in self.as_seq:
            if each is not None:
                alphas.append(each)
        for each in self.as_last_seq:
            alphas.append(each)
        for each in self.as_res:
            if each is not None:
                alphas.append(each)
        for each in self.as_last_res:
            if each is not None:
                alphas.append(each)
        return alphas

    def sample(self, eps):
        '''
        to sample one candidate edge type per link
        '''
        idxes_seq = []
        idxes_res = []
        if np.random.uniform() < eps:
            for i in range(len(self.metas)):
                temp = []
                temp.append(None if self.as_seq[i] is None else torch.randint(low=0, high=self.as_seq[i].size(-1),
                                                                              size=self.as_seq[i].size()[:-1]))
                temp.append(torch.randint(low=0, high=self.as_last_seq[i].size(-1), size=(1,)))
                idxes_seq.append(temp)
            for i in range(len(self.metas)):
                temp = []
                temp.append(None if self.as_res[i] is None else torch.randint(low=0, high=self.as_res[i].size(-1),
                                                                              size=self.as_res[i].size()[:-1]))
                temp.append(
                    None if self.as_last_res[i] is None else torch.randint(low=0, high=self.as_last_res[i].size(-1),
                                                                           size=self.as_last_res[i].size()[:-1]))
                idxes_res.append(temp)
        else:
            for i in range(len(self.metas)):
                temp = []
                temp.append(None if self.as_seq[i] is None else torch.argmax(F.softmax(self.as_seq[i], dim=-1), dim=-1))
                temp.append(torch.argmax(F.softmax(self.as_last_seq[i], dim=-1), dim=-1))
                idxes_seq.append(temp)
            for i in range(len(self.metas)):
                temp = []
                temp.append(None if self.as_res[i] is None else torch.argmax(F.softmax(self.as_res[i], dim=-1), dim=-1))
                temp.append(
                    None if self.as_last_res[i] is None else torch.argmax(F.softmax(self.as_last_res[i], dim=-1),
                                                                          dim=-1))
                idxes_res.append(temp)
        return idxes_seq, idxes_res

    def forward(self, node_feats, node_types, adjs, idxes_seq, idxes_res):
        hid = torch.zeros((len(node_types), self.n_hid))
        for i in range(len(node_feats)):
            hid[node_types == i] = self.ws[i](node_feats[i])
        temps = [];
        attns = []
        for i, meta in enumerate(self.metas):
            ws_seq = []
            ws_seq.append(None if self.as_seq[i] is None else F.softmax(self.as_seq[i], dim=-1))
            ws_seq.append(F.softmax(self.as_last_seq[i], dim=-1))
            ws_res = []
            ws_res.append(None if self.as_res[i] is None else F.softmax(self.as_res[i], dim=-1))
            ws_res.append(None if self.as_last_res[i] is None else F.softmax(self.as_last_res[i], dim=-1))
            hidi = meta(hid, adjs, ws_seq, idxes_seq[i], ws_res, idxes_res[i])
            temps.append(hidi)
            attni = self.attn_fc2(torch.tanh(self.attn_fc1(temps[-1])))
            attns.append(attni)

        hids = torch.stack(temps, dim=0).transpose(0, 1)
        attns = F.softmax(torch.cat(attns, dim=-1), dim=-1)
        out = (attns.unsqueeze(dim=-1) * hids).sum(dim=1)
        return out

    def parse(self):
        '''
        to derive a meta graph indicated by arch parameters
        '''
        idxes_seq, idxes_res = self.sample(0.)
        msg_seq = [];
        msg_res = []
        for i in range(len(idxes_seq)):
            map_seq = [self.cstr[idxes_seq[i][1].item()]]
            msg_seq.append(map_seq if idxes_seq[i][0] is None else idxes_seq[i][0].tolist() + map_seq)
            assert (len(msg_seq[i]) == self.metas[i].n_step)

            temp_res = []
            if idxes_res[i][1] is not None:
                for item in idxes_res[i][1].tolist():
                    if item < len(self.cstr):
                        temp_res.append(self.cstr[item])
                    else:
                        assert (item == len(self.cstr))
                        temp_res.append(self.n_adjs - 1)
                if idxes_res[i][0] is not None:
                    temp_res = idxes_res[i][0].tolist() + temp_res
            assert (len(temp_res) == self.metas[i].n_step * (self.metas[i].n_step - 1) // 2)
            msg_res.append(temp_res)

        return msg_seq, msg_res
