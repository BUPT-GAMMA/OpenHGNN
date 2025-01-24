import numpy
import os
import json

import numpy as np
import torch as th
import torch.sparse as sparse
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from . import BaseModel, register_model
import dgl

@register_model('GIE')
class GIE(BaseModel):
    @classmethod
    def build_model_from_args(cls, args, hg):
        return cls(args, hg)
    def __init__(self, args, hg):
        super(GIE, self).__init__()
        self.device = args.device
        self.ent_num = hg.num_nodes()
        self.rel_num = len(hg.canonical_etypes)
        self.dim = args.hidden_dim
        self.margin = args.margin
        self.dis_norm = args.dis_norm
        self.scale = th.Tensor([1./np.sqrt(self.dim)]).double().to(self.device)
        self.n_emb = nn.Embedding(self.ent_num, self.dim)
        self.r_emb = nn.Embedding(self.rel_num, self.dim)
        self.r_diagE = nn.Embedding(self.rel_num, self.dim)
        self.r_diagH = nn.Embedding(self.rel_num, self.dim)
        self.r_diagS = nn.Embedding(self.rel_num, self.dim)
        self.context_vec = nn.Embedding(self.rel_num, self.dim)
        self.act = nn.Softmax(dim=1)
        self.c = nn.Parameter(th.ones((self.rel_num, 1), dtype=th.double), requires_grad=True)
        self.v = nn.Parameter(th.ones((self.rel_num, 1), dtype=th.double), requires_grad=True)
        self.u = nn.Parameter(th.ones((self.rel_num, 1), dtype=th.double), requires_grad=True)
        self.MIN_NORM = 1e-15
        nn.init.xavier_uniform_(self.n_emb.weight.data)
        nn.init.xavier_uniform_(self.r_emb.weight.data)
        nn.init.xavier_uniform_(self.r_diagE.weight.data)
        nn.init.xavier_uniform_(self.r_diagH.weight.data)
        nn.init.xavier_uniform_(self.r_diagS.weight.data)
        nn.init.xavier_uniform_(self.context_vec.weight.data)
    def forward(self, h, r, t):
        h = h.to(self.device)
        r = r.to(self.device)
        t = t.to(self.device)
        if h.shape == th.Size([]):
            h = h.view(1).repeat(t.shape[0])
        if r.shape == th.Size([]):
            r = r.view(1).repeat(h.shape[0])
        if t.shape == th.Size([]):
            t = t.view(1).repeat(h.shape[0])
        h_emb = self.n_emb(h)
        r_emb = self.r_emb(r)
        t_emb = self.n_emb(t)
        ## Euclidean space projection
        c = F.softplus(self.c[r])
        r_exp0c = self.expmap0(r_emb, c)
        res_E = self.givens_rotations(self.r_diagE(r), h_emb).view((-1, 1, self.dim))

        ## hyperbolic space projection
        v = F.softplus(self.v[r])
        h_exp0v = self.expmap0(h_emb, v)
        r_exp0v = self.expmap0(r_emb, v)
        lh_H = self.project(self.mobius_add(h_exp0v, r_exp0v, v), v)
        res_H = self.logmap0(self.givens_rotations(self.r_diagH(r), lh_H), v).view((-1, 1, self.dim))

        ## Hyperspherical Space Projection
        u = F.softplus(self.u[r])
        h_exp0u = self.expmap0(h_emb, u)
        r_exp0u = self.expmap0(r_emb, u)
        lh_S = self.project(self.mobius_add(h_exp0u, r_exp0u, u), u)
        res_S = self.logmap0(self.givens_rotations(self.r_diagS(r), lh_S), u).view((-1, 1, self.dim))

        ## Aggregation with weights
        cands = th.cat([res_E, res_H, res_S], dim=1)
        context_vec = self.context_vec(r).view((-1, 1, self.dim))
        att_W = self.act(th.sum(context_vec * cands * self.scale, dim=-1, keepdim=True))
        ## Inter(E_h, H_h, S_h) = exp_0^c(\lambda_E E_h + \lambda_H log_0^v(H_h) + \lambda_S log_0^u(S_h))
        hr_emb = self.project(self.mobius_add(self.expmap0(th.sum(att_W * cands, dim=1), c), r_exp0c, c),c)


        return (self.similarity_score(hr_emb, t_emb, c) + self.margin).view(-1)

    def similarity_score(self, x, v, c):
        sqrt_c = c ** 0.5
        v_norm = th.norm(v, p=2, dim=-1, keepdim=True)
        xv = th.sum(x * v / v_norm, dim=-1, keepdim=True)
        gamma = self.tanh(sqrt_c * v_norm) / sqrt_c
        x2 = th.sum(x * x, dim=-1, keepdim=True)
        c1 = 1 - 2 * c * gamma * xv + c * gamma ** 2
        c2 = 1 - c * x2
        num = th.sqrt((c1 ** 2) * x2 + (c2 ** 2) * (gamma ** 2) - (2 * c1 * c2) * gamma * xv)
        denom = 1 - 2 * c * gamma * xv + (c ** 2) * (gamma ** 2) * x2
        pairwise_norm = num / denom.clamp_min(self.MIN_NORM)
        dist = self.artanh(sqrt_c * pairwise_norm)
        return - (2 * dist / sqrt_c) ** 2

    def tanh(self, x):
        return x.clamp(-15, 15).tanh()

    def artanh(self, x):
        return Artanh.apply(x)

    def givens_rotations(self, r, x):
        ## vector rotation
        givens = r.view((r.shape[0], -1, 2))
        givens = givens / th.norm(givens, p=2, dim=-1, keepdim=True).clamp_min(1e-15)
        x = x.view((r.shape[0], -1, 2))
        x_rot = givens[:, :, 0:1] * x + givens[:, :, 1:] * th.cat((-x[:, :, 1:], x[:, :, 0:1]), dim=-1)
        return x_rot.view((r.shape[0], -1))

    def mobius_add(self, x, y, c):
        ## Möbius addition
        x2 = th.sum(x * x, dim=-1, keepdim=True)
        y2 = th.sum(y * y, dim=-1, keepdim=True)
        xy = th.sum(x * y, dim=-1, keepdim=True)
        num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
        denom = 1 + 2 * c * xy + c ** 2 * x2 * y2
        return num / denom.clamp_min(self.MIN_NORM)

    def expmap0(self, u, c):
        sqrt_c = c ** 0.5
        u_norm = u.norm(dim=-1, p=2, keepdim=True).clamp_min(self.MIN_NORM)
        ## tanh(\sqrt{c} {\lambda_x ||u||}/2)*v/\sqrt{c} {\lambda_x ||u||}/2
        temp = self.tanh(sqrt_c * u_norm) * u
        gamma_1 = temp / (sqrt_c * u_norm)
        return self.project(gamma_1, c)

    def logmap0(self, y, c):
        sqrt_c = c ** 0.5
        y_norm = y.norm(dim=-1, p=2, keepdim=True).clamp_min(self.MIN_NORM)
        return y / y_norm / sqrt_c * self.artanh(sqrt_c * y_norm)

    def project(self, x, c):
        norm = x.norm(dim=-1, p=2, keepdim=True).clamp_min(self.MIN_NORM)
        eps = 1e-5
        ## \frac{1}{\sqrt{c}}
        maxnorm = (1 - eps) / (c ** 0.5)
        cond = norm > maxnorm
        projected = x / norm * maxnorm
        return th.where(cond, projected, x)

class Artanh(th.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.clamp(-1 + 1e-5, 1 - 1e-5)
        ctx.save_for_backward(x)
        dtype = x.dtype
        x = x.double()
        return (th.log_(1 + x).sub_(th.log_(1 - x))).mul_(0.5).to(dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output / (1 - input ** 2)

