
from . import BaseModel, register_model
import torch
from torch import nn
from torch.nn import functional as F
@register_model('LTE')
class LTE(BaseModel):

    @classmethod
    def build_model_from_args(cls, config):
        return cls(config)

    def __init__(self, config):
        super().__init__()

        self.model = TransE(config)

    def forward(self, *args):
        return self.model(*args)

    def extra_loss(self):
        pass




def get_param(shape):
    param = nn.Parameter(torch.Tensor(*shape))
    nn.init.xavier_normal_(param.data)
    return param


class LTEModel(nn.Module):
    def __init__(self, params=None):
        super(LTEModel, self).__init__()
        self.bceloss = torch.nn.BCELoss()
        self.p = params
        num_ents = self.p.num_ents
        num_rels = self.p.num_rels
        self.init_embed = get_param((num_ents, self.p.init_dim))
        self.device = "cuda"

        self.init_rel = get_param((num_rels * 2, self.p.init_dim))

        self.bias = nn.Parameter(torch.zeros(num_ents))

        self.h_ops_dict = nn.ModuleDict({
            'p': nn.Linear(self.p.init_dim, self.p.gcn_dim, bias=False),
            'b': nn.BatchNorm1d(self.p.gcn_dim),
            'd': nn.Dropout(self.p.hid_drop),
            'a': nn.Tanh(),
        })

        self.t_ops_dict = nn.ModuleDict({
            'p': nn.Linear(self.p.init_dim, self.p.gcn_dim, bias=False),
            'b': nn.BatchNorm1d(self.p.gcn_dim),
            'd': nn.Dropout(self.p.hid_drop),
            'a': nn.Tanh(),
        })

        self.r_ops_dict = nn.ModuleDict({
            'p': nn.Linear(self.p.init_dim, self.p.gcn_dim, bias=False),
            'b': nn.BatchNorm1d(self.p.gcn_dim),
            'd': nn.Dropout(self.p.hid_drop),
            'a': nn.Tanh(),
        })

        self.x_ops = self.p.x_ops
        self.r_ops = self.p.r_ops
        self.diff_ht = False

    def calc_loss(self, pred, label):
        return self.loss(pred, label)

    def loss(self, pred, true_label):
        return self.bceloss(pred, true_label)

    def exop(self, x, r, x_ops=None, r_ops=None, diff_ht=False):
        x_head = x_tail = x
        if len(x_ops) > 0:
            for x_op in x_ops.split("."):
                if diff_ht:
                    x_head = self.h_ops_dict[x_op](x_head)
                    x_tail = self.t_ops_dict[x_op](x_tail)
                else:
                    x_head = x_tail = self.h_ops_dict[x_op](x_head)

        if len(r_ops) > 0:
            for r_op in r_ops.split("."):
                r = self.r_ops_dict[r_op](r)

        return x_head, x_tail, r


class TransE(LTEModel):
    def __init__(self,  params=None):
        super(self.__class__, self).__init__( params)
        num_ents=params.num_ents
        num_rels=params.num_rels
        self.loop_emb = get_param([1, self.p.init_dim])

    def forward(self,g, sub, rel):
        x = self.init_embed
        r = self.init_rel

        x_h, x_t, r = self.exop(x - self.loop_emb, r, self.x_ops, self.r_ops)

        sub_emb = torch.index_select(x_h, 0, sub)
        rel_emb = torch.index_select(r, 0, rel)
        all_ent = x_t

        obj_emb = sub_emb + rel_emb
        x = self.p.gamma - \
            torch.norm(obj_emb.unsqueeze(1) - all_ent, p=1, dim=2)
        score = torch.sigmoid(x)

        return score


class DistMult(LTEModel):
    def __init__(self, num_ents, num_rels, params=None):
        super(self.__class__, self).__init__(num_ents, num_rels, params)

    def forward(self, g, sub, rel):
        x = self.init_embed
        r = self.init_rel

        x_h, x_t, r = self.exop(x, r, self.x_ops, self.r_ops)

        sub_emb = torch.index_select(x_h, 0, sub)
        rel_emb = torch.index_select(r, 0, rel)
        all_ent = x_t

        obj_emb = sub_emb * rel_emb
        x = torch.mm(obj_emb, all_ent.transpose(1, 0))
        x += self.bias.expand_as(x)
        score = torch.sigmoid(x)

        return score


class ConvE(LTEModel):
    def __init__(self, num_ents, num_rels, params=None):
        super(self.__class__, self).__init__(num_ents, num_rels, params)
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(self.p.num_filt)
        self.bn2 = torch.nn.BatchNorm1d(self.p.embed_dim)

        self.hidden_drop = torch.nn.Dropout(self.p.hid_drop)
        self.hidden_drop2 = torch.nn.Dropout(self.p.conve_hid_drop)
        self.feature_drop = torch.nn.Dropout(self.p.feat_drop)
        self.m_conv1 = torch.nn.Conv2d(1, out_channels=self.p.num_filt, kernel_size=(self.p.ker_sz, self.p.ker_sz),
                                       stride=1, padding=0, bias=self.p.bias)

        flat_sz_h = int(2 * self.p.k_w) - self.p.ker_sz + 1
        flat_sz_w = self.p.k_h - self.p.ker_sz + 1
        self.flat_sz = flat_sz_h * flat_sz_w * self.p.num_filt
        self.fc = torch.nn.Linear(self.flat_sz, self.p.embed_dim)

    def concat(self, e1_embed, rel_embed):
        e1_embed = e1_embed.view(-1, 1, self.p.embed_dim)
        rel_embed = rel_embed.view(-1, 1, self.p.embed_dim)
        stack_inp = torch.cat([e1_embed, rel_embed], 1)
        stack_inp = torch.transpose(stack_inp, 2, 1).reshape(
            (-1, 1, 2 * self.p.k_w, self.p.k_h))
        return stack_inp

    def forward(self, g, sub, rel):
        x = self.init_embed
        r = self.init_rel

        x_h, x_t, r = self.exop(x, r, self.x_ops, self.r_ops)

        sub_emb = torch.index_select(x_h, 0, sub)
        rel_emb = torch.index_select(r, 0, rel)
        all_ent = x_t

        stk_inp = self.concat(sub_emb, rel_emb)
        x = self.bn0(stk_inp)
        x = self.m_conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_drop(x)
        x = x.view(-1, self.flat_sz)
        x = self.fc(x)
        x = self.hidden_drop2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = torch.mm(x, all_ent.transpose(1, 0))
        x += self.bias.expand_as(x)

        score = torch.sigmoid(x)
        return score



