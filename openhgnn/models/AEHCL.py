import math

import torch
import torch as th
import torch.nn as nn
from openhgnn.models import BaseModel, register_model


@register_model("AEHCL")
class AEHCL(BaseModel):
    r"""
        **Title:** Abnormal Event Detection via Hypergraph Contrastive Learning

        **Authors:** Bo Yan, Cheng Yang, Chuan Shi, Jiawei Liu, Xiaochen Wang

        AEHCL was introduced in `[paper] <http://www.shichuan.org/doc/145.pdf>`_
        and parameters are defined as follows:

        Parameters
        ----------
        center_type: str
            Center node's type.
        context_type: list(str)
            All context node's type.
        in_dim: int
            Node features' dimension.
        out_dim: int
            Hidden layers' dimension.
        num_of_attention_heads: int
            The number of attention heads.
        batch_size: int
            The size of each batch.
        all_len: int
            The number of nodes of an event.
        weight_inter: float
            The weight of inter loss function.
        weight_intra_multi: float
            The weight of intra multi loss function.
        weight_intra_pair: float
            The weight of intra pair loss function.
        t: float
            Temperature.

    """

    @classmethod
    def build_model_from_args(cls, args):
        return AEHCL(center_type=args.center_type, context_type=args.context_type, in_dim=args.in_dim,
                     out_dim=args.hidden_dim, num_of_attention_heads=args.num_of_attention_heads,
                     batch_size=args.batch_size, all_len=args.all_len, weight_inter=args.weight_inter,
                     weight_intra_multi=args.weight_intra_multi, weight_intra_pair=args.weight_intra_pair, t=args.t)

    def __init__(self, center_type, context_type, in_dim, out_dim, num_of_attention_heads, batch_size, all_len,
                 weight_inter, weight_intra_multi, weight_intra_pair, t):
        super(AEHCL, self).__init__()

        self.weight_inter = weight_inter
        self.weight_intra_multi = weight_intra_multi
        self.weight_intra_pair = weight_intra_pair

        self.fc = nn.ModuleDict()
        self.fc[center_type] = nn.Sequential(
            nn.Linear(in_dim, out_dim, bias=True),
            nn.ELU(),
            nn.Linear(out_dim, out_dim, bias=True),
        )
        for tp in context_type:
            self.fc[tp] = nn.Sequential(
                nn.Linear(in_dim, out_dim, bias=True),
                nn.ELU(),
                nn.Linear(out_dim, out_dim, bias=True),
            )

        for key in self.fc.keys():
            for model in self.fc[key]:
                if isinstance(model, nn.Linear):
                    nn.init.xavier_normal_(model.weight, gain=1.414)

        self.fc3 = nn.Linear(out_dim * 3, out_dim, bias=False)

        self.ebl = nn.Bilinear(out_dim * 2, out_dim * 2, 1)
        self.ebl_t = nn.Bilinear(out_dim, out_dim, 1)

        self.selfatt = SelfAttention(center_type, context_type, out_dim, num_of_attention_heads, len(context_type) + 1)
        self.maxpool = nn.MaxPool2d(kernel_size=(all_len - 1, 1))
        self.att = Attention(center_type, context_type, out_dim)

        self.sigmoid = nn.Sigmoid()

        self.BCE_loss = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([1.], dtype=torch.float32))

        self.center_type = center_type
        self.batch_size = batch_size
        self.t = t

    def forward(self, event, pos_event, neg_event, neg_context, neg_entity, event_mask, type_num):

        # inter

        # event
        event_center = event[self.center_type]  # (batch,1,in_dim)
        event_center_z = self.fc[self.center_type](event_center)  # (batch,1,out_dim)

        event_context = dict()
        event_context_z = dict()
        for key in event.keys():
            if key != self.center_type:
                event_context[key] = event[key]  # (batch,len[key],in_dim)
                event_context_z[key] = self.fc[key](event_context[key])  # (batch,len[key],out_dim)

        event_z = dict()
        event_z[self.center_type] = event_center_z
        for key in event_context_z.keys():
            event_z[key] = event_context_z[key]

        event_features = self.att(event_z)  # (batch,out_dim*2)

        # pos
        pos_event_z = dict()
        for key in pos_event.keys():
            pos_event_z[key] = self.fc[key](pos_event[key])

        pos_event_features = self.att(pos_event_z)  # (batch,out_dim*2)
        pos_inter = self.ebl(pos_event_features, event_features)  # (batch,1)

        # neg
        neg_event_z = dict()
        for key in neg_event.keys():
            neg_event_z[key] = self.fc[key](neg_event[key])

        neg_event_features = self.att(neg_event_z)  # (batch,out_dim*2)
        neg_inter = self.ebl(neg_event_features, event_features)  # (batch,1)

        # intra_multi

        # pos
        pos_context = None
        type_num_ = None
        for key in event_context_z.keys():
            if pos_context is None:
                pos_context = event_context_z[key]
                type_num_ = type_num[key]
            else:
                pos_context = torch.cat((pos_context, event_context_z[key]), 1)  # (batch,all_len-1,out_dim)
                type_num_ = torch.cat((type_num_, type_num[key]), 1) # (batch,all_len-1)
        pos_context = self.selfatt(pos_context, type_num_)  # (batch,all_len-1,out_dim)
        pos_context = self.maxpool(pos_context)  # (batch,1,out_dim)
        pos_multi = self.ebl_t(pos_context, event_center_z)  # (batch,1,1)
        pos_multi = torch.squeeze(pos_multi, -1)  # (batch,1)

        # neg
        neg_context_h = dict()
        for key in neg_context.keys():
            if key != self.center_type:
                neg_context_h[key] = self.fc[key](neg_context[key])
        neg_context_ = None
        for key in neg_context_h.keys():
            if neg_context_ is None:
                neg_context_ = neg_context_h[key]
            else:
                neg_context_ = torch.cat((neg_context_, neg_context_h[key]), 1)
        neg_context_ = self.selfatt(neg_context_, type_num_)
        neg_context_ = self.maxpool(neg_context_)
        neg_multi = self.ebl_t(neg_context_, event_center_z)  # (batch,1,1)
        neg_multi = torch.squeeze(neg_multi, -1)  # (batch,1)

        # intra_pair

        # pos
        pos_entity_z = event_center_z  # (batch,1,out_dim)
        pos_entity_mask = event_mask[self.center_type] # (batch,1)
        for key in event_context_z.keys():
            pos_entity_z = torch.cat((pos_entity_z, event_context_z[key]), 1)  # (batch,all_len,out_dim)
            pos_entity_mask = torch.cat((pos_entity_mask, event_mask[key]), -1) # (batch,all_len)
        pos_norm = torch.norm(pos_entity_z, dim=-1, keepdim=True)
        pos_entity_z = pos_entity_z / pos_norm  # (batch,all_len,out_dim)
        pos_matrix = torch.matmul(pos_entity_z, torch.transpose(pos_entity_z, 1, 2))  # (batch,all_len,all_len)
        pos_matrix = torch.exp(pos_matrix / self.t)  # (batch,all_len,all_len)
        mask = torch.ones_like(pos_matrix) - torch.eye(pos_matrix.shape[1]).to(pos_matrix.device)
        pos_matrix_ = torch.mul(pos_matrix, mask)  # (batch,all_len,all_len)
        pos_entity_mask_ = torch.unsqueeze(pos_entity_mask, 1) # (batch,1,all_len)
        pos_entity_mask_ = pos_entity_mask_.expand(pos_matrix_.shape) # (batch,all_len,all_len)
        pos_matrix_ = pos_matrix_ * pos_entity_mask_ # (batch,all_len,all_len)


        # neg
        neg_entity_z = None
        for key in neg_entity.keys():
            if neg_entity_z is None:
                neg_entity_z = self.fc[key](neg_entity[key])
            else:
                neg_entity_z = torch.cat((neg_entity_z, self.fc[key](neg_entity[key])), 1) # (batch,all_len,neg_num,out_dim)
        neg_norm = torch.norm(neg_entity_z, dim=-1, keepdim=True) # (batch,all_len,neg_num,1)
        neg_entity_z = neg_entity_z / neg_norm # (batch,all_len,neg_num,out_dim)
        neg_matrix = torch.matmul(torch.unsqueeze(pos_entity_z, 2), torch.transpose(neg_entity_z, 2, 3)) # (batch,all_len,1,neg_num)
        neg_matrix = torch.squeeze(neg_matrix) # (batch,all_len,neg_num)
        neg_matrix_ = torch.exp(neg_matrix / self.t)  # (batch,all_len,neg_num)

        # pos_inter, neg_inter, pos_multi, neg_multi, pos_matrix, neg_matrix -> lose / score
        lbl = torch.unsqueeze(torch.cat((torch.ones(pos_inter.shape[0]), torch.zeros(neg_inter.shape[0]))), 1)  # (2*batch,1)

        # inter
        inter = torch.cat((pos_inter, neg_inter), 0)  # (2*batch,1)
        lbl = lbl.to(inter.device)
        inter_loss = self.BCE_loss(inter, lbl)

        # intra_multi
        intra_multi = torch.cat((pos_multi, neg_multi), 0)  # (2*batch,1)
        intra_multi_loss = self.BCE_loss(intra_multi, lbl)

        # intra_pair
        pos_ = torch.sum(pos_matrix_, -1)  # (batch, all_len)
        neg_ = torch.sum(neg_matrix_, -1)  # (batch, all_len)
        intra_pair = - torch.log((pos_ + 1e-10) / (pos_ + neg_ + 1e-10))  # (batch,all_len)
        intra_pair = intra_pair * pos_entity_mask # (batch,all_len)
        intra_pair_loss = torch.sum(intra_pair, -1)  # (batch)
        intra_pair_loss = intra_pair_loss / torch.sum(pos_entity_mask, -1)

        inter_loss = torch.mean(inter_loss)
        intra_multi_loss = torch.mean(intra_multi_loss)
        intra_pair_loss = torch.mean(intra_pair_loss)

        loss = self.weight_inter * inter_loss + self.weight_intra_multi * intra_multi_loss + self.weight_intra_pair * intra_pair_loss

        if self.training:
            return loss
        else:
            inter_score = self.sigmoid(pos_inter)  # (batch,1)
            inter_score = torch.squeeze(inter_score) # (batch)

            intra_multi_score = self.sigmoid(pos_multi)  # (batch,1)
            intra_multi_score = torch.squeeze(intra_multi_score) # (batch)

            mask = torch.eye(pos_matrix.shape[1]) * 10000000.0  # (all_len,all_len)
            mask = torch.unsqueeze(mask, 0) # (1,all_len,all_len)
            mask = mask.to(pos_matrix.device)
            mask_ = torch.unsqueeze(pos_entity_mask, -1) # (batch,all_len,1)
            mask_ = torch.matmul(mask_, torch.transpose(mask_, 1, 2)) # (batch,all_len,all_len)
            mask_ = - mask_ + 1.0 # (batch,all_len,all_len)
            mask_ = mask_ * 10000000.0

            pos_matrix_ = pos_matrix + mask_ + mask
            intra_pair_score = []
            for i in range(pos_matrix_.shape[0]):
                _ = torch.min(pos_matrix_[i])
                intra_pair_score.append(_)

            intra_pair_score = torch.FloatTensor(intra_pair_score).to(pos_matrix.device) #
            intra_pair_score = self.sigmoid(intra_pair_score)

            score = - self.weight_inter * inter_score - self.weight_intra_multi * intra_multi_score - self.weight_intra_pair * intra_pair_score # (batch)

            return score



class Attention(nn.Module):
    def __init__(self, center_type, context_type, out_dim):
        super().__init__()
        self.center_type = center_type
        self.bilinear = nn.ModuleDict()
        for tp in context_type:
            self.bilinear[tp] = nn.Linear(out_dim, out_dim, bias=False)
            nn.init.xavier_normal_(self.bilinear[tp].weight, gain=1.414)

        self.leakyrelu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)

        self.att_drop = nn.Dropout(0.5)

    def forward(self, event):
        center = event[self.center_type]  # (batch,1,out_dim)
        context_z = None
        context_k = None
        for key in event.keys():
            if key != self.center_type:
                context_k_ = self.bilinear[key](event[key])  # (batch,len[key],out_dim)
                if context_k is None:
                    context_z = event[key]
                    context_k = context_k_
                else:
                    context_z = torch.cat((context_z, event[key]), 1)  # (batch,all_len-1,out_dim)
                    context_k = torch.cat((context_k, context_k_), 1)  # (batch,all_len-1,out_dim)

        context_coef = context_k.matmul(torch.transpose(center, 1, 2))  # (batch,all_len-1,1)
        context_coef = torch.squeeze(context_coef)  # (batch,all_len-1)
        att = self.leakyrelu(context_coef)
        att = self.softmax(att)  # (batch,all_len-1)
        att = torch.unsqueeze(att, 2)  # (batch,all_len-1,1)
        context_aggre = (att * context_z).sum(dim=1)  # (batch,out_dim)
        cur_event = torch.cat((context_aggre, torch.squeeze(center)), 1)  # (batch,out_dim*2)

        return cur_event


class SelfAttention(nn.Module):
    def __init__(self, center_type, context_type, out_dim, num_of_attention_heads, type_n):
        super().__init__()

        self.num_attention_heads = num_of_attention_heads
        self.attention_head_size = int(out_dim / num_of_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(out_dim, self.all_head_size)
        self.key = nn.Linear(out_dim, self.all_head_size)
        self.value = nn.Linear(out_dim, self.all_head_size)

        self.dense = nn.Linear(self.all_head_size, out_dim)

        self.type_embedding = nn.Embedding(type_n, out_dim, padding_idx=0)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, type_num):  # (batch,all_len-1,out_dim)
        type_embedding = self.type_embedding(type_num) # (batch,all_len-1,out_dim)
        hidden_states = hidden_states + type_embedding # (batch,all_len-1,out_dim)

        mixed_query_layer = self.query(hidden_states)  # (batch,all_len-1,all_head_size)
        mixed_key_layer = self.key(hidden_states)  # (batch,all_len-1,all_head_size)
        mixed_value_layer = self.value(hidden_states)  # (batch,all_len-1,all_head_size)

        query_layer = self.transpose_for_scores(mixed_query_layer)  # (batch,num_of_head,all_len-1,head_size)
        key_layer = self.transpose_for_scores(mixed_key_layer)  # (batch,num_of_head,all_len-1,head_size)
        value_layer = self.transpose_for_scores(mixed_value_layer)  # (batch,num_of_head,all_len-1,head_size)

        attention_scores = torch.matmul(query_layer,
                                        key_layer.transpose(-1, -2))  # (batch,num_of_head,all_len-1,all_len-1)
        attention_scores = attention_scores / math.sqrt(
            self.attention_head_size)  # (batch,num_of_head,all_len-1,all_len-1)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)  # (batch,num_of_head,all_len-1,all_len-1)
        context_layer = torch.matmul(attention_probs, value_layer)  # (batch,num_of_head,all_len-1,head_size)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  # (batch,all_len-1,num_of_head,head_size)

        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)  # (batch,all_len-1,all_head_size)

        output = self.dense(context_layer)  # (batch,all_len-1,out_dim)

        return output
