import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import math

def clones(module, N):
    return nn.ModuleList([deepcopy(module) for _ in range(N)])

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class Resnet_0(nn.Module):
    def __init__(self, size, dropout):
        super(Resnet_0, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_1, x_2, x_3, sublayer):
        return x_1 + self.dropout(sublayer(self.norm(x_1), self.norm(x_2), x_3))

class Resnet_1(nn.Module):
    def __init__(self, size, dropout):
        super(Resnet_1, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.res_0 = Resnet_0(size, dropout)
        self.res_1 = Resnet_1(size, dropout)
        self.size = size

    def forward(self, x_1, x_2, x_3, mask=None):
        x = self.res_0(x_1, x_2, x_3, lambda x1, x2, x3: self.self_attn(x1, x2, x3, mask))
        return self.res_1(x, self.feed_forward)

class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask1=None, mask2=None):
        for layer in self.layers:
            x = layer(x, mask1, mask2)
        return self.norm(x)

def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, 1e-8)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class TriLevelMultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, d_type, dropout=0.1):
        super(TriLevelMultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.linears_type = clones(nn.Linear(d_type, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.coef = 0.5

    def forward(self, x_1, x_2, x_3, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = x_1.size(0)
        query_r, key_r, value_r = [
            l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (x_1, x_1, x_1))
        ]
        query_n, key_n, value_n = [
            l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (x_2, x_2, x_2))
        ]
        query_t, key_t, value_t = [
            l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears_type, (x_3, x_3, x_3))
        ]
        x_r, attn_r = attention(query_r, key_r, value_r, mask=mask, dropout=self.dropout)
        x_n, attn_n = attention(query_n, key_n, value_n, mask=mask, dropout=self.dropout)
        x_t, attn_t = attention(query_t, key_t, value_t, mask=mask, dropout=self.dropout)
        x = x_r * (1 - self.coef) + (x_n + x_t) * self.coef
        self.attn = attn_r * (1 - self.coef) + (attn_n + attn_t) * self.coef
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))