import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FullyConnect(nn.Module):
    def __init__(self, in_ft, out_ft, act=nn.Identity(), drop_prob=0.0, isBias=False):
        super().__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        if self.fc.bias is not None:
            self.fc.bias.data.fill_(0.0)

        if isBias:
            self.bias = nn.Parameter(torch.empty(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        self.act = act
        self.drop_prob = drop_prob
        self.isBias = isBias
        
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)


    def forward(self, emb):
        # emb (batch_size, ft)
        emb = F.dropout(emb, self.drop_prob, training=self.training)
        e = self.fc(emb)  #  (batch_size, d)
        if self.isBias:
            e += self.bias
        return self.act(e)

    
class FullyConnect2(nn.Module):
    def __init__(self, in_ft, hid_unit, out_ft, drop_prob=0.0, isBias=False):
        super().__init__()
        self.fc = FullyConnect(in_ft, hid_unit, act=nn.PReLU(), drop_prob=drop_prob,isBias=isBias)
        self.fc2 = FullyConnect(hid_unit, out_ft, act=nn.PReLU(), drop_prob=drop_prob,isBias=isBias)
        self.dense = FullyConnect(out_ft, 1, act=nn.Identity(), drop_prob=drop_prob,isBias=isBias)
        

    def forward(self, emb):
        # emb (batch_size, ft)
        e = self.fc(emb)  #  (batch_size, d)
        e2 = self.fc2(e)
        out = self.dense(e2)
        return out