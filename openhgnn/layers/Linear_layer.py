import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Linear_layer(nn.Module):
    def __init__(self, in_ft, out_ft, act=nn.PReLU(), drop_prob=0.0, isBias=False):
        super().__init__()
        self.linear = nn.Linear(in_ft, out_ft, bias=False)

#         if isBias:
#             self.bias = nn.Parameter(torch.empty(out_ft))
#             self.bias.data.fill_(0.0)
#         else:
#             self.register_parameter('bias', None)

        self.act = act
        self.isBias = isBias
        self.drop_prob = drop_prob
        
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, emb):
        # emb (batch_size, ft)
#         emb = F.dropout(emb, self.drop_prob, training=self.training)
        e = self.linear(emb) #  (batch_size, d)
#         if self.isBias:
#             e += self.bias
        e_out = self.act(e)
        return e_out

    
class GNN(nn.Module):
    def __init__(self, nb_rel, in_ft, out_ft, act=nn.PReLU(), drop_prob=0.5, isBias=False):
        super().__init__()
        self.encoder = nn.ModuleList()
        for i in range(nb_rel):
            self.encoder.append(GCN(in_ft, out_ft, act=act, isBias=isBias))

    def forward(self, embs):
        outs = []
        for emb in embs:  # emb (batch_size, ft)
            outs.append(self.encoder(emb))
        outs = torch.stack(outs, 0)  # outs (nb_rel, batch_size, ft)
        return outs
