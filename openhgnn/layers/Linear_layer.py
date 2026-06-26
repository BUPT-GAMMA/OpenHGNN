import torch
import torch.nn as nn

class Linear_layer(nn.Module):
    def __init__(self, in_ft, out_ft, act=nn.PReLU(), drop_prob=0.0, isBias=False):
        super().__init__()
        self.linear = nn.Linear(in_ft, out_ft, bias=False)
        self.act = act

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, emb):
        e = self.linear(emb)
        e_out = self.act(e)
        return e_out