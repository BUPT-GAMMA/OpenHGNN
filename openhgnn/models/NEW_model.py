import torch.nn as nn
import torch.nn.functional as F


class MLP_follow_model(nn.Module):

    def __init__(self, model, h_dim, out_dim):
        super(MLP_follow_model, self).__init__()
        self.gnn_model = model

        self.project = nn.Sequential(
            # nn.Linear(h_dim, h_dim),
            # nn.Tanh(),
            nn.Linear(h_dim, out_dim, bias=False)
        )

    def forward(self, hg, h=None, category=None):
        if h is None:
            h = self.gnn_model(hg)
        else:
            h = self.gnn_model(hg, h)
        for i in h:
            h[i] = self.project(h[i])
        return h




