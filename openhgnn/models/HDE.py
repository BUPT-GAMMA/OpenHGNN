import torch
import torch.nn as nn
import torch.nn.functional as F
from . import BaseModel, register_model


class GNN(nn.Module):
    """
    Aggregate 2-hop neighbor.
    """
    def __init__(self, input_dim, output_dim, num_neighbor, use_bias=True):
        super(GNN, self).__init__()
        self.input_dim = int(input_dim)
        self.num_fea = int(input_dim)
        self.output_dim = int(output_dim)
        self.num_neighbor = num_neighbor
        self.use_bias = use_bias
        self.linear1 = nn.Linear(self.input_dim * 2, 64)
        self.linear2 = nn.Linear(64+self.num_fea, 64)
        self.linear3 = nn.Linear(64, self.output_dim)

    def forward(self, fea):
        node = fea[:, :self.num_fea]
        neigh1 = fea[:, self.num_fea:self.num_fea * (self.num_neighbor + 1)]
        neigh1 = torch.reshape(neigh1, [-1, self.num_neighbor, self.num_fea])

        neigh2 = fea[:, self.num_fea * (self.num_neighbor + 1):]
        neigh2 = torch.reshape(neigh2, [-1, self.num_neighbor, self.num_neighbor, self.num_fea])
        neigh2_agg = torch.mean(neigh2, dim=2)
        tmp = torch.cat([neigh1, neigh2_agg], dim=2)
        tmp = F.relu(self.linear1(tmp))
        emb = torch.cat([node, torch.mean(tmp, dim=1)], dim=1)
        emb = F.relu(self.linear2(emb))
        emb = F.relu(self.linear3(emb))

        return emb


@register_model('HDE')
class HDE(BaseModel):
    def __init__(self, input_dim, output_dim, num_neighbor, use_bias=True):
        super(HDE, self).__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.num_neighbor = num_neighbor
        self.use_bias = use_bias
        self.aggregator = GNN(input_dim=input_dim, output_dim=output_dim, num_neighbor=num_neighbor)
        self.linear1 = nn.Linear(2*self.output_dim, 32)
        self.linear2 = nn.Linear(32, 2)

    def forward(self, fea_a, fea_b):
        emb_a = self.aggregator(fea_a)
        emb_b = self.aggregator(fea_b)
        emb = torch.cat([emb_a, emb_b], dim=1)
        emb = F.relu(self.linear1(emb))
        output = self.linear2(emb)

        return output

    @classmethod
    def build_model_from_args(cls, args, hg):
        return cls(input_dim=args.input_dim,
                   output_dim=args.output_dim,
                   num_neighbor=args.num_neighbor,
                   use_bias=args.use_bias)
