import torch.nn as nn
import torch.nn.functional as F
import dgl.nn as dglnn
from openhgnn.models import BaseModel, register_model
from dgl.nn.pytorch.conv import GINConv

@register_model('GIN')
class GIN(BaseModel):
    @classmethod
    def build_model_from_args(cls, args, hg):
        return cls(input_dim=args.input_dim,
                   hidden_dim=args.hidden_dim,
                   output_dim=args.out_dim,
                   num_hidden_layers=args.num_layers - 1,
                   rel_names=hg.etypes,
                   learn_eps=args.learn_eps,
                   aggregate=args.aggregate)

    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden_layers, rel_names, learn_eps, aggregate):
        super(GIN, self).__init__()
        self.rel_names = rel_names
        self.learn_eps = learn_eps
        self.aggregate = aggregate
        self.layers = nn.ModuleList()
        # input 2 hidden
        self.layers.append(GINLayer(
            input_dim, hidden_dim, self.rel_names, self.learn_eps, self.aggregate))
        for i in range(num_hidden_layers):
            self.layers.append(GINLayer(
                hidden_dim, hidden_dim, self.rel_names, self.learn_eps, self.aggregate
            ))
        self.linear_prediction = nn.ModuleList()
        for _ in range(num_hidden_layers + 1):
            self.linear_prediction.append(nn.Linear(hidden_dim, output_dim))
        self.drop = nn.Dropout(0.5)

    def forward(self, hg, h_dict):
        output_dict = dict()
        logits = dict()
        
        if hasattr(hg, 'ntypes'):
            # full graph training,
            for ntype in hg.ntypes:
                output_dict[ntype] = []
                logits[ntype] = 0
            # Perform forward pass through all layers of the GNN
            for layer in self.layers:
                h_dict = layer(hg, h_dict)
                for ntype in hg.ntypes:
                    output_dict[ntype].append(h_dict[ntype])
            # perform graph sum pooling over all nodes in each layer
            for ntype in hg.ntypes:
                for i, h in enumerate(output_dict[ntype]):
                    logits[ntype] = logits[ntype] + self.drop(self.linear_prediction[i](h))
                logits[ntype] = F.softmax(logits[ntype], dim=-1)
                
        else:
            # minibatch training, block
            for layer, block in zip(self.layers, hg):
                for ntype in block.ntypes:
                    output_dict[ntype] = []
                    logits[ntype] = 0
                # Perform forward pass through all layers of the GNN
                h_dict = layer(block, h_dict)
                for ntype in block.ntypes:
                    output_dict[ntype].append(h_dict[ntype])
            # perform graph sum pooling over all nodes in each layer
            for ntype in block.ntypes:
                for i, h in enumerate(output_dict[ntype]):
                    logits[ntype] = logits[ntype] + self.drop(self.linear_prediction[i](h))
                logits[ntype] = F.softmax(logits[ntype], dim=-1)
        
        return logits

class GINLayer(nn.Module):
    def __init__(self, input_dim, output_dim, rel_names, learn_eps, aggregate):
        super(GINLayer, self).__init__()
        self.conv = dglnn.HeteroGraphConv({
            rel: GINBase(input_dim, output_dim, learn_eps)
            for rel in rel_names
        }, aggregate)

    def forward(self, g, h_dict):
        h_dict = self.conv(g, h_dict)
        out_put = {}
        for n_type, h in h_dict.items():
            out_put[n_type] = h.squeeze()
        return out_put

class GINBase(nn.Module):
    def __init__(self, input_dim, output_dim, learn_eps):
        super(GINBase, self).__init__()

        mlp = MLP(input_dim, output_dim)
        self.ginlayer = GINConv(mlp, learn_eps=learn_eps) # set to True if learning epsilon
        self.batch_norm = nn.BatchNorm1d(output_dim)

    def forward(self, g, h):
        h = self.ginlayer(g, h)
        h = self.batch_norm(h)
        h = F.relu(h)
        return h

class MLP(nn.Module):
    """Construct two-layer MLP-type aggreator for GIN model"""

    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.linears = nn.ModuleList()
        # two-layer MLP
        self.linears.append(nn.Linear(input_dim, output_dim, bias=False))
        self.linears.append(nn.Linear(output_dim, output_dim, bias=False))
        self.batch_norm = nn.BatchNorm1d((output_dim))

    def forward(self, x):
        h = x
        h = F.relu(self.batch_norm(self.linears[0](h)))
        return self.linears[1](h)
