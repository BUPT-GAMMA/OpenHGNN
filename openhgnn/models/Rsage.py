import dgl
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn as dglnn
from . import BaseModel, register_model


@register_model('Rsage')
class Rsage(BaseModel):
    @classmethod
    def build_model_from_args(cls, args, hg):
        return cls(in_dim=args.in_dim,
                   out_dim=args.out_dim,
                   h_dim=args.hidden_dim,
                   etypes=hg.etypes,
                   aggregator_type=args.aggregator_type,
                   num_hidden_layers=args.num_layers - 2,
                   dropout=args.dropout)

    def __init__(self, in_dim,
                 out_dim, 
                 h_dim, 
                 etypes, 
                 aggregator_type,
                 num_hidden_layers=1, 
                 dropout=0):
        super(Rsage, self).__init__()
        self.rel_names = etypes
        self.layers = nn.ModuleList()
        # input 2 hidden
        self.layers.append(RsageLayer(
            in_dim, h_dim, aggregator_type, self.rel_names, activation=F.relu, dropout=dropout))
        for i in range(num_hidden_layers):
            self.layers.append(RsageLayer(
                h_dim, h_dim, aggregator_type, self.rel_names, activation=F.relu, dropout=dropout
            ))
        self.layers.append(RsageLayer(
            h_dim, out_dim, aggregator_type, self.rel_names, activation=None))
        return

    def forward(self, hg, h_dict=None):
        if hasattr(hg, 'ntypes'):
            # full graph training,
            for layer in self.layers:
                h_dict = layer(hg, h_dict)
        else:
            # minibatch training, block
            for layer, block in zip(self.layers, hg):
                h_dict = layer(block, h_dict)
        return h_dict


class RsageLayer(nn.Module):

    def __init__(self,
                 in_feat,
                 out_feat,
                 aggregator_type,
                 rel_names,
                 activation=None,
                 dropout=0.0,
                 bias=True):
        super(RsageLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.aggregator_type = aggregator_type
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.conv = dglnn.HeteroGraphConv({
            rel: dgl.nn.pytorch.SAGEConv(in_feat, out_feat, aggregator_type, bias=bias)
            for rel in rel_names
        })

    def forward(self, g, h_dict):
        h_dict = self.conv(g, h_dict)
        out_put = {}
        for n_type, h in h_dict.items():
            out_put[n_type] = h.squeeze()
        return out_put