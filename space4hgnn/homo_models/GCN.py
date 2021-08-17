import dgl
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
from openhgnn.models import BaseModel, register_model


@register_model('GCN')
class GCN(BaseModel):
    @classmethod
    def build_model_from_args(cls, args, hg):
        #g = dgl.to_homogeneous(hg, ndata=['feature'])
        return cls(hg,
                   args.in_dim,
                   args.hidden_dim,
                   args.out_dim,
                   args.num_layers,
                   F.elu,
                   args.dropout)

    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super(GCN, self).__init__()
        self.g = g
        self.h = g.ndata['feature']
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(in_feats, n_hidden, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation))
        # output layer
        self.layers.append(GraphConv(n_hidden, n_classes))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features):
        h = self.h
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(self.g, h)
        return h