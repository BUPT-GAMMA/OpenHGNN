import dgl
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
from openhgnn.models import BaseModel, register_model, hetero_linear


@register_model('GCN')
class GCN(BaseModel):
    @classmethod
    def build_model_from_args(cls, args, hg):

        return cls(hg,
                   args.in_dim,
                   args.hidden_dim,
                   args.out_dim,
                   args.num_layers,
                   F.elu,
                   args.dropout)

    def __init__(self,
                 hg,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super(GCN, self).__init__()
        self.hg = hg
        self.layers = nn.ModuleList()
        # input layer
        #self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation))
        # hidden layers
        for i in range(n_layers):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation))
        # output layer
        self.layers.append(GraphConv(n_hidden, n_classes))
        self.dropout = nn.Dropout(p=dropout)


        linear_list1 = []
        for ntype in self.hg.ntypes:
            in_dim = self.hg.nodes[ntype].data['h'].shape[1]
            linear_list1.append((ntype, in_dim, n_hidden))
        # * ================== Project feature Layer==================
        self.hetero_linear = hetero_linear(linear_list1)
        #self.preprocess()

    def preprocess(self):
        self.g = dgl.to_homogeneous(self.hg)
        self.h = self.g.ndata.pop('h')

    def forward(self, hg):
        with hg.local_scope():
            hg.ndata['h'] = self.hetero_linear(hg.ndata['h'])
            homo_g = dgl.to_homogeneous(hg, ndata=['h'])
            homo_g = dgl.remove_self_loop(homo_g)
            homo_g = dgl.add_self_loop(homo_g)
            h = homo_g.ndata.pop('h')
            for i, layer in enumerate(self.layers):
                if i != 0:
                    h = self.dropout(h)
                h = layer(homo_g, h)
            # homo_g.ndata['out_h'] = h
            # out_h = dgl.to_heterogeneous(homo_g, hg.ntypes, hg.etypes).ndata['out_h']
            out_h = self.h2dict(h, hg.ndata['h'])
        return out_h

    def h2dict(self, h, hdict):
        pre = 0
        for i, value in hdict.items():
            hdict[i] = h[pre:value.shape[0]+pre]
            pre += value.shape[0]
        return hdict