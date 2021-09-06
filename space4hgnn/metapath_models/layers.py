import torch.nn as nn

import dgl
from openhgnn.models.MetapathConv import MetapathConv
from openhgnn.models.macro_layer.SemanticConv import SemanticAttention


class MPConv(nn.Module):
    def __init__(self, name, dim_in, dim_out, bias=False, **kwargs):
        super(MPConv, self).__init__()
        macro_func = kwargs['macro_func']
        meta_paths = kwargs['meta_paths']
        if macro_func == 'attention':
            macro_func = SemanticAttention(dim_out)
        elif macro_func == 'sum':
            macro_func = Aggr_sum
        elif macro_func == 'mean':
            macro_func = Aggr_mean
        elif macro_func == 'max':
            macro_func = Aggr_max
        self.model = MetapathConv(
            meta_paths,
            [layer_dict[name](dim_in, dim_out, num_heads=1, bias=bias)
            #[dgl.nn.pytorch.GraphConv(dim_in, dim_out, bias=bias)
            for _ in meta_paths],
            macro_func
        )
        self.meta_paths = meta_paths

    def forward(self, mp_g_list, h):
        h = self.model(mp_g_list, h)
        return h


def Aggr_sum(z):
    return z.sum(1)


def Aggr_max(z):
    return z.max(1)


def Aggr_mean(z):
    return z.mean(1)

layer_dict = {
    'gcnconv': dgl.nn.pytorch.GraphConv,
    'sageconv': dgl.nn.pytorch.SAGEConv,
    'gatconv': dgl.nn.pytorch.GATConv,
    'ginconv': dgl.nn.pytorch.GINConv,
    'appnpconv': dgl.nn.pytorch.APPNPConv
}
