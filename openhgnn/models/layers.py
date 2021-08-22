import torch
import torch.nn as nn
import torch.nn.functional as F


class GeneralLinear(nn.Module):
    '''General Linear'''

    def __init__(self, in_features, out_features, act=None, dropout=0.0, has_l2norm=True, has_bn=True, **kwargs):
        super(GeneralLinear, self).__init__()
        self.has_l2norm = has_l2norm
        has_bn = has_bn
        self.layer = nn.Linear(in_features, out_features, bias=not has_bn)
        layer_wrapper = []
        if has_bn:
            layer_wrapper.append(nn.BatchNorm1d(out_features))
        if dropout > 0:
            layer_wrapper.append(nn.Dropout(p=dropout))
        if act is not None:
            layer_wrapper.append(act)
        self.post_layer = nn.Sequential(*layer_wrapper)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.layer.weight)

    def forward(self, batch_h):
        batch_h = self.layer(batch_h)
        batch_h = self.post_layer(batch_h)
        if self.has_l2norm:
            batch_h = F.normalize(batch_h, p=2, dim=1)
        return batch_h


class HeteroLinearLayer(nn.Module):
    def __init__(self, linear_list, act=None, dropout=0.0, has_l2norm=True, has_bn=True):
        super(HeteroLinearLayer, self).__init__()

        self.layer = nn.ModuleDict({})
        for linear in linear_list:
            self.layer[linear[0]] = GeneralLinear(in_features=linear[1], out_features=linear[2], act=act,
                                                  dropot=dropout, has_l2norm=has_l2norm, has_bn=has_bn)

    def forward(self, dict_h):
        # note must set new_h dict, or overwrite dict_h
        new_h = {}
        if isinstance(dict_h, dict):
            for name, batch_h in dict_h.items():
                new_h[name] = self.layer[name](batch_h)
        return new_h


class HeteroMLPLayer(nn.Module):
    def __init__(self, linear_list, num_layers=1, act=None, dropout=0.0, has_l2norm=True, has_bn=True, **kwargs):
        super(HeteroMLPLayer, self).__init__()
        self.layers = nn.ModuleDict({})
        for linear in linear_list:
            name = linear[0]
            in_dim = linear[1]
            out_dim = linear[2]
            inner_dim = out_dim
            nn_list = []
            for i in range(num_layers):
                in_d = in_dim if i == 0 else inner_dim
                out_d = out_dim if i == num_layers - 1 else inner_dim
                layer = GeneralLinear(in_features=in_d, out_features=out_d, act=act,
                                      dropot=dropout, has_l2norm=has_l2norm, has_bn=has_bn)

                nn_list.append(layer)
            self.layers[name] = nn.Sequential(*nn_list)

    def forward(self, dict_h):
        new_h = {}
        if isinstance(dict_h, dict):
            for name, batch_h in dict_h.items():
                new_h[name] = self.layers[name](batch_h)
        return new_h
