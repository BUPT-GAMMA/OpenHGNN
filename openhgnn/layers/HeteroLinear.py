import torch
import torch.nn as nn
import torch.nn.functional as F


class GeneralLinear(nn.Module):
    r"""
    Description
    ------------
    General Linear, combined with activation, normalization(batch and L2), dropout and so on.

    Parameters
    ------------
    in_features : int
        size of each input sample, which is fed into nn.Linear
    out_features : int
        size of each output sample, which is fed into nn.Linear
    act : callable activation function/layer or None, optional
        If not None, applies an activation function to the updated node features.
        Default: ``None``.
    dropout : float, optional
        Dropout rate. Default: ``0.0``
    has_l2norm : bool
        If True, applies torch.nn.functional.normalize to the node features at last of forward(). Default: ``True``
    has_bn : bool
        If True, applies torch.nn.BatchNorm1d to the node features after applying nn.Linear.

    """

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

    def forward(self, batch_h: torch.Tensor) -> torch.Tensor:
        r"""
        Description
        ------------
        Apply Linear, BatchNorm1d, Dropout and normalize(if need).
        """
        batch_h = self.layer(batch_h)
        batch_h = self.post_layer(batch_h)
        if self.has_l2norm:
            batch_h = F.normalize(batch_h, p=2, dim=1)
        return batch_h


class HeteroLinearLayer(nn.Module):
    r"""
    Description
    ------------
    Transform feature with nn.Linear. In general, heterogeneous feature has different dimension as input.
    Even though they may have same dimension, they may have different semantic in every dimension.
    So we use a linear layer for each node type to map all node features to a shared feature space.

    Parameters
    ----------
    linear_dict : dict
        Key of dict can be node type(node name), value of dict is a list contains input dimension and output dimension.

    Examples
    ----------

    >>>import torch as th
    >>>linear_dict = {}
    >>>linear_dict['author'] = [110, 64]
    >>>linear_dict['paper'] = [128,64]
    >>>h_dict = {}
    >>>h_dict['author'] = th.tensor(10, 110)
    >>>h_dict['paper'] = th.tensor(5, 128)
    >>>layer = HeteroLinearLayer(linear_dict)
    >>>out_dict = layer(h_dict)

    """
    def __init__(self, linear_dict, act=None, dropout=0.0, has_l2norm=True, has_bn=True, **kwargs):
        super(HeteroLinearLayer, self).__init__()

        self.layer = nn.ModuleDict({})
        for name, linear_dim in linear_dict.items():
            self.layer[name] = GeneralLinear(in_features=linear_dim[0], out_features=linear_dim[1], act=act,
                                                  dropot=dropout, has_l2norm=has_l2norm, has_bn=has_bn)

    def forward(self, dict_h: dict) -> dict:
        r"""
        Parameters
        ----------
        dict_h : dict
            A dict of heterogeneous feature

        return dict_h
        """
        # note must set new_h dict, or overwrite dict_h
        new_h = {}
        if isinstance(dict_h, dict):
            for name, batch_h in dict_h.items():
                new_h[name] = self.layer[name](batch_h)
        return new_h


class HeteroMLPLayer(nn.Module):
    r"""
    Description
    -------------
    HeteroMLPLayer contains multiple GeneralLinears, different with HeteroLinearLayer.
    The latter contains only one layer.

    Parameters
    ----------
    linear_dict : dict
        Key of dict can be node type(node name), value of dict is a list contains input, hidden and output dimension.

    """
    def __init__(self, linear_dict, act=None, dropout=0.0, has_l2norm=True, has_bn=True, final_act=False, **kwargs):
        super(HeteroMLPLayer, self).__init__()
        self.layers = nn.ModuleDict({})
        for name, linear_dim in linear_dict.items():
            nn_list = []
            n_layer = len(linear_dim) - 1
            for i in range(n_layer):
                in_dim = linear_dim[i]
                out_dim = linear_dim[i+1]
                if i == n_layer - 1:
                    if final_act:
                        layer = GeneralLinear(in_features=in_dim, out_features=out_dim, act=act,
                                              dropot=dropout, has_l2norm=has_l2norm, has_bn=has_bn)
                    else:
                        layer = GeneralLinear(in_features=in_dim, out_features=out_dim, act=None,
                                          dropot=dropout, has_l2norm=has_l2norm, has_bn=has_bn)
                else:
                    layer = GeneralLinear(in_features=in_dim, out_features=out_dim, act=act,
                                      dropot=dropout, has_l2norm=has_l2norm, has_bn=has_bn)

                nn_list.append(layer)
            self.layers[name] = nn.Sequential(*nn_list)

    def forward(self, dict_h):
        new_h = {}
        if isinstance(dict_h, dict):
            for name, batch_h in dict_h.items():
                new_h[name] = self.layers[name](batch_h)
        return new_h


class HeteroFeature(nn.Module):
    r"""
    Description
    ------------
    This is a feature preprocessing component which is dealt with various heterogeneous feature situation.

    In general, we will face the following three situations.

        1. The dataset has not feature at all.

        2. The dataset has features in every node type.

        3. The dataset has features of a part of node types.

    To deal with that, we implement the HeteroFeature.In every situation, we can see that

        1. We will build embeddings for all node types.

        2. We will build linear layer for all node types.

        3. We will build embeddings for parts of node types and linear layer for parts of node types which have original feature.

    Parameters
    ----------
    h_dict: dict
        Input heterogeneous feature dict,
        key of dict means node type,
        value of dict means corresponding feature of the node type.
        It can be None if the dataset has no feature.
    n_nodes_dict: dict
        Key of dict means node type,
        value of dict means number of nodes.
    embed_size: int
        Dimension of embedding, and used to assign to the output dimension of Linear which transform the original feature.
    need_trans: bool, optional
        A flag to control whether to transform original feature linearly. Default is ``True``.

    Attributes
    -----------
    embed_dict : nn.ParameterDict
        store the embeddings

    hetero_linear : HeteroLinearLayer
        A heterogeneous linear layer to transform original feature.
    """
    def __init__(self, h_dict, n_nodes_dict, embed_size, act=None, need_trans=True, all_feats=True):
        """

        @param h_dict:
        @param n_dict:
        @param embed_size:
        @param need_trans:
        @param all_feats:
        """

        super(HeteroFeature, self).__init__()
        self.n_nodes_dict = n_nodes_dict
        self.embed_size = embed_size
        self.h_dict = h_dict
        self.need_trans = need_trans
        self.embed_dict = nn.ParameterDict()
        linear_dict = {}
        for ntype, n_nodes in self.n_nodes_dict.items():
            h = h_dict.get(ntype)
            if h is None:
                if all_feats:
                    embed = nn.Parameter(torch.FloatTensor(n_nodes, self.embed_size))
                    # initrange = 1.0 / self.embed_size
                    # nn.init.uniform_(embed, -initrange, initrange)
                    nn.init.xavier_uniform_(embed, gain=nn.init.calculate_gain('relu'))
                    self.embed_dict[ntype] = embed
            else:
                linear_dict[ntype] = [h.shape[1], self.embed_size]
        if need_trans:
            self.hetero_linear = HeteroLinearLayer(linear_dict, act=act)

    def forward(self):
        r"""
        Description
        ------------
        return feature.

        Returns
        -------
        dict [str, th.Tensor]
            The output feature dictionary of feature.
        """
        out_dict = {}
        for ntype, _ in self.n_nodes_dict.items():
            if self.h_dict.get(ntype) is None:
                out_dict[ntype] = self.embed_dict[ntype]
        if self.need_trans:
            out_dict.update(self.hetero_linear(self.h_dict))
        else:
            out_dict.update(self.h_dict)
        return out_dict

    def forward_nodes(self, nodes_dict):
        out_feature = {}
        for ntype, nid in nodes_dict.items():
            if self.h_dict.get(ntype) is None:
                out_feature[ntype] = self.embed_dict[ntype][nid]
            else:
                if self.need_trans:
                    out_feature[ntype] = self.hetero_linear(self.h_dict)[ntype][nid]
                else:
                    out_feature[ntype] = self.h_dict[ntype][nid]
        return out_feature