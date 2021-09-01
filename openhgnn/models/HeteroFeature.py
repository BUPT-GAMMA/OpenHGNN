import torch as th
import torch.nn as nn
from .layers import HeteroLinearLayer


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
    1.

    Parameters
    ----------
    h_dict: dict
        Input heterogeneous feature dict,
        key of dict means node type,
        value of dict means corresponding feature of the node type.
        It can be None if the dataset has no feature.
    n_dict: dict
        Key of dict means node type,
        value of dict means number of nodes.
    embed_size: int
        Dimension of embedding, and used to assign to the output dimension of Linear which transform the original feature.
    need_trans: bool, optional
        A flag to control whether to transform original feature linearly. Default is ``True``.

    """
    def __init__(self, h_dict, n_dict, embed_size, need_trans=True, all_feats=True):
        """

        @param h_dict:
        @param n_dict:
        @param embed_size:
        @param need_trans:
        @param all_feats:
        """

        super(HeteroFeature, self).__init__()
        self.n_dict = n_dict
        self.embed_size = embed_size
        self.h_dict = h_dict
        self.need_trans = need_trans
        self.embed_dict = nn.ParameterDict()
        linear_dict = {}
        for ntype, n_nodes in n_dict.items():
            h = h_dict.get(ntype)
            if h is None:
                if all_feats:
                    embed = nn.Parameter(th.FloatTensor(n_nodes, self.embed_size))
                    # initrange = 1.0 / self.embed_size
                    # nn.init.uniform_(embed, -initrange, initrange)
                    nn.init.xavier_uniform_(embed, gain=nn.init.calculate_gain('relu'))
                    self.embed_dict[ntype] = embed
            else:
                linear_dict[ntype] = [h.shape[1], self.embed_size]
        if need_trans:
            self.hetero_linear = HeteroLinearLayer(linear_dict)

    def forward(self):
        h_dict = {}
        for ntype, _ in self.n_dict.items():
            if self.h_dict.get(ntype) is None:
                h_dict[ntype] = self.embed_dict[ntype]
        if self.need_trans:
            h_dict.update(self.hetero_linear(self.h_dict))
        else:
            h_dict.update(self.h_dict)
        return h_dict
