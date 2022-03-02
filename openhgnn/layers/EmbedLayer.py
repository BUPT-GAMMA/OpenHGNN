import torch as th
import torch.nn as nn
import torch.nn.functional as F


class HeteroEmbedLayer(nn.Module):
    r"""
    Embedding layer for featureless heterograph.

    Parameters
    -----------
    n_nodes_dict : dict[str, int]
        Key of dict means node type,
        value of dict means number of nodes.
    embed_size : int
        Dimension of embedding,
    activation : callable activation function/layer or None, optional
        If not None, applies an activation function to the updated node features.
        Default: ``None``.
    dropout : float, optional
        Dropout rate. Default: ``0.0``
    """

    def __init__(self,
                 n_nodes_dict,
                 embed_size,
                 embed_name='embed',
                 activation=None,
                 dropout=0.0):
        super(HeteroEmbedLayer, self).__init__()

        self.embed_size = embed_size
        self.embed_name = embed_name
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

        # create weight embeddings for each node for each relation
        self.embeds = nn.ParameterDict()
        for ntype, nodes in n_nodes_dict.items():
            embed = nn.Parameter(th.FloatTensor(nodes, self.embed_size))
            # initrange = 1.0 / self.embed_size
            # nn.init.uniform_(embed, -initrange, initrange)
            nn.init.xavier_uniform_(embed, gain=nn.init.calculate_gain('relu'))
            self.embeds[ntype] = embed

    def forward(self, ):
        r"""
        Returns
        -------
        The output embeddings.
        """
        out_feature = {}
        for key, embed in self.embeds.items():
            out_feature[key] = embed
        return out_feature

    def forward_nodes(self, nodes_dict):
        r"""

        Parameters
        ----------
        nodes_dict : dict[str, th.Tensor]
            Key of dict means node type, value of dict means idx of nodes.

        Returns
        -------
        out_feature : dict[str, th.Tensor]
            Output feature.
        """
        out_feature = {}
        for key, nid in nodes_dict.items():
            out_feature[key] = self.embeds[key][nid]
        return out_feature


class multi_Linear(nn.Module):
    def __init__(self, linear_list, bias=False):
        super(multi_Linear, self).__init__()
        self.encoder = nn.ModuleDict({})
        for linear in linear_list:
            self.encoder[linear[0]] = nn.Linear(in_features=linear[1], out_features=linear[2], bias=bias)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.encoder.weight)

    def forward(self, name_linear, h):
        h = self.encoder[name_linear](h)
        return h


class multi_2Linear(nn.Module):
    def __init__(self, linear_list, bias=False):
        super(multi_2Linear, self).__init__()
        hidden_dim = 16
        self.hidden_layer = nn.ModuleDict({})
        self.output_layer = nn.ModuleDict({})
        for linear in linear_list:
            self.hidden_layer[linear[0]] = nn.Linear(in_features=linear[1], out_features=hidden_dim, bias=bias)
            self.output_layer[linear[0]] = nn.Linear(in_features=hidden_dim, out_features=linear[2], bias=bias)

    def forward(self, name_linear, h):
        h = F.relu(self.hidden_layer[name_linear](h))
        h = self.output_layer[name_linear](h)
        return h


class hetero_linear(nn.Module):
    def __init__(self, linear_list, bias=False):
        super(hetero_linear, self).__init__()
        # In one graph, the node with different node type may have different dimension size as the input.
        # The feature_mapping NN feature the input dimension project to another same dimension size.
        # So the out_dim is just a scalar.

        # n_feats are graph dgl.ndata name.
        self.encoder = multi_Linear(linear_list, bias)

    def forward(self, h_dict):
        h_out = {}
        for ntype, h in h_dict.items():
            h = self.encoder(ntype, h)
            h_out[ntype] = h
        return h_out