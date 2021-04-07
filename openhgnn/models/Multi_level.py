import torch.nn as nn


class Base_model(nn.Module):
    def __init__(self):
        super(Base_model, self).__init__()

    def Micro_layer(self, h_dict):
        return h_dict

    def Macro_layer(self, h_dict):
        return h_dict

    def forward(self, h_dict):
        h_dict = self.Micro_layer(h_dict)
        h_dict = self.Macro_layer(h_dict)

        return h_dict


class Multi_level(nn.Module):
    def __init__(self):
        super(Multi_level, self).__init__()
        self.micro_layer = None
        self.macro_layer = None

    def forward(self):
        return


import dgl.nn.pytorch as dglnn
conv = dglnn.HeteroGraphConv({
    'follows' : dglnn.GraphConv(...),
    'plays' : dglnn.GraphConv(...),
    'sells' : dglnn.SAGEConv(...)},
    aggregate='sum')


from openhgnn.models.micro_layer.LSTM_conv import LSTMConv
class HGConvLayer(nn.Module):
    def __init__(self, graph: dgl.DGLHeteroGraph, input_dim: int, hidden_dim: int, n_heads: int = 4,
                 dropout: float = 0.2, residual: bool = True):
        """
        :param graph: a heterogeneous graph
        :param input_dim: int, input dimension
        :param hidden_dim: int, hidden dimension
        :param n_heads: int, number of attention heads
        :param dropout: float, dropout rate
        :param residual: boolean, residual connections or not
        """
        super(HGConvLayer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.dropout = dropout
        self.residual = residual

        # hetero conv modules
        self.micro_conv = dglnn.HeteroGraphConv({
            etype: LSTMConv(dim=input_dim)
            for srctype, etype, dsttype in graph.canonical_etypes
        })

        # different types aggregation module
        self.macro_conv = MacroConv(in_feats=hidden_dim * n_heads, out_feats=hidden_dim,
                                                             num_heads=n_heads,
                                                             dropout=dropout, negative_slope=0.2)

        if self.residual:
            # residual connection
            self.res_fc = nn.ModuleDict()
            self.residual_weight = nn.ParameterDict()
            for ntype in graph.ntypes:
                self.res_fc[ntype] = nn.Linear(input_dim, n_heads * hidden_dim, bias=True)
                self.residual_weight[ntype] = nn.Parameter(torch.randn(1))

        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        for weight in self.node_transformation_weight:
            nn.init.xavier_normal_(self.node_transformation_weight[weight], gain=gain)
        for weight in self.nodes_attention_weight:
            nn.init.xavier_normal_(self.nodes_attention_weight[weight], gain=gain)
        for weight in self.edge_type_transformation_weight:
            nn.init.xavier_normal_(self.edge_type_transformation_weight[weight], gain=gain)
        for weight in self.central_node_transformation_weight:
            nn.init.xavier_normal_(self.central_node_transformation_weight[weight], gain=gain)

        nn.init.xavier_normal_(self.edge_types_attention_weight, gain=gain)
        if self.residual:
            for ntype in self.res_fc:
                nn.init.xavier_normal_(self.res_fc[ntype].weight, gain=gain)

    def forward(self, graph: dgl.DGLHeteroGraph, node_features: dict):
        """
        :param graph: dgl.DGLHeteroGraph
        :param node_features: dict, {"type": features}
        :return: output_features: dict, {"type": features}
        """
        # dictionary of input source features and destination features
        input_src = node_features
        if graph.is_block:
            input_dst = {}
            for ntype in node_features:
                input_dst[ntype] = node_features[ntype][:graph.number_of_dst_nodes(ntype)]
        else:
            input_dst = node_features
        # same type neighbors aggregation
        # relation_features, dict, {(stype, etype, dtype): features}
        relation_features = self.micro_conv(graph, input_src, input_dst, self.node_transformation_weight,
                                             self.nodes_attention_weight)
        # different types aggregation
        output_features = self.macro_conv(graph, input_dst, relation_features,
                                                 self.edge_type_transformation_weight,
                                                 self.central_node_transformation_weight,
                                                 self.edge_types_attention_weight)

        if self.residual:
            for ntype in output_features:
                alpha = F.sigmoid(self.residual_weight[ntype])
                output_features[ntype] = output_features[ntype] * alpha + self.res_fc[ntype](input_dst[ntype]) * (1 - alpha)

        return output_features