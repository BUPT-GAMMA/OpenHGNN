import torch as th
from torch import nn
import torch.nn.functional as F

import dgl
from dgl.nn.pytorch.softmax import edge_softmax
import dgl.function as fn

class ATTConv(nn.Module):
    '''
    It is macro_layer of the models [HetGNN].
    It presents in the 3.3.2 Types Combination of the paper.

    Parameters
    ----------
    dim : int
        Input feature dimension.
    ntypes : list
        Node types.


    ''
    In this framework, to make embedding dimension consistent and models tuning easy,
        we use the same dimension d for content embedding in Section 3.2,
        aggregated content embedding in Section 3.3, and output node embedding in Section 3.3.
        ''
    So just give one dim parameter.

    Note:
        We don't implement multi-heads version.

        atten_w is specific to the center node type, agnostic to the neighbor node type.
    '''
    def __init__(self, ntypes, dim):
        super(ATTConv, self).__init__()
        self.ntypes = ntypes
        self.activation = nn.LeakyReLU()
        self.atten_w = nn.ModuleDict({})
        for n in self.ntypes:
            self.atten_w[n] = nn.Linear(in_features=dim * 2, out_features=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, hg, h_neigh, h_center):
        with hg.local_scope():
            if hg.is_block:
                h_dst = {k: v[:hg.number_of_dst_nodes(k)] for k, v in h_center.items()}
            else:
                h_dst = h_center
            # n_types is the number of embedding need to be aggregate
            n_types = len(self.ntypes) + 1
            outputs = {}
            for n in self.ntypes:
                h = h_dst[n]
                batch_size = h.shape[0]

                concat_h = []
                concat_emd = []
                for i in range(len(h_neigh[n])):
                    concat_h.append(th.cat((h, h_neigh[n][i]), 1))
                    concat_emd.append(h_neigh[n][i])
                concat_h.append(th.cat((h, h), 1))
                concat_emd.append(h)

                # compute weights
                concat_h = th.hstack(concat_h).view(batch_size * n_types, self.dim * 2)
                atten_w = self.activation(self.atten_w[n](concat_h)).view(batch_size, n_types)
                atten_w = self.softmax(atten_w).view(batch_size, 1, 4)

                # weighted combination
                concat_emd = th.hstack(concat_emd).view(batch_size, n_types, self.dim)

                weight_agg_batch = th.bmm(atten_w, concat_emd).view(batch_size, self.dim)
                outputs[n] = weight_agg_batch
            return outputs


class MacroConv(nn.Module):
    """
    Parameters
    ----------
    in_feats : int
        Input feature size.
    out_feats : int
        Output feature size.
    num_heads : int
        Number of heads in Multi-Head Attention.
    dropout : float, optional
        Dropout rate, defaults: ``0``.
    """

    def __init__(self, in_feats: int,
                 out_feats: int,
                 num_heads: int,
                 dropout: float = 0.0,
                 negative_slope: float = 0.2):
        super(MacroConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._num_heads = num_heads

        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

    def forward(self,
                graph,
                input_dst: dict,
                relation_features: dict,
                edge_type_transformation_weight: nn.ParameterDict,
                central_node_transformation_weight: nn.ParameterDict,
                edge_types_attention_weight: nn.Parameter):
        """
        :param graph: dgl.DGLHeteroGraph
        :param input_dst: dict: {ntype: features}
        :param relation_features: dict: {(stype, etype, dtype): features}
        :param edge_type_transformation_weight: ParameterDict {etype: (n_heads * hidden_dim, n_heads * hidden_dim)}
        :param central_node_transformation_weight:  ParameterDict {ntype: (input_central_node_dim, n_heads * hidden_dim)}
        :param edge_types_attention_weight: Parameter (n_heads, 2 * hidden_dim)
        :return: output_features: dict, {"type": features}
        """
        output_features = {}
        for ntype in input_dst:
            if graph.number_of_dst_nodes(ntype) != 0:
                # (N_ntype, self._in_feats)
                central_node_feature = input_dst[ntype]
                # (N_ntype, n_heads, hidden_dim)
                central_node_feature = th.matmul(central_node_feature, central_node_transformation_weight[ntype]). \
                    view(-1, self._num_heads, self._out_feats)
                types_features = []
                for relation_tuple in relation_features:
                    stype, etype, dtype = relation_tuple
                    if dtype == ntype:
                        # (N_ntype, n_heads * hidden_dim)
                        types_features.append(th.matmul(relation_features[relation_tuple],
                                                           edge_type_transformation_weight[etype]))
                        # TODO: another aggregation equation
                        # relation_features[relation_tuple] -> (N_ntype, n_heads * hidden_dim), (N_ntype, n_heads, hidden_dim)
                        # edge_type_transformation_weight -> (n_heads, hidden_dim, hidden_dim)
                        # each element -> (N_ntype, n_heads * hidden_dim)
                        # types_features.append(torch.einsum('abc,bcd->abd', relation_features[relation_tuple].reshape(-1, self._num_heads, self._out_feats),
                        #                                    edge_type_transformation_weight[etype]).flatten(start_dim=1))
                # Tensor, (relations_num, N_ntype, n_heads * hidden_dim)
                types_features = th.stack(types_features, dim=0)
                # if the central node only interacts with one relation, then the attention score is 1,
                # directly assgin the transformed feature to the central node
                if types_features.shape[0] == 1:
                    output_features[ntype] = types_features.squeeze(dim=0)
                else:
                    # Tensor, (relations_num, N_ntype, n_heads, hidden_dim)
                    types_features = types_features.view(types_features.shape[0], -1, self._num_heads, self._out_feats)
                    # (relations_num, N_ntype, n_heads, hidden_dim)
                    stacked_central_features = th.stack([central_node_feature for _ in range(types_features.shape[0])],
                                                           dim=0)
                    # (relations_num, N_ntype, n_heads, 2 * hidden_dim)
                    concat_features = th.cat((stacked_central_features, types_features), dim=-1)
                    # (relations_num, N_ntype, n_heads, 1) -> (n_heads, 2 * hidden_dim) * (relations_num, N_ntype, n_heads, 2 * hidden_dim)
                    attention_scores = (edge_types_attention_weight * concat_features).sum(dim=-1, keepdim=True)
                    attention_scores = self.leaky_relu(attention_scores)
                    attention_scores = F.softmax(attention_scores, dim=0)
                    # (N_ntype, n_heads, hidden_dim)
                    output_feature = (attention_scores * types_features).sum(dim=0)
                    output_feature = self.dropout(output_feature)
                    output_feature = output_feature.reshape(-1, self._num_heads * self._out_feats)
                    output_features[ntype] = output_feature

        return output_features