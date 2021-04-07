import torch
from torch import nn
import torch.nn.functional as F

import dgl
from dgl.nn.pytorch.softmax import edge_softmax
import dgl.function as fn


class MicroConv(nn.Module):

    def __init__(self, in_feats: tuple, out_feats: int, num_heads: int, dropout: float = 0.0, negative_slope: float = 0.2):
        """
        Parameters
        ----------
        in_feats : pair of ints
            Input feature size.
        out_feats : int
            Output feature size.
        num_heads : int
            Number of heads in Multi-Head Attention.
        dropout : float, optional
            Dropout rate, defaults: 0.
        negative_slope : float, optional
            Negative slope rate, defaults: 0.2.
        """
        super(MicroConv, self).__init__()
        self._in_src_feats, self._in_dst_feats = in_feats[0], in_feats[1]
        self._out_feats = out_feats
        self._num_heads = num_heads

        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

    def forward(self, graph: dgl.DGLHeteroGraph, feat: tuple, dst_node_transformation_weight: nn.Parameter,
                src_node_transformation_weight: nn.Parameter, src_nodes_attention_weight: nn.Parameter):
        r"""Compute graph attention network layer.
        Parameters
        ----------
        graph : specific relational DGLHeteroGraph
        feat : pair of torch.Tensor
            The pair contains two tensors of shape (N_{in}, D_{in_{src}})` and (N_{out}, D_{in_{dst}}).
        dst_node_transformation_weight: Parameter (input_dst_dim, n_heads * hidden_dim)
        src_node_transformation_weight: Parameter (input_src_dim, n_heads * hidden_dim)
        src_nodes_attention_weight: Parameter (n_heads, 2 * hidden_dim)
        Returns
        -------
        torch.Tensor, shape (N, H, D_out)` where H is the number of heads, and D_out is size of output feature.
        """
        graph = graph.local_var()
        # Tensor, (N_src, input_src_dim)
        feat_src = self.dropout(feat[0])
        # Tensor, (N_dst, input_dst_dim)
        feat_dst = self.dropout(feat[1])
        # Tensor, (N_src, n_heads, hidden_dim) -> (N_src, input_src_dim) * (input_src_dim, n_heads * hidden_dim)
        feat_src = torch.matmul(feat_src, src_node_transformation_weight).view(-1, self._num_heads, self._out_feats)
        # Tensor, (N_dst, n_heads, hidden_dim) -> (N_dst, input_dst_dim) * (input_dst_dim, n_heads * hidden_dim)
        feat_dst = torch.matmul(feat_dst, dst_node_transformation_weight).view(-1, self._num_heads, self._out_feats)

        # first decompose the weight vector into [a_l || a_r], then
        # a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j, This implementation is much efficient
        # Tensor, (N_dst, n_heads, 1),   (N_dst, n_heads, hidden_dim) * (n_heads, hidden_dim)
        e_dst = (feat_dst * src_nodes_attention_weight[:, :self._out_feats]).sum(dim=-1, keepdim=True)
        # Tensor, (N_src, n_heads, 1),   (N_src, n_heads, hidden_dim) * (n_heads, hidden_dim)
        e_src = (feat_src * src_nodes_attention_weight[:, self._out_feats:]).sum(dim=-1, keepdim=True)
        # (N_src, n_heads, hidden_dim), (N_src, n_heads, 1)
        graph.srcdata.update({'ft': feat_src, 'e_src': e_src})
        # (N_dst, n_heads, 1)
        graph.dstdata.update({'e_dst': e_dst})
        # compute edge attention, e_src and e_dst are a_src * Wh_src and a_dst * Wh_dst respectively.
        graph.apply_edges(fn.u_add_v('e_src', 'e_dst', 'e'))
        # shape (edges_num, heads, 1)
        e = self.leaky_relu(graph.edata.pop('e'))

        # compute softmax
        graph.edata['a'] = edge_softmax(graph, e)

        graph.update_all(fn.u_mul_e('ft', 'a', 'msg'), fn.sum('msg', 'ft'))
        # (N_dst, n_heads * hidden_dim),   (N_dst, n_heads, hidden_dim) reshape
        dst_features = graph.dstdata.pop('ft').reshape(-1, self._num_heads * self._out_feats)

        dst_features = F.relu(dst_features)

        return dst_features



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

    def __init__(self, in_feats: int, out_feats: int, num_heads: int, dropout: float = 0.0, negative_slope: float = 0.2):
        super(MacroConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._num_heads = num_heads

        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

    def forward(self, graph, input_dst: dict, relation_features: dict, edge_type_transformation_weight: nn.ParameterDict,
                central_node_transformation_weight: nn.ParameterDict, edge_types_attention_weight: nn.Parameter):
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
                central_node_feature = torch.matmul(central_node_feature, central_node_transformation_weight[ntype]). \
                    view(-1, self._num_heads, self._out_feats)
                types_features = []
                for relation_tuple in relation_features:
                    stype, etype, dtype = relation_tuple
                    if dtype == ntype:
                        # (N_ntype, n_heads * hidden_dim)
                        types_features.append(torch.matmul(relation_features[relation_tuple],
                                                           edge_type_transformation_weight[etype]))
                        # TODO: another aggregation equation
                        # relation_features[relation_tuple] -> (N_ntype, n_heads * hidden_dim), (N_ntype, n_heads, hidden_dim)
                        # edge_type_transformation_weight -> (n_heads, hidden_dim, hidden_dim)
                        # each element -> (N_ntype, n_heads * hidden_dim)
                        # types_features.append(torch.einsum('abc,bcd->abd', relation_features[relation_tuple].reshape(-1, self._num_heads, self._out_feats),
                        #                                    edge_type_transformation_weight[etype]).flatten(start_dim=1))
                # Tensor, (relations_num, N_ntype, n_heads * hidden_dim)
                types_features = torch.stack(types_features, dim=0)
                # if the central node only interacts with one relation, then the attention score is 1,
                # directly assgin the transformed feature to the central node
                if types_features.shape[0] == 1:
                    output_features[ntype] = types_features.squeeze(dim=0)
                else:
                    # Tensor, (relations_num, N_ntype, n_heads, hidden_dim)
                    types_features = types_features.view(types_features.shape[0], -1, self._num_heads, self._out_feats)
                    # (relations_num, N_ntype, n_heads, hidden_dim)
                    stacked_central_features = torch.stack([central_node_feature for _ in range(types_features.shape[0])],
                                                           dim=0)
                    # (relations_num, N_ntype, n_heads, 2 * hidden_dim)
                    concat_features = torch.cat((stacked_central_features, types_features), dim=-1)
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