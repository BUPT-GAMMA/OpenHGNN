import dgl
import torch as th
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from . import BaseModel, register_model
import tqdm
import torch

from dgl.ops import edge_softmax
import dgl.function as fn


@register_model('RHGNN')
class RHGNN(BaseModel):
    r"""
    This is the main method of model RHGNN

    Parameters
    ----------
    graph: dgl.DGLHeteroGraph
        a heterogeneous graph
    input_dim_dict: dict
        node input dimension dictionary
    hidden_dim: int
        node hidden dimension
    relation_input_dim: int
        relation input dimension
    relation_hidden_dim: int
        relation hidden dimension
    num_layers: int
        number of stacked layers
    n_heads: int
        number of attention heads
    dropout: float
        dropout rate
    negative_slope: float
        negative slope
    residual: boolean
        residual connections or not
    norm: boolean
        layer normalization or not
    """
    @classmethod
    def build_model_from_args(cls, args, hg):
        input_dim_dict = {ntype: hg.nodes[ntype].data['h'].shape[1] for ntype in hg.ntypes}
        return cls(graph=hg, input_dim_dict=input_dim_dict, hidden_dim=args.hidden_dim,
                   relation_input_dim=args.relation_hidden_units,
                   relation_hidden_dim=args.relation_hidden_units,
                   num_layers=args.num_layers,category=args.category,
                   out_dim=args.out_dim
                   )

    def __init__(self, graph: dgl.DGLHeteroGraph, input_dim_dict, hidden_dim: int,
                     relation_input_dim: int,
                     relation_hidden_dim: int,
                     num_layers: int, category,
                     out_dim,
                     n_heads: int = 4,
                     dropout: float = 0.2, negative_slope: float = 0.2,
                     residual: bool = True, norm: bool = True):

        super(RHGNN, self).__init__()

        self.category = category
        self.input_dim_dict = input_dim_dict
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.relation_input_dim = relation_input_dim
        self.relation_hidden_dim = relation_input_dim
        self.n_heads = n_heads
        self.dropout = dropout
        self.negative_slope = negative_slope
        self.residual = residual
        self.out_dim = out_dim
        self.norm = norm

        print(graph.etypes)
        # relation embedding dictionary
        self.relation_embedding = nn.ParameterDict({
            etype: nn.Parameter(torch.randn(relation_input_dim, 1)) for etype in graph.etypes
        })

        # align the dimension of different types of nodes
        self.projection_layer = nn.ModuleDict({
            ntype: nn.Linear(self.input_dim_dict[ntype], hidden_dim * n_heads) for ntype in input_dim_dict
        })

        # each layer takes in the heterogeneous graph as input
        self.layers = nn.ModuleList()

        # for each relation_layer
        self.layers.append(
            R_HGNN_Layer(graph, hidden_dim * n_heads, hidden_dim, relation_input_dim, relation_hidden_dim, n_heads,
                         dropout, negative_slope, residual, norm))
        for _ in range(1, self.num_layers):
            self.layers.append(R_HGNN_Layer(graph, hidden_dim * n_heads, hidden_dim, relation_hidden_dim * n_heads,
                                            relation_hidden_dim, n_heads, dropout, negative_slope, residual, norm))

        # transformation matrix for target node representation under each relation
        self.node_transformation_weight = nn.ParameterDict({
            etype: nn.Parameter(torch.randn(n_heads, hidden_dim, hidden_dim)) for etype in graph.etypes
        })

        # transformation matrix for relation representation
        self.relation_transformation_weight = nn.ParameterDict({
            etype: nn.Parameter(torch.randn(n_heads, relation_hidden_dim, hidden_dim)) for etype in graph.etypes
        })

        # different relations fusing module
        self.relation_fusing = RelationFusing(node_hidden_dim=hidden_dim,
                                              relation_hidden_dim=relation_hidden_dim,
                                              num_heads=n_heads,
                                              dropout=dropout, negative_slope=negative_slope)
        self.classifier = nn.Linear(self.hidden_dim * self.n_heads, self.out_dim)  #### todo

        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')

        for etype in self.relation_embedding:
            nn.init.xavier_normal_(self.relation_embedding[etype], gain=gain)
        for ntype in self.projection_layer:
            nn.init.xavier_normal_(self.projection_layer[ntype].weight, gain=gain)
        for etype in self.node_transformation_weight:
            nn.init.xavier_normal_(self.node_transformation_weight[etype], gain=gain)
        for etype in self.relation_transformation_weight:
            nn.init.xavier_normal_(self.relation_transformation_weight[etype], gain=gain)

    def forward(self, blocks: list, relation_target_node_features=None, relation_embedding: dict = None):


        r"""

        Parameters
        ----------
        blocks: list
            list of sampled dgl.DGLHeteroGraph
        relation_target_node_features: dict
            target node features under each relation, e.g {(srctype, etype, dsttype): features}
        relation_embedding: dict
            embedding for each relation, e.g {etype: feature} or None

        """
        relation_target_node_features = {}
        for stype, etype, dtype in blocks[0].canonical_etypes:
            relation_target_node_features[(stype, etype, dtype)] = blocks[0].srcnodes[dtype].data.get('h').to(torch.float32)

        # target relation feature projection
        for stype, reltype, dtype in relation_target_node_features:
            relation_target_node_features[(stype, reltype, dtype)] = self.projection_layer[dtype](
                relation_target_node_features[(stype, reltype, dtype)])

        # each relation is associated with a specific type, if no semantic information is given,
        # then the one-hot representation of each relation is assign with trainable hidden representation
        if relation_embedding is None:
            relation_embedding = {}
            for etype in self.relation_embedding:
                relation_embedding[etype] = self.relation_embedding[etype].flatten()

        # graph convolution
        for block, layer in zip(blocks, self.layers):
            relation_target_node_features, relation_embedding = layer(block, relation_target_node_features,
                                                                      relation_embedding)


        relation_fusion_embedding_dict = {}
        # relation_target_node_features -> {(srctype, etype, dsttype): target_node_features}
        for dsttype in set([dtype for _, _, dtype in relation_target_node_features]):
            relation_target_node_features_dict = {etype: relation_target_node_features[(stype, etype, dtype)]
                                                  for stype, etype, dtype in relation_target_node_features}
            etypes = [etype for stype, etype, dtype in relation_target_node_features if dtype == dsttype]
            dst_node_features = [relation_target_node_features_dict[etype] for etype in etypes]
            dst_relation_embeddings = [relation_embedding[etype] for etype in etypes]
            dst_node_feature_transformation_weight = [self.node_transformation_weight[etype] for etype in etypes]
            dst_relation_embedding_transformation_weight = [self.relation_transformation_weight[etype] for etype in etypes]

            # Tensor, shape (heads_num * hidden_dim)
            dst_node_relation_fusion_feature = self.relation_fusing(dst_node_features,
                                                                    dst_relation_embeddings,
                                                                    dst_node_feature_transformation_weight,
                                                                    dst_relation_embedding_transformation_weight)

            relation_fusion_embedding_dict[dsttype] = dst_node_relation_fusion_feature

        # relation_fusion_embedding_dict, {ntype: tensor -> (nodes, n_heads * hidden_dim)}
        # relation_target_node_features, {(srctype, etype, dsttype): (dst_nodes, n_heads * hidden_dim)}
        classifier_result = self.classifier(relation_fusion_embedding_dict[self.category])
        # return relation_fusion_embedding_dict, relation_target_node_features
        return {self.category: classifier_result}

    def inference(self, graph: dgl.DGLHeteroGraph, relation_target_node_features: dict, relation_embedding: dict = None,
                  device: str = 'cuda:0'):
        r"""
        mini-batch inference of final representation over all node types. Outer loop: Interate the layers, Inner loop: Interate the batches

        Parameters
        ----------
        graph: dgl.DGLHeteroGraph
            The whole relational graphs
        relation_target_node_features:  dict
            target node features under each relation, e.g {(srctype, etype, dsttype): features}
        relation_embedding: dict
            embedding for each relation, e.g {etype: feature} or None
        device: str
            device

        """

        with torch.no_grad():

            if relation_embedding is None:
                relation_embedding = {}
                for etype in self.relation_embedding:
                    relation_embedding[etype] = self.relation_embedding[etype].flatten()

            # interate over each layer
            for index, layer in enumerate(self.layers):
                # Tensor, features of all relation embeddings of the target nodes, store on cpu
                y = {
                    (stype, etype, dtype): torch.zeros(graph.number_of_nodes(dtype), self.hidden_dim * self.n_heads) for
                    stype, etype, dtype in graph.canonical_etypes}

                # full sample for each type of nodes
                sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
                dataloader = dgl.dataloading.NodeDataLoader(
                    graph,
                    {ntype: torch.arange(graph.number_of_nodes(ntype)) for ntype in graph.ntypes},
                    sampler,
                    batch_size=1280,
                    shuffle=True,
                    drop_last=False,
                    num_workers=4)

                tqdm_dataloader = tqdm(dataloader, ncols=120)
                for batch, (input_nodes, output_nodes, blocks) in enumerate(tqdm_dataloader):
                    block = blocks[0].to(device)

                    # for relational graphs that only contain a single type of nodes, construct the input and output node dictionary
                    if len(set(blocks[0].ntypes)) == 1:
                        input_nodes = {blocks[0].ntypes[0]: input_nodes}
                        output_nodes = {blocks[0].ntypes[0]: output_nodes}

                    input_features = {(stype, etype, dtype): relation_target_node_features[(stype, etype, dtype)][
                        input_nodes[dtype]].to(device)
                                      for stype, etype, dtype in relation_target_node_features.keys()}

                    input_relation_features = relation_embedding

                    if index == 0:
                        # target relation feature projection for the first layer in the full batch inference
                        for stype, reltype, dtype in input_features:
                            input_features[(stype, reltype, dtype)] = self.projection_layer[dtype](
                                input_features[(stype, reltype, dtype)])
                    h, input_relation_features = layer(block, input_features, input_relation_features)
                    for stype, reltype, dtype in h.keys():
                        y[(stype, reltype, dtype)][output_nodes[dtype]] = h[(stype, reltype, dtype)].cpu()

                    tqdm_dataloader.set_description(f'inference for the {batch}-th batch in model {index}-th layer')

                # update the features of all the nodes (after the graph convolution) in the whole graph
                relation_target_node_features = y
                # relation embedding is updated after each layer
                relation_embedding = input_relation_features

            for stype, etype, dtype in relation_target_node_features:
                relation_target_node_features[(stype, etype, dtype)] = relation_target_node_features[
                    (stype, etype, dtype)].to(device)

            relation_fusion_embedding_dict = {}
            # relation_target_node_features -> {(srctype, etype, dsttype): target_node_features}
            for dsttype in set([dtype for _, _, dtype in relation_target_node_features]):

                relation_target_node_features_dict = {etype: relation_target_node_features[(stype, etype, dtype)]
                                                      for stype, etype, dtype in relation_target_node_features}
                etypes = [etype for stype, etype, dtype in relation_target_node_features if dtype == dsttype]
                dst_node_features = [relation_target_node_features_dict[etype] for etype in etypes]
                dst_relation_embeddings = [relation_embedding[etype] for etype in etypes]
                dst_node_feature_transformation_weight = [self.node_transformation_weight[etype] for etype in etypes]
                dst_relation_embedding_transformation_weight = [self.relation_transformation_weight[etype] for etype in etypes]

                # use mini-batch to avoid out of memory in inference
                relation_fusion_embedding = []
                index = 0
                batch_size = 2560
                while index < dst_node_features[0].shape[0]:
                    # Tensor, shape (heads_num * hidden_dim)
                    relation_fusion_embedding.append(self.relation_fusing(
                        [dst_node_feature[index: index + batch_size, :] for dst_node_feature in dst_node_features],
                        dst_relation_embeddings,
                        dst_node_feature_transformation_weight,
                        dst_relation_embedding_transformation_weight))
                    index += batch_size
                relation_fusion_embedding_dict[dsttype] = torch.cat(relation_fusion_embedding, dim=0)

            # relation_fusion_embedding_dict, {ntype: tensor -> (nodes, n_heads * hidden_dim)}
            # relation_target_node_features, {ntype: tensor -> (num_relations, nodes, n_heads * hidden_dim)}
            return relation_fusion_embedding_dict, relation_target_node_features


# hetetoConv
class HeteroGraphConv(nn.Module):
    r"""
    A generic module for computing convolution on heterogeneous graphs.

    The heterograph convolution applies sub-modules on their associating
    relation graphs, which reads the features from source nodes and writes the
    updated ones to destination nodes. If multiple relations have the same
    destination node types, their results are aggregated by the specified method.

    If the relation graph has no edge, the corresponding module will not be called.

    Parameters
    ----------
    mods : dict[str, nn.Module]
        Modules associated with every edge types.
    """


    def __init__(self, mods: dict):
        super(HeteroGraphConv, self).__init__()
        self.mods = nn.ModuleDict(mods)

    def forward(self, graph: dgl.DGLHeteroGraph, input_src: dict, input_dst: dict, relation_embedding: dict,
                node_transformation_weight: nn.ParameterDict, relation_transformation_weight: nn.ParameterDict):
        r"""
        call the forward function with each module.

        Parameters
        ----------
        graph: DGLHeteroGraph
            The Heterogeneous Graph.
        input_src: dict[tuple, Tensor]
            Input source node features {relation_type: features, }
        input_dst: dict[tuple, Tensor]
            Input destination node features {relation_type: features, }
        relation_embedding: dict[etype, Tensor]
            Input relation features {etype: feature}
        node_transformation_weight: nn.ParameterDict
            weights {ntype, (inp_dim, hidden_dim)}
        relation_transformation_weight: nn.ParameterDict
            weights {etype, (n_heads, 2 * hidden_dim)}

        Returns
        -------
        outputs: dict[tuple, Tensor]
            Output representations for every relation -> {(stype, etype, dtype): features}.
        """

        # find reverse relation dict
        reverse_relation_dict = {}
        for srctype, reltype, dsttype in list(input_src.keys()):
            for stype, etype, dtype in input_src:
                if stype == dsttype and dtype == srctype and etype != reltype:
                    reverse_relation_dict[reltype] = etype
                    break

        # dictionary, {(srctype, etype, dsttype): representations}
        outputs = dict()

        for stype, etype, dtype in graph.canonical_etypes:
            rel_graph = graph[stype, etype, dtype]
            if rel_graph.number_of_edges() == 0:
                continue
            # for example, (author, writes, paper) relation, take author as src_nodes, take paper as dst_nodes
            dst_representation = self.mods[etype](rel_graph,
                                                  (input_src[(dtype, reverse_relation_dict[etype], stype)],
                                                   input_dst[(stype, etype, dtype)]),
                                                  node_transformation_weight[dtype],
                                                  node_transformation_weight[stype],
                                                  relation_embedding[etype],
                                                  relation_transformation_weight[etype])

            # dst_representation (dst_nodes, hid_dim)
            outputs[(stype, etype, dtype)] = dst_representation

        return outputs


# relation crossing
class RelationCrossing(nn.Module):

    def __init__(self, in_feats: int, out_feats: int, num_heads: int, dropout: float = 0.0, negative_slope: float = 0.2):
        r"""
        Relation crossing layer

        Parameters
        ----------
        in_feats : pair of ints
            input feature size
        out_feats : int
            output feature size
        num_heads : int
            number of heads in Multi-Head Attention
        dropout : float
            optional, dropout rate, defaults: 0.0
        negative_slope : float
            optional, negative slope rate, defaults: 0.2
        """
        super(RelationCrossing, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._num_heads = num_heads

        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

    def forward(self, dsttype_node_features: torch.Tensor, relations_crossing_attention_weight: nn.Parameter):
        r"""
        Parameters
        ----------
        dsttype_node_features:
            a tensor of (dsttype_node_relations_num, num_dst_nodes, n_heads * hidden_dim)
        relations_crossing_attention_weight:
            Parameter the shape is (n_heads, hidden_dim)
        Returns:
        ----------
        output_features: Tensor

        """
        if len(dsttype_node_features) == 1:
            # (num_dst_nodes, n_heads * hidden_dim)
            dsttype_node_features = dsttype_node_features.squeeze(dim=0)
        else:
            # (dsttype_node_relations_num, num_dst_nodes, n_heads, hidden_dim)
            dsttype_node_features = dsttype_node_features.reshape(dsttype_node_features.shape[0], -1, self._num_heads, self._out_feats)
            # shape -> (dsttype_node_relations_num, dst_nodes_num, n_heads, 1),  (dsttype_node_relations_num, dst_nodes_num, n_heads, hidden_dim) * (n_heads, hidden_dim)
            dsttype_node_relation_attention = (dsttype_node_features * relations_crossing_attention_weight).sum(dim=-1, keepdim=True)
            dsttype_node_relation_attention = F.softmax(self.leaky_relu(dsttype_node_relation_attention), dim=0)
            # shape -> (dst_nodes_num, n_heads, hidden_dim),  (dsttype_node_relations_num, dst_nodes_num, n_heads, hidden_dim) * (dsttype_node_relations_num, dst_nodes_num, n_heads, 1)
            dsttype_node_features = (dsttype_node_features * dsttype_node_relation_attention).sum(dim=0)
            dsttype_node_features = self.dropout(dsttype_node_features)
            # shape -> (dst_nodes_num, n_heads * hidden_dim)
            dsttype_node_features = dsttype_node_features.reshape(-1, self._num_heads * self._out_feats)

        return dsttype_node_features

# relation fusing
class RelationFusing(nn.Module):

    def __init__(self, node_hidden_dim: int, relation_hidden_dim: int, num_heads: int, dropout: float = 0.0,
                 negative_slope: float = 0.2):
        r"""

        Parameters
        ----------
        node_hidden_dim: int
            node hidden feature size
        relation_hidden_dim: int
            relation hidden feature size
        num_heads: int
            number of heads in Multi-Head Attention
        dropout: float
            dropout rate, defaults: 0.0
        negative_slope: float
            negative slope, defaults: 0.2
        """
        super(RelationFusing, self).__init__()
        self.node_hidden_dim = node_hidden_dim
        self.relation_hidden_dim = relation_hidden_dim
        self.num_heads = num_heads

        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

    def forward(self, dst_node_features: list, dst_relation_embeddings: list,
                dst_node_feature_transformation_weight: list,
                dst_relation_embedding_transformation_weight: list):
        r"""
        Parameters
        ----------
        dst_node_features: list
            e.g [each shape is (num_dst_nodes, n_heads * node_hidden_dim)]
        dst_relation_embeddings: list
            e.g [each shape is (n_heads * relation_hidden_dim)]
        dst_node_feature_transformation_weight: list
            e.g [each shape is (n_heads, node_hidden_dim, node_hidden_dim)]
        dst_relation_embedding_transformation_weight:  list
            e.g [each shape is (n_heads, relation_hidden_dim, relation_hidden_dim)]

        Returns
        ----------
        dst_node_relation_fusion_feature: Tensor
            the target node representation after relation-aware representations fusion
        """
        if len(dst_node_features) == 1:
            # (num_dst_nodes, n_heads * hidden_dim)
            dst_node_relation_fusion_feature = dst_node_features[0]
        else:
            # (num_dst_relations, nodes, n_heads, node_hidden_dim)
            dst_node_features = torch.stack(dst_node_features, dim=0).reshape(len(dst_node_features), -1,
                                                                              self.num_heads, self.node_hidden_dim)
            # (num_dst_relations, n_heads, relation_hidden_dim)
            dst_relation_embeddings = torch.stack(dst_relation_embeddings, dim=0).reshape(len(dst_node_features),
                                                                                          self.num_heads,
                                                                                          self.relation_hidden_dim)
            # (num_dst_relations, n_heads, node_hidden_dim, node_hidden_dim)
            dst_node_feature_transformation_weight = torch.stack(dst_node_feature_transformation_weight, dim=0).reshape(
                len(dst_node_features), self.num_heads,
                self.node_hidden_dim, self.node_hidden_dim)
            # (num_dst_relations, n_heads, relation_hidden_dim, relation_hidden_dim)
            dst_relation_embedding_transformation_weight = torch.stack(dst_relation_embedding_transformation_weight,
                                                                       dim=0).reshape(len(dst_node_features),
                                                                                      self.num_heads,
                                                                                      self.relation_hidden_dim,
                                                                                      self.node_hidden_dim)
            # shape (num_dst_relations, nodes, n_heads, hidden_dim)
            dst_node_features = torch.einsum('abcd,acde->abce', dst_node_features,
                                             dst_node_feature_transformation_weight)

            # shape (num_dst_relations, n_heads, hidden_dim)
            dst_relation_embeddings = torch.einsum('abc,abcd->abd', dst_relation_embeddings,
                                                   dst_relation_embedding_transformation_weight)

            # shape (num_dst_relations, nodes, n_heads, 1)
            attention_scores = (dst_node_features * dst_relation_embeddings.unsqueeze(dim=1)).sum(dim=-1, keepdim=True)
            attention_scores = F.softmax(self.leaky_relu(attention_scores), dim=0)
            # (nodes, n_heads, hidden_dim)
            dst_node_relation_fusion_feature = (dst_node_features * attention_scores).sum(dim=0)
            dst_node_relation_fusion_feature = self.dropout(dst_node_relation_fusion_feature)
            # (nodes, n_heads * hidden_dim)
            dst_node_relation_fusion_feature = dst_node_relation_fusion_feature.reshape(-1,
                                                                                        self.num_heads * self.node_hidden_dim)

        return dst_node_relation_fusion_feature

# relationGraphConv
class RelationGraphConv(nn.Module):

    def __init__(self, in_feats: tuple, out_feats: int, num_heads: int, dropout: float = 0.0, negative_slope: float = 0.2):
        r"""
        Relation graph convolution layer

        Parameters
        ----------
        in_feats : pair of ints
            input feature size
        out_feats : int
            output feature size
        num_heads : int
            number of heads in Multi-Head Attention
        dropout : float
            optional, dropout rate, defaults: 0
        negative_slope : float
            optional, negative slope rate, defaults: 0.2
        """
        super(RelationGraphConv, self).__init__()
        self._in_src_feats, self._in_dst_feats = in_feats[0], in_feats[1]
        self._out_feats = out_feats
        self._num_heads = num_heads

        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.relu = nn.ReLU()

    def forward(self, graph: dgl.DGLHeteroGraph, feat: tuple, dst_node_transformation_weight: nn.Parameter,
                src_node_transformation_weight: nn.Parameter, relation_embedding: torch.Tensor,
                relation_transformation_weight: nn.Parameter):
        r"""

        Parameters
        ----------
        graph : specific relational DGLHeteroGraph
        feat : pair of torch.Tensor
            e.g The pair contains two tensors of shape (N_{in}, D_{in_{src}})` and (N_{out}, D_{in_{dst}}).
        dst_node_transformation_weight:
            e.g Parameter (input_dst_dim, n_heads * hidden_dim)
        src_node_transformation_weight:
            e.g Parameter (input_src_dim, n_heads * hidden_dim)
        relation_embedding: torch.Tensor
            e.g (relation_input_dim)
        relation_transformation_weight:
            e,g Parameter (relation_input_dim, n_heads * 2 * hidden_dim)

        Returns
        -------
        dst_features: torch.Tensor
            shape (N, H, D_out)` where H is the number of heads, and D_out is size of output feature.
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
        # Tensor, (n_heads, 2 * hidden_dim) -> (1, input_dst_dim) * (input_dst_dim, n_heads * hidden_dim)
        relation_attention_weight = torch.matmul(relation_embedding.unsqueeze(dim=0), relation_transformation_weight).view(self._num_heads, 2 * self._out_feats)

        # first decompose the weight vector into [a_l || a_r], then
        # a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j, This implementation is much efficient
        # Tensor, (N_dst, n_heads, 1),   (N_dst, n_heads, hidden_dim) * (n_heads, hidden_dim)
        e_dst = (feat_dst * relation_attention_weight[:, :self._out_feats]).sum(dim=-1, keepdim=True)
        # Tensor, (N_src, n_heads, 1),   (N_src, n_heads, hidden_dim) * (n_heads, hidden_dim)
        e_src = (feat_src * relation_attention_weight[:, self._out_feats:]).sum(dim=-1, keepdim=True)
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

        graph.update_all(fn.u_mul_e('ft', 'a', 'msg'), fn.sum('msg', 'feat'))
        # (N_dst, n_heads * hidden_dim), reshape (N_dst, n_heads, hidden_dim)
        dst_features = graph.dstdata.pop('feat').reshape(-1, self._num_heads * self._out_feats)

        dst_features = self.relu(dst_features)

        return dst_features




class R_HGNN_Layer(nn.Module):
    def __init__(self, graph, input_dim: int, hidden_dim: int, relation_input_dim: int,
                 relation_hidden_dim: int, n_heads: int = 8, dropout: float = 0.2, negative_slope: float = 0.2,
                 residual: bool = True, norm: bool = False):
        """
        Parameters
        ----------
        graph:
            a heterogeneous graph
        input_dim: int
            node input dimension
        hidden_dim: int
            node hidden dimension
        relation_input_dim: int
            relation input dimension
        relation_hidden_dim: int
            relation hidden dimension
        n_heads: int
            number of attention heads
        dropout: float
            dropout rate
        negative_slope: float
            negative slope
        residual: boolean
            residual connections or not
        norm: boolean
            layer normalization or not
        """
        super(R_HGNN_Layer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.relation_input_dim = relation_input_dim
        self.relation_hidden_dim = relation_hidden_dim
        self.n_heads = n_heads
        self.dropout = dropout
        self.negative_slope = negative_slope
        self.residual = residual
        self.norm = norm

        # node transformation parameters of each type
        self.node_transformation_weight = nn.ParameterDict({
            ntype: nn.Parameter(torch.randn(input_dim, n_heads * hidden_dim))
            for ntype in graph.ntypes
        })

        # relation transformation parameters of each type, used as attention queries
        self.relation_transformation_weight = nn.ParameterDict({
            etype: nn.Parameter(torch.randn(relation_input_dim, n_heads * 2 * hidden_dim))
            for etype in graph.etypes
        })

        # relation propagation layer of each relation
        self.relation_propagation_layer = nn.ModuleDict({
            etype: nn.Linear(relation_input_dim, n_heads * relation_hidden_dim)
            for etype in graph.etypes
        })

        # hetero conv modules, each RelationGraphConv deals with a single type of relation
        self.hetero_conv = HeteroGraphConv({
            etype: RelationGraphConv(in_feats=(input_dim, input_dim), out_feats=hidden_dim,
                                     num_heads=n_heads, dropout=dropout, negative_slope=negative_slope)
            for etype in graph.etypes
        })

        if self.residual:
            # residual connection
            self.res_fc = nn.ModuleDict()
            self.residual_weight = nn.ParameterDict()
            for ntype in graph.ntypes:
                self.res_fc[ntype] = nn.Linear(input_dim, n_heads * hidden_dim)
                self.residual_weight[ntype] = nn.Parameter(torch.randn(1))

        if self.norm:
            self.layer_norm = nn.ModuleDict({ntype: nn.LayerNorm(n_heads * hidden_dim) for ntype in graph.ntypes})

        # relation type crossing attention trainable parameters
        self.relations_crossing_attention_weight = nn.ParameterDict({
            etype: nn.Parameter(torch.randn(n_heads, hidden_dim))
            for etype in graph.etypes
        })
        # different relations crossing layer
        self.relations_crossing_layer = RelationCrossing(in_feats=n_heads * hidden_dim,
                                                         out_feats=hidden_dim,
                                                         num_heads=n_heads,
                                                         dropout=dropout,
                                                         negative_slope=negative_slope)

        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        for weight in self.node_transformation_weight:
            nn.init.xavier_normal_(self.node_transformation_weight[weight], gain=gain)
        for weight in self.relation_transformation_weight:
            nn.init.xavier_normal_(self.relation_transformation_weight[weight], gain=gain)
        for etype in self.relation_propagation_layer:
            nn.init.xavier_normal_(self.relation_propagation_layer[etype].weight, gain=gain)
        if self.residual:
            for ntype in self.res_fc:
                nn.init.xavier_normal_(self.res_fc[ntype].weight, gain=gain)
        for weight in self.relations_crossing_attention_weight:
            nn.init.xavier_normal_(self.relations_crossing_attention_weight[weight], gain=gain)

    def forward(self, graph: dgl.DGLHeteroGraph, relation_target_node_features: dict, relation_embedding: dict):
        """

        :param graph: dgl.DGLHeteroGraph
        :param relation_target_node_features: dict, {relation_type: target_node_features shape (N_nodes, input_dim)},
               each value in relation_target_node_features represents the representation of target node features
        :param relation_embedding: embedding for each relation, dict, {etype: feature}
        :return: output_features: dict, {relation_type: target_node_features}
        """
        # in each relation, target type of nodes has an embedding
        # dictionary of {(srctype, etypye, dsttype): target_node_features}
        input_src = relation_target_node_features

        if graph.is_block:
            input_dst = {}
            for srctype, etypye, dsttype in relation_target_node_features:
                input_dst[(srctype, etypye, dsttype)] = relation_target_node_features[(srctype, etypye, dsttype)][
                                                        :graph.number_of_dst_nodes(dsttype)]
        else:
            input_dst = relation_target_node_features

        # output_features, dict {(srctype, etypye, dsttype): target_node_features}
        output_features = self.hetero_conv(graph, input_src, input_dst, relation_embedding,
                                           self.node_transformation_weight, self.relation_transformation_weight)

        # residual connection for the target node
        if self.residual:
            for srctype, etype, dsttype in output_features:
                alpha = torch.sigmoid(self.residual_weight[dsttype])
                output_features[(srctype, etype, dsttype)] = output_features[(srctype, etype, dsttype)] * alpha + \
                                                             self.res_fc[dsttype](
                                                                 input_dst[(srctype, etype, dsttype)]) * (1 - alpha)

        output_features_dict = {}
        # different relations crossing layer
        for srctype, etype, dsttype in output_features:
            # (dsttype_node_relations_num, dst_nodes_num, n_heads * hidden_dim)
            dst_node_relations_features = torch.stack([output_features[(stype, reltype, dtype)]
                                                   for stype, reltype, dtype in output_features if dtype == dsttype], dim=0)

            output_features_dict[(srctype, etype, dsttype)] = self.relations_crossing_layer(dst_node_relations_features,
                                                                                            self.relations_crossing_attention_weight[etype])

        # layer norm for the output
        if self.norm:
            for srctype, etype, dsttype in output_features_dict:
                output_features_dict[(srctype, etype, dsttype)] = self.layer_norm[dsttype](output_features_dict[(srctype, etype, dsttype)])

        relation_embedding_dict = {}
        for etype in relation_embedding:
            relation_embedding_dict[etype] = self.relation_propagation_layer[etype](relation_embedding[etype])

        # relation features after relation crossing layer, {(srctype, etype, dsttype): target_node_features}
        # relation embeddings after relation update, {etype: relation_embedding}
        return output_features_dict, relation_embedding_dict