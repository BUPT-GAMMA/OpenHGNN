import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from . import BaseModel, register_model


@register_model('HGSL')
class HGSL(BaseModel):
    r"""
    HGSL, Heterogeneous Graph Structure Learning from `paper <http://www.shichuan.org/doc/100.pdf>`_.

    Parameters
    ----------
    feat_dims : dict
        The feature dimensions of different node types.
    undirected_relations : str
        The HGSL model can only handle undirected heterographs, while in the dgl.heterograph format, directed edges are
        stored in two different edge types, separately and symmetrically, to represent undirected edge. Hence you have
        to specify which relations are those distinct undirected relations. In this parameter, each undirected relation
        is separated with a comma. For example, in a heterograph with 2 undirected relations: paper-author and
        paper-subject, there are 4 type of edges stored in the dgl.heterograph: paper-author, author-paper,
        paper-subject, subject-paper. Then this parameter can be "paper-author,paper-subject",
        "author-paper,paper-subject", "paper-author,subject-paper" or "author-paper,subject-paper".
    device: str
        The GPU device to select, like 'cuda:0'.
    metapaths : list
        The metapath name list.
    mp_emb_dim : int
        The dimension of metapath embeddings from metapath2vec.
    hidden_dim : int
        The dimension of mapped features in the graph generating procedure.
    num_heads: int
        Number of heads in the K-head weighted cosine similarity function.
    fs_eps : float
        Threshold of feature similarity graph :math:`\epsilon^{FS}`.
    fp_eps : float
        Threshold of feature propagation graph :math:`\epsilon^{FP}`.
    mp_eps : float
        Threshold of semantic graph :math:`\epsilon^{MP}`.
    gnn_emd_dim : int
        The dimension of hidden layers of the downstream GNN.
    gnn_dropout : float
        The dropout ratio of features in the downstream GNN.
    category : str
        The target node type which the model will predict on.
    out_dim : int
        number of classes of the target node type.

    Attributes
    -----------
    fgg_direct : nn.ModuleDict
        Feature similarity graph generator(:math:`S_r^{FS}`) dict in equation 2 of paper, in which keys are
        undirected-relation strs.
    fgg_left: nn.ModuleDict
        Feature propagation graph generator(:math:`S_r^{FH}`) dict which generates the graphs in equation 5 of paper.
    fgg_right: nn.ModuleDict
        Feature propagation graph generator(:math:`S_r^{FT}`) dict which generates the graphs in equation 6 of paper.
    fg_agg : nn.ModuleDict
        A channel attention layer, in which a layer fuses one feature similarity graph and two feature propagation
        graphs generated, in equation 7 of paper.
    sgg_gen : nn.ModuleDict
        Semantic subgraph generator(:math:`S_{r,m}^{MP}`) dict, in equation 8 of paper.
    sg_agg : nn.ModuleDict
        The channel attention layer which fuses semantic subgraphs, in equation 9 of paper.
    overall_g_agg : nn.ModuleDict
        The channel attention layer which fuses the learned feature graph, semantic graph and the original graph.
    encoder : nn.ModuleDict
        The type-specific mapping layer in equation 1 of paper.

    Note
    ----
    This model under the best config has some slight differences compared with the code given by the paper author,
    which seems having little impact on performance:

    1. The regularization item in loss is on all parameters of the model, while in the author's code, it is only on the
       generated adjacent matrix. If you want to implement the latter, a new task of OpenHGNN is needed.

    2. The normalization of input adjacent matrix is separately on different adjacent matrices of different
       relations, while in the author's code, it is on the entire adjacent matrix composed of adjacent matrices of all
       relations.
    """

    @classmethod
    def build_model_from_args(cls, args, hg):

        feat_dims = dict()
        for ntype in hg.ntypes:
            if 'h' in hg.nodes[ntype].data.keys():
                feat_dims[ntype] = hg.nodes[ntype].data['h'].shape[1]
            else:
                feat_dims[ntype] = 128

        # Extract undirected_relations
        und_rels = args.undirected_relations.split(',')
        undirected_relations = list()
        for etype in hg.canonical_etypes:
            if etype[1] in und_rels:
                undirected_relations.append(etype)

        device = hg.device

        metapaths = list()
        if args.dataset_name == 'acm4GTN':
            for feature_name in hg.nodes["paper"].data.keys():
                if "m2v" in feature_name:
                    metapaths.append(feature_name)
            mp_emb_dim = hg.nodes["paper"].data["pap_m2v_emb"].shape[1]
        
        elif args.dataset_name == 'dblp4GTN':
            for feature_name in hg.nodes['paper'].data.keys():
                if 'h' not in feature_name:
                    metapaths.append(feature_name)
            mp_emb_dim = hg.nodes['paper'].data['PAPCP'].shape[1]
        
        elif args.dataset_name == 'yelp4HGSL':
            for feature_name in hg.nodes['b'].data.keys():
                if 'h' not in feature_name:
                    metapaths.append(feature_name)
            mp_emb_dim = hg.nodes['b'].data['bub'].shape[1]
            
        else:
            raise NotImplemented("HGSL on dataset {} has not been implemented".format(args.dataset_name))

        return cls(feat_dims=feat_dims, undirected_relations=undirected_relations, device=device, metapaths=metapaths,
                   mp_emb_dim=mp_emb_dim, hidden_dim=args.hidden_dim, num_heads=args.num_heads,
                   fs_eps=args.fs_eps, fp_eps=args.fp_eps, mp_eps=args.mp_eps,
                   gnn_emd_dim=args.gnn_emd_dim, gnn_dropout=args.gnn_dropout,
                   category=args.category, num_class=args.out_dim)

    def __init__(self, feat_dims, undirected_relations, device, metapaths, mp_emb_dim, hidden_dim, num_heads, fs_eps,
                 fp_eps, mp_eps, gnn_emd_dim, gnn_dropout, category, num_class):

        super().__init__()
        self.device = device
        self.ud_rels = undirected_relations
        self.node_types = list(feat_dims.keys())
        self.feat_dims = feat_dims
        self.non_linear = nn.ReLU()
        self.category = category
        self.metapaths = metapaths

        nnmd = nn.ModuleDict
        self.fgg_direct, self.fgg_left, self.fgg_right, self.fg_agg, self.sgg_gen, self.sg_agg, self.overall_g_agg = \
            nnmd({}), nnmd({}), nnmd({}), nnmd({}), nnmd({}), nnmd({}), nnmd({})
        # Feature encoder
        self.encoder = nnmd(
            dict(zip(self.node_types, [nn.Linear(feat_dims[node_type], hidden_dim) for node_type in self.node_types])))

        for canonical_etype in undirected_relations:
            undirected_relation = canonical_etype[1]
            # Feature Graph Generator
            self.fgg_direct[undirected_relation] = GraphGenerator(hidden_dim, num_heads, fs_eps, self.device)
            self.fgg_left[undirected_relation] = GraphGenerator(feat_dims[canonical_etype[0]], num_heads, fp_eps,
                                                                self.device)
            self.fgg_right[undirected_relation] = GraphGenerator(feat_dims[canonical_etype[2]], num_heads, fp_eps,
                                                                 self.device)
            self.fg_agg[undirected_relation] = GraphChannelAttLayer(3)

            # Semantic Graph Generator
            self.sgg_gen[undirected_relation] = nnmd(dict(
                zip(metapaths, [GraphGenerator(mp_emb_dim, num_heads, mp_eps, self.device) for _ in metapaths])))
            self.sg_agg[undirected_relation] = GraphChannelAttLayer(len(metapaths))

            # Overall Graph Generator
            self.overall_g_agg[undirected_relation] = GraphChannelAttLayer(3)

        # Graph Convolution
        if len(set(feat_dims.values())) == 1:
            self.GCN = GCN(list(self.feat_dims.values())[0], gnn_emd_dim, num_class, gnn_dropout)
        else:
            raise Exception("Downstream model GCN can only accept features for "
                            "different node types of the same dimension")

    def forward(self, hg, h_features):
        r"""
        Parameters
        ----------
        hg : dgl.DGlHeteroGraph
            All input data is stored in this graph.
            The graph should be an undirected heterogeneous graph.
            Every node type in graph should have its feature named 'h' and the same feature dimension.
            Every node type in graph should have its metapath2vec embedding feature named 'xxx_m2v_emb'
            and the same feature dimension.
        h_features : dict

        Returns
        --------
        result : dict
            The target node type and the corresponding node embeddings.
        """
        def generate_node_indexes(hg):
            indexes = dict()
            index = 0
            for node_type in hg.ntypes:
                indexes[node_type] = (index, index + hg.num_nodes(node_type))
                index += hg.num_nodes(node_type)

            return indexes

        def construct_homo_adj(new_adjs, hg, node_indexes, device):
            new_homo_adj = torch.zeros(size=(hg.num_nodes(), hg.num_nodes())).to(device)

            for canonical_etype, new_adj in new_adjs.items():
                row_range = node_indexes[canonical_etype[0]]
                column_range = node_indexes[canonical_etype[2]]
                new_homo_adj[row_range[0]:row_range[1], column_range[0]:column_range[1]] = new_adj

            temp = new_homo_adj.clone()
            new_homo_adj = temp + new_homo_adj.t()
            # new_homo_adj += new_homo_adj.t()
            new_homo_adj = F.normalize(new_homo_adj, dim=0, p=1)
            return new_homo_adj

        def construct_homo_feature(hg, device):
            homo_feature = list()
            for ntype in hg.ntypes:
                homo_feature.append(hg.nodes[ntype].data['h'])

            homo_feature = torch.cat(homo_feature, dim=0).to(device)
            return homo_feature

        # Heterogeneous Feature Mapping
        mapped_feats = dict()
        for ntype in self.node_types:
            if 'h' in hg.nodes[ntype].data.keys():
                mapped_feats[ntype] = self.non_linear(self.encoder[ntype](hg.nodes[ntype].data['h'].clone()))
            else:
                mapped_feats[ntype] = self.non_linear(self.encoder[ntype](h_features[ntype].clone()))

        # Heterogeneous Graph Generation
        new_adjs = dict()
        for canonical_etype in self.ud_rels:
            undirected_relation = canonical_etype[1]
            ori_g = F.normalize(hg.adj(etype=canonical_etype).to_dense().to(self.device), dim=1, p=2)

            # Feature Graph Generation
            fg_direct = self.fgg_direct[undirected_relation](mapped_feats[canonical_etype[0]],
                                                             mapped_feats[canonical_etype[2]])
            if 'h' in hg.nodes[canonical_etype[0]].data.keys() and 'h' in hg.nodes[canonical_etype[2]].data.keys():
                fmat_l, fmat_r = hg.nodes[canonical_etype[0]].data['h'], hg.nodes[canonical_etype[2]].data['h']
            else:
                fmat_l, fmat_r = h_features[canonical_etype[0]], h_features[canonical_etype[2]]
            sim_l, sim_r = self.fgg_left[undirected_relation](fmat_l, fmat_l), self.fgg_right[undirected_relation](
                fmat_r, fmat_r)
            fg_left, fg_right = sim_l.mm(ori_g), sim_r.mm(ori_g.t()).t()

            feat_g = self.fg_agg[undirected_relation]([fg_direct, fg_left, fg_right])

            # Semantic Graph Generation
            sem_g_list = [self.sgg_gen[undirected_relation][mp](hg.nodes[canonical_etype[0]].data[mp],
                                                                hg.nodes[canonical_etype[2]].data[mp]) for mp in
                          self.metapaths]
            sem_g = self.sg_agg[undirected_relation](sem_g_list)

            # Overall Graph
            new_adjs[canonical_etype] = self.overall_g_agg[undirected_relation]([feat_g, sem_g, ori_g])

        node_indexes = generate_node_indexes(hg)
        new_homo_adj = construct_homo_adj(new_adjs, hg, node_indexes, self.device)
        homo_feature = construct_homo_feature(hg, self.device)
        x = self.GCN(homo_feature, new_homo_adj)
        result = {self.category: x[node_indexes[self.category][0]:node_indexes[self.category][1], :]}
        return result


class MetricCalcLayer(nn.Module):
    r"""
    Calculate metric in equation 3 of paper.

    Parameters
    ----------
    nhid : int
        The dimension of mapped features in the graph generating procedure.
    """
    def __init__(self, nhid):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(1, nhid))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, h):
        r"""
        Parameters
        ----------
        h : tensor
            The result of the Hadamard product in equation 3 of paper.
        """
        return h * self.weight


class GraphGenerator(nn.Module):
    r"""
    Generate a graph using similarity.
    """
    def __init__(self, dim, num_head=2, threshold=0.1, dev=None):
        super(GraphGenerator, self).__init__()
        self.threshold = threshold
        self.metric_layer = nn.ModuleList()
        for i in range(num_head):
            self.metric_layer.append(MetricCalcLayer(dim))
        self.num_head = num_head
        self.dev = dev

    def forward(self, left_h, right_h):
        r"""
        Parameters
        ----------
        left_h : tensor
            The first input embedding matrix.
        right_h : tensor
            The second input embedding matrix.
        """
        def cos_sim(a, b, eps=1e-8):
            a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
            a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
            b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
            sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
            return sim_mt

        if torch.sum(left_h) == 0 or torch.sum(right_h) == 0:
            return torch.zeros((left_h.shape[0], right_h.shape[0])).to(self.dev)
        s = torch.zeros((left_h.shape[0], right_h.shape[0])).to(self.dev)
        zero_lines = torch.nonzero(torch.sum(left_h, 1) == 0)
        # The ReLU function will generate zero lines, which lead to the nan (divided by zero) problem.
        if len(zero_lines) > 0:
            left_h[zero_lines, :] += 1e-8
        for i in range(self.num_head):
            weighted_left_h = self.metric_layer[i](left_h)
            weighted_right_h = self.metric_layer[i](right_h)
            s += cos_sim(weighted_left_h, weighted_right_h)
        s /= self.num_head
        s = torch.where(s < self.threshold, torch.zeros_like(s), s)
        return s


class GraphChannelAttLayer(nn.Module):
    r"""
    The graph channel attention layer in equation 7, 9 and 10 of paper.
    """
    def __init__(self, num_channel):
        super(GraphChannelAttLayer, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(num_channel, 1, 1))
        nn.init.constant_(self.weight, 0.1)  # equal weight

    def forward(self, adj_list):
        r"""
        Parameters
        ----------
        adj_list : list
            The list of adjacent matrices.
        """
        adj_list = torch.stack(adj_list)
        # Row normalization of all graphs generated
        adj_list = F.normalize(adj_list, dim=1, p=1)
        # Hadamard product + summation -> Conv
        return torch.sum(adj_list * F.softmax(self.weight, dim=0), dim=0)


class GCN(nn.Module):
    r"""
    The downstream GCN model.
    """
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        r"""
        Parameters
        ----------
        x : tensor
            The feature matrix.
        adj : tensor
            The adjacent matrix.
        """
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x


class GraphConvolution(nn.Module):
    r"""
    The downstream GCN layer.
    """
    def __init__(self, in_features, out_features, bias=True):

        def reset_parameters(self):
            stdv = 1. / math.sqrt(self.weight.size(1))
            self.weight.data.uniform_(-stdv, stdv)
            if self.bias is not None:
                self.bias.data.uniform_(-stdv, stdv)

        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        reset_parameters(self)

    def forward(self, inputs, adj):
        r"""
        Parameters
        ----------
        inputs : tensor
            The feature matrix.
        adj : tensor
            The adjacent matrix.
        """
        support = torch.mm(inputs, self.weight)  # HW in GCN
        output = torch.mm(adj, support)  # AHW
        if self.bias is not None:
            return output + self.bias
        else:
            return output
