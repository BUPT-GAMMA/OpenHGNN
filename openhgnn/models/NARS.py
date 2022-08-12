import dgl
import torch as th
import torch.nn as nn
from . import BaseModel, register_model
import dgl.function as fn


@register_model('NARS')
class NARS(BaseModel):
    r"""
        `SCALABLE GRAPH NEURAL NETWORKS FOR HETEROGENEOUS GRAPHS <https://arxiv.org/pdf/2011.09679.pdf>`_.

        Given a heterogeneous graph :math:`G` and its edge relation type set :math:`\mathcal{R}`,
        our proposed method first samples :math:`K` unique subsets from :math:`\mathcal{R}`.
        Then for each sampled subset :math:`R_i \subseteq \mathcal{R}`, we generate a relation subgraph
        :math:`G_i` from :math:`G` in which only edges whose type belongs to :math:`R_i` are kept.
        We treat :math:`G_i` as a homogeneous graph or a bipartite graph, and perform neighbor aggregation to generate :math:`L`-hop neighbor features for each node.
        Let :math:`H_{v,0}` be the input features (of dimension :math:`D`) for node :math:`v`. For each subgraph :math:`G_i`
        , the :math:`l`-th hop
        features :math:`H_{v,l}^{i}` are computed as

        .. math::
            H_{v, l}^{i}=\sum_{u \in N_{i}(v)} \frac{1}{\left|N_{i}(v)\right|} H_{u, l-1}^{i}


        where :math:`N_i(v)` is the set of neighbors of node :math:`v` in :math:`G_i`.

        For each layer :math:`l`, we let the model adaptively learn which relation-subgraph features to use by aggregating
        features from different subgraphs :math:`G_i` with learnable 1-D convolution. The aggregated :math:`l`-hop
        features across all subgraphs are calculated as


        .. math::
            H_{v, l}^{a g g}=\sum_{i=1}^{K} a_{i, l} \cdot H_{v, l}^{i}


        where :math:`H^i` is the neighbor averaging features on subgraph :math:`G_i` and :math:`a_{i,l}` is a learned vector of length equal
        to the feature dimension :math:`D`.

        Parameters
        ----------
        num_hops : int
            Number of hops.
        category : str
            Type of predicted nodes.
        hidden_dim : int
            The dimention of hidden layer.
        num_feats : int
            The number of relation subsets.

        Note
        ----
        We do not support the dataset without feature, (e.g. HGBn-Freebase
        because the model performs neighbor aggregation to generate :math:`L`-hop neighbor features at once.

        """

    @classmethod
    def build_model_from_args(cls, args, hg):
        num_hops = args.num_hops + 1

        return cls(num_hops=num_hops,
                   args = args,
                   hg = hg,
                   )

    def __init__(self, num_hops, args, hg):
        super(NARS, self).__init__()
        self.category = args.category
        self.dropout = args.dropout
        self.input_dropout = args.input_dropout
        # self.feats = feats
        self.device = args.device
        self.num_hops = num_hops
        self.args = args

        in_size = hg.nodes[args.category].data["h"].shape[1]

        etypes = hg.canonical_etypes
        mps = []
        for etype in etypes:
            if etype[0] == args.category:
                for dst_e in etypes:
                    if etype[0] == dst_e[2] and etype[2] == dst_e[0] and etype[0] != etype[2]:
                        mps.append([etype, dst_e])
        self.mps = mps
        self.num_feats = len(mps)

        with th.no_grad():
            self.feats = preprocess_features(hg, mps, args, args.device, self.args.category)
            print("Done preprocessing")

        self.seq = nn.Sequential(
            WeightedAggregator(self.num_feats, in_size, num_hops),
            SIGN(in_size, args.hidden_dim, args.out_dim, num_hops,
                 args.ff_layer, args.dropout, args.input_dropout)
        )

    def forward(self, hg, h_dict):

        #ffeats = [x.to(self.device) for x in self.feats]
        ffeats = [x.to(self.device) for x in self.feats]
        return {self.category: self.seq.forward(ffeats)}

    def reset_parameters(self):
        self.seq.register_parameter()


def preprocess_features(g, mps, args, device, predict):
    """
        pre-process heterogeneous graph g to generate neighbor-averaged features
        for each relation subsets

        Parameters
        -----------
        g :
            heterogeneous graph
        rel_subsets :
            relations of subsets
        args :
            arguments
        device :
            device

        Return
        ------
            new features of each relation subsets

    """
    category_dim = g.nodes[predict].data["h"].shape[1]
    for ntype in g.ntypes:
        ntype_dim = g.nodes[ntype].data["h"].shape[1]
        if category_dim != ntype_dim:
            rand_weight = th.Tensor(ntype_dim, category_dim).uniform_(-0.5, 0.5).to(device)
            g.nodes[ntype].data["h"] = th.matmul(g.nodes[ntype].data["h"], rand_weight)

    num_paper, feat_size = g.nodes[predict].data["h"].shape

    new_feats = [th.zeros(num_paper, len(mps), feat_size) for _ in range(args.num_hops + 1)]

    for subset_id, subset in enumerate(mps):
        # print(subset)
        feats = gen_rel_subset_feature(g, subset, args, device, predict)
        for i in range(args.num_hops + 1):
            feat = feats[i]
            new_feats[i][:feat.shape[0], subset_id, :] = feat
        feats = None
    return new_feats

def gen_rel_subset_feature(g, rel_subset, args, device, predict):
    """
        Build relation subgraph given relation subset and generate multi-hop
        neighbor-averaged feature on this subgraph

        Parameters
        ----------
        g :
            Heterogeneous graph
        rel_subset :
            relation of subsets
        args :
            arguments
        device :
            device

        Returns
        ------
        new features of a relation subsets
    """

    new_g = g.edge_type_subgraph(rel_subset).to(g.device)

    ntypes = new_g.ntypes
    # set node feature and calc deg
    for ntype in ntypes:
        num_nodes = new_g.number_of_nodes(ntype)
        if num_nodes < g.nodes[ntype].data["h"].shape[0]:
            new_g.nodes[ntype].data["hop_0"] = g.nodes[ntype].data["h"][:num_nodes, :]
        else:
            new_g.nodes[ntype].data["hop_0"] = g.nodes[ntype].data["h"]
        deg = 0
        for etype in new_g.etypes:
            _, _, dtype = new_g.to_canonical_etype(etype)
            if ntype == dtype:
                deg = deg + new_g.in_degrees(etype=etype)
        if th.is_tensor(deg):
            norm = 1.0 / deg.float()
            norm[th.isinf(norm)] = 0
            new_g.nodes[ntype].data["norm"] = norm.view(-1, 1).to(device)

    res = []

    # compute k-hop feature
    for hop in range(1, args.num_hops + 1):
        ntype2feat = {}
        for etype in new_g.etypes:
            stype, _, dtype = new_g.to_canonical_etype(etype)
            new_g[etype].update_all(fn.copy_u(f'hop_{hop-1}', 'm'), fn.sum('m', 'new_feat'))
            new_feat = new_g.nodes[dtype].data.pop("new_feat")
            assert("new_feat" not in new_g.nodes[stype].data)
            if dtype in ntype2feat:
                ntype2feat[dtype] += new_feat
            else:
                ntype2feat[dtype] = new_feat
        for ntype in new_g.ntypes:
            assert ntype in ntype2feat  # because subgraph is not directional
            feat_dict = new_g.nodes[ntype].data
            old_feat = feat_dict.pop(f"hop_{hop-1}")
            if ntype == predict:
                res.append(old_feat.cpu())
            feat_dict[f"hop_{hop}"] = ntype2feat.pop(ntype).mul_(feat_dict["norm"])

    res.append(new_g.nodes[predict].data.pop(f"hop_{args.num_hops}").cpu())
    return res



class FeedForwardNet(nn.Module):
    """
        A feedforward net.

        Input
        ------
        in_feats :
            input feature dimention
        hidden :
            hidden layer dimention
        out_feats :
            output feature dimention
        num_layers :
            number of layers
        dropout :
            dropout rate
    """
    def __init__(self, in_feats, hidden, out_feats, num_layers, dropout):
        super(FeedForwardNet, self).__init__()
        self.layers = nn.ModuleList()
        self.num_layers = num_layers
        if num_layers == 1:
            self.layers.append(nn.Linear(in_feats, out_feats))
        else:
            self.layers.append(nn.Linear(in_feats, hidden))
            for i in range(num_layers - 2):
                self.layers.append(nn.Linear(hidden, hidden))
            self.layers.append(nn.Linear(hidden, out_feats))
        if self.num_layers > 1:
            self.prelu = nn.PReLU()
            self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight, gain=gain)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        for layer_id, layer in enumerate(self.layers):
            x = layer(x)
            if layer_id < self.num_layers - 1:
                x = self.dropout(self.prelu(x))
        return x


class SIGN(nn.Module):
    """
        The SIGN model.

        Parameters
        ------------
        in_feats :
            input feature dimention
        hidden :
            hidden layer dimention
        out_feats :
            output feature dimention
        num_hops :
            number of hops
        num_layers :
            number of layers
        dropout :
            dropout rate
        input_drop :
            whether or not to dropout when inputting features

    """
    def __init__(
        self, in_feats, hidden, out_feats, num_hops, num_layers, dropout, input_drop
    ):
        super(SIGN, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.prelu = nn.PReLU()
        self.inception_ffs = nn.ModuleList()
        self.input_drop = input_drop
        for i in range(num_hops):
            self.inception_ffs.append(
                FeedForwardNet(in_feats, hidden, hidden, num_layers, dropout)
            )
        self.project = FeedForwardNet(
            num_hops * hidden, hidden, out_feats, num_layers, dropout
        )

    def forward(self, feats):
        hidden = []
        for feat, ff in zip(feats, self.inception_ffs):
            if self.input_drop:
                feat = self.dropout(feat)
            hidden.append(ff(feat))
        out = self.project(self.dropout(self.prelu(th.cat(hidden, dim=-1))))
        return out


class WeightedAggregator(nn.Module):
    """
        Get new features by multiplying the old features by the weight matrix.

        Parameters
        -------------
        num_feats :
            number of subsets
        in_feats :
            input feature dimention
        num_hops :
            number of hops


    """
    def __init__(self, num_feats, in_feats, num_hops):
        super(WeightedAggregator, self).__init__()
        self.agg_feats = nn.ParameterList()
        for _ in range(num_hops):
            self.agg_feats.append(nn.Parameter(th.Tensor(num_feats, in_feats)))
            nn.init.xavier_uniform_(self.agg_feats[-1])

    def forward(self, feats):
        new_feats = []
        for feat, weight in zip(feats, self.agg_feats):
            new_feats.append((feat * weight.unsqueeze(0)).sum(dim=1).squeeze())
        return new_feats


