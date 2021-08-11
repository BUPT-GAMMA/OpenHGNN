import dgl
import math
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv, EdgeWeightNorm
from ..utils import transform_relation_graph_list
from . import BaseModel, register_model
import dgl.function as fn

@register_model('NARS')
class NARS(BaseModel):
    """
    NARS model
    Arguments
    ----------
    seq: sequential model
    Input
    ----------
    num_hops: number of hops
    num_feats: number of subset
    num_classes: number of node types
    category: type of predicted nodes
    in_size: input feature dimention
    hidden_size: Hidden layer dimension
    dropout: probability of an element to be dropped out.
    ff_layer: number of layers
    feats: preprocessed feats
    device: device
    Output
    ----------
    output feature
    """
    @classmethod
    def build_model_from_args(cls, args, hg):
        num_hops = args.R + 1
        rel_subsets =  [['paper-author'], ['paper-field']]

        with th.no_grad():
            feats = preprocess_features(hg, rel_subsets, args, args.device)
            print("Done preprocessing")

        _, num_feats, in_feats = feats[0].shape

        return cls(num_hops=num_hops,
                   num_feats=num_feats,
                   num_classes=args.out_dim,
                   category=args.category,
                   in_size=in_feats,
                   hidden_size=args.hidden_dim,
                   dropout=args.dropout,
                   ff_layer=args.ff_layer,
                   input_dropout=args.input_dropout,
                   feats=feats,
                   device=args.device
                   )

    def __init__(self, num_hops, num_feats, num_classes, category, in_size, hidden_size, dropout, ff_layer, input_dropout, feats, device):
        super(NARS, self).__init__()
        self.category = category
        self.dropout = dropout
        self.input_dropout = input_dropout
        self.feats = feats
        self.device = device

        self.seq = nn.Sequential(
            WeightedAggregator(num_feats, in_size, num_hops),
            SIGN(in_size, hidden_size, num_classes, num_hops,
                 ff_layer, dropout, input_dropout)
        )

    def forward(self, hg):
        ffeats = [x.to(self.device) for x in self.feats]
        return {self.category: self.seq.forward(ffeats)}

    def reset_parameters(self):
        self.seq.register_parameter()


def preprocess_features(g, rel_subsets, args, device):
    """
    pre-process heterogeneous graph g to generate neighbor-averaged features
    for each relation subsets
    """
    paper_dim = g.nodes["paper"].data["feat"].shape[1]
    author_dim = g.nodes["author"].data["feat"].shape[1]

    assert (paper_dim >= author_dim)
    if paper_dim > author_dim:
        print(f"Randomly embedding features from dimension {author_dim} to {paper_dim}")
        author_feat = g.nodes["author"].data.pop("feat")
        field_feat = g.nodes["field"].data.pop("feat")
        rand_weight = th.Tensor(author_dim, paper_dim).uniform_(-0.5, 0.5).to(device)
        g.nodes["author"].data["feat"] = th.matmul(author_feat, rand_weight)
        g.nodes["field"].data["feat"] = th.matmul(field_feat, rand_weight)

    num_paper, feat_size = g.nodes["paper"].data["feat"].shape   # TODO
    new_feats = [th.zeros(num_paper, len(rel_subsets), feat_size) for _ in range(args.R + 1)]
    print("Start generating features for each sub-metagraph:")
    for subset_id, subset in enumerate(rel_subsets):
        print(subset)
        feats = gen_rel_subset_feature(g, subset, args, device)
        for i in range(args.R + 1):
            feat = feats[i]
            new_feats[i][:feat.shape[0], subset_id, :] = feat
        feats = None
    return new_feats

def gen_rel_subset_feature(g, rel_subset, args, device):
    """
    Build relation subgraph given relation subset and generate multi-hop
    neighbor-averaged feature on this subgraph
    """
    if args.cpu_preprocess:
        device = "cpu"
    new_edges = {}
    ntypes = set()
    for etype in rel_subset:
        stype, _, dtype = g.to_canonical_etype(etype)
        src, dst = g.all_edges(etype=etype)
        src = src.cpu().numpy()
        dst = dst.cpu().numpy()
        new_edges[(stype, etype, dtype)] = (src, dst)
        new_edges[(dtype, etype + "_r", stype)] = (dst, src)
        ntypes.add(stype)
        ntypes.add(dtype)
    new_g = dgl.heterograph(new_edges)

    # set node feature and calc deg
    for ntype in ntypes:
        num_nodes = new_g.number_of_nodes(ntype)
        if num_nodes < g.nodes[ntype].data["feat"].shape[0]:
            new_g.nodes[ntype].data["hop_0"] = g.nodes[ntype].data["feat"][:num_nodes, :]
        else:
            new_g.nodes[ntype].data["hop_0"] = g.nodes[ntype].data["feat"].to(device)
        deg = 0
        for etype in new_g.etypes:
            _, _, dtype = new_g.to_canonical_etype(etype)
            if ntype == dtype:
                deg = deg + new_g.in_degrees(etype=etype)
        norm = 1.0 / deg.float()
        norm[th.isinf(norm)] = 0
        new_g.nodes[ntype].data["norm"] = norm.view(-1, 1).to(device)

    res = []

    # compute k-hop feature
    for hop in range(1, args.R + 1):
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
            if ntype == "paper":
                res.append(old_feat.cpu())
            feat_dict[f"hop_{hop}"] = ntype2feat.pop(ntype).mul_(feat_dict["norm"])

    res.append(new_g.nodes["paper"].data.pop(f"hop_{args.R}").cpu())
    return res



class FeedForwardNet(nn.Module):
    def __init__(self, in_feats, hidden, out_feats, n_layers, dropout):
        super(FeedForwardNet, self).__init__()
        self.layers = nn.ModuleList()
        self.n_layers = n_layers
        if n_layers == 1:
            self.layers.append(nn.Linear(in_feats, out_feats))
        else:
            self.layers.append(nn.Linear(in_feats, hidden))
            for i in range(n_layers - 2):
                self.layers.append(nn.Linear(hidden, hidden))
            self.layers.append(nn.Linear(hidden, out_feats))
        if self.n_layers > 1:
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
            if layer_id < self.n_layers - 1:
                x = self.dropout(self.prelu(x))
        return x


class SIGN(nn.Module):
    def __init__(
        self, in_feats, hidden, out_feats, num_hops, n_layers, dropout, input_drop
    ):
        super(SIGN, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.prelu = nn.PReLU()
        self.inception_ffs = nn.ModuleList()
        self.input_drop = input_drop
        for i in range(num_hops):
            self.inception_ffs.append(
                FeedForwardNet(in_feats, hidden, hidden, n_layers, dropout)
            )
        self.project = FeedForwardNet(
            num_hops * hidden, hidden, out_feats, n_layers, dropout
        )

    def forward(self, feats):
        hidden = []
        for feat, ff in zip(feats, self.inception_ffs):
            if self.input_drop:
                feat = self.dropout(feat)
            hidden.append(ff(feat))
        out = self.project(self.dropout(self.prelu(th.cat(hidden, dim=-1))))
        return th.log_softmax(out, dim=-1)


class WeightedAggregator(nn.Module):
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


