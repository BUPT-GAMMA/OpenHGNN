import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
import dgl
class feature_mapping(nn.Module):
    def __init__(self, g, out_dim, n_feats=None, in_dim=None):
        super(feature_mapping, self).__init__()
        # n_feats & in_dim are used to decide the in_features of nn.Linear
        if (n_feats and in_dim) is None:
            ValueError("Please feed one of the n_feats or in_dim")
        self.encoder = nn.ModuleDict({})
        ntypes = g.ntypes
        for ntype in ntypes:
            if in_dim is None:
                in_dim = g.nodes[ntype].data[n_feats].shape[1]
            self.encoder[ntype] = nn.Linear(in_features=in_dim, out_features=out_dim, bias=True)

    def forward(self, g, feats_in, feats_out):
        h_dict = {}
        for ntype in g.ntypes:
            h = g.nodes[ntype].data[feats_in]
            h_proj = self.encoder[ntype](h)
            h_dict[ntype] = h_proj
        g.ndata[feats_out] = h_dict
        # g.nodes[ntype].data['h_proj'] = h_proj
        return

class NSHE(nn.Module):

    def __init__(self, g, gnn_model, project_dim, emd_dim, context_dim):
        super(NSHE, self).__init__()
        self.gnn_model = gnn_model
        self.norm_emb = True
        # dimension of transform: after projecting, after aggregation, after CE_encoder
        self.project_dim = project_dim
        self.emd_dim = emd_dim
        self.context_dim = context_dim
        # * ================== Project feature Layer==================
        self.feature_proj = feature_mapping(g, out_dim=self.project_dim, n_feats='h')
        # * ================== Neighborhood Agg(gnn_model)==================
        if self.gnn_model == "GCN":
            self.gnn = GraphConv(self.project_dim, self.emd_dim, activation=F.relu)

        # * ================== Context encoder(called CE in the paper)=================
        self.context_encoder = feature_mapping(g, out_dim=self.context_dim, in_dim=self.emd_dim)
        # * ================== NSI Classification================
        in_feats = self.emd_dim + self.context_dim * (len(g.ntypes) - 1)
        self.linear_classifier = feature_mapping(g, out_dim=1, in_dim=in_feats)

    def forward(self, g, ns_samples):
        # * =============== Encode heterogeneous feature ================
        self.feature_proj(g, 'h', 'h_proj')
        g_homo = dgl.to_homogeneous(g, ndata=['h_proj'])
        # * =============== Node Embedding Generation ===================
        h = g_homo.ndata['h_proj']
        h = self.gnn(g_homo, h)
        if self.norm_emb:
            # Independently normalize each dimension
            h = F.normalize(h, p=2, dim=1)
        # Context embedding generation
        g_homo.ndata['x'] = h
        hg_2 = dgl.to_heterogeneous(g_homo, g.ntypes, g.etypes)
        self.context_encoder(hg_2, 'x', 'h_context')
        for ns_instance in ns_samples:
            target = ns_instance
            break
        return h, 1
