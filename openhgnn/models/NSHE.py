import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
import dgl
import dgl.nn as dglnn
from . import BaseModel, register_model
from ..layers.HeteroLinear import HeteroMLPLayer, HeteroLinearLayer

'''
In paper repo performance		
ACM<Classification>     Micro-F1 = 0.8412, Macro-F1 = 0.8327
Run author code		
ACM-Deepwalk<Classification>     Micro-F1 = 0.8408, Macro-F1 = 0.8411
OpenHGNN-DGL
ACM<Classification>     Micro-F1 = 0.8495, Macro-F1 = 0.8478


In paper repo performance		
IMDB<Classification>     Micro-F1 = 0.5921, Macro-F1 = 0.5835
Run author code		
IMDB-Deepwalk<Classification>     Micro-F1 = 0.5938, Macro-F1 = 0.5804
OpenHGNN-DGL
IMDB<Classification>     Micro-F1 = 0.6209, Macro-F1 = 0.6053
'''


@register_model('NSHE')
class NSHE(BaseModel):
    r"""
    NSHE[IJCAI2020]
    Network Schema Preserving Heterogeneous Information Network Embedding
    `Paper Link <http://www.shichuan.org/doc/87.pdf>`
    `Code Link https://github.com/Andy-Border/NSHE`

    """
    @classmethod
    def build_model_from_args(cls, args, hg):
        return cls(hg, 'GCN', project_dim=args.dim_size['project'],
                   emd_dim=args.dim_size['emd'], context_dim=args.dim_size['context'],
                   num_heads=args.num_heads, dropout=args.dropout)

    def __init__(self, g, gnn_model, project_dim, emd_dim, context_dim, num_heads, dropout):
        super(NSHE, self).__init__()
        self.gnn_model = gnn_model
        self.norm_emb = True
        # dimension of transform: after projecting, after aggregation, after CE_encoder
        self.project_dim = project_dim
        self.emd_dim = emd_dim
        self.context_dim = context_dim
        self.num_heads = num_heads
        self.dropout = dropout

        # * ================== encoder config==================
        linear_dict1 = {}
        linear_dict2 = {}
        linear_dict3 = {}
        cla_dim = self.emd_dim + self.context_dim * (len(g.ntypes) - 1)
        for ntype in g.ntypes:
            in_dim = g.nodes[ntype].data['h'].shape[1]
            linear_dict1[ntype] = (in_dim, self.project_dim)
            linear_dict2[ntype] = (self.emd_dim, self.context_dim)
            linear_dict3[ntype] = (cla_dim, 1)
        # * ================== Project feature Layer==================
        self.feature_proj = HeteroLinearLayer(linear_dict1, has_l2norm=False, has_bn=False)
        # * ================== Neighborhood Agg(gnn_model)==================
        if self.gnn_model == "GCN":
            self.gnn1 = GraphConv(self.project_dim, self.emd_dim, norm="none", activation=F.relu)
            self.gnn2 = GraphConv(self.emd_dim, self.emd_dim, norm="none", activation=None)
        elif self.gnn_model == "GAT":
            self.gnn1 = GraphConv(self.project_dim, self.emd_dim, activation=F.relu)
            self.gnn2 = GraphConv(self.emd_dim, self.emd_dim, activation=None)

        # * ================== Context encoder(called CE in the paper)=================
        self.context_encoder = HeteroLinearLayer(linear_dict2, has_l2norm=False, has_bn=False)
        # * ================== NSI Classification================
        self.linear_classifier = HeteroMLPLayer(linear_dict3, has_l2norm=False, has_bn=False)

    def forward(self, hg, h):
        with hg.local_scope():
            # * =============== Encode heterogeneous feature ================
            h_dict = self.feature_proj(h)
            hg.ndata['h_proj'] = h_dict
            g_homo = dgl.to_homogeneous(hg, ndata=['h_proj'])
            # * =============== Node Embedding Generation ===================
            h = g_homo.ndata['h_proj']
            #h = self.gnn1(g_homo, h)
            h = self.gnn2(g_homo, h)
            if self.norm_emb:
                # Independently normalize each dimension
                h = F.normalize(h, p=2, dim=1)
            # Context embedding generation
            # g_homo.ndata['h'] = h
            emd = self.h2dict(h, h_dict)
            #hg.ndata['h'] = emd

        return emd, h

    def h2dict(self, h, hdict):
        pre = 0
        for i, value in hdict.items():
            hdict[i] = h[pre:value.shape[0]+pre]
            pre += value.shape[0]
        return hdict


class NSHELayer(nn.Module):
    def __init__(self, in_feat, out_feat, num_heads, rel_names, activation=None, dropout=0.0):
        super(NSHELayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.rel_names = rel_names
        self.num_heads = num_heads
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

        # 定义异构图卷积
        self.conv = dgl.nn.HeteroGraphConv({
            rel: dgl.nn.pytorch.GATConv(in_feat, out_feat // num_heads, num_heads)
            for rel in rel_names
        })

        # 自环权重
        self.loop_weight = nn.Parameter(th.Tensor(in_feat, out_feat))
        nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain('relu'))
        # 偏置
        self.h_bias = nn.Parameter(th.Tensor(out_feat))
        nn.init.zeros_(self.h_bias)

    def forward(self, g, inputs):
        g = g.local_var()

        if g.is_block:
            inputs_src = inputs
            inputs_dst = {k: v[:g.number_of_dst_nodes(k)] for k, v in inputs.items()}
        else:
            inputs_src = inputs_dst = inputs

        hs = self.conv(g, inputs_src)

        def _apply(ntype, h):
            h = h + th.matmul(inputs_dst[ntype], self.loop_weight)
            h = h + self.h_bias
            if self.activation:
                h = self.activation(h)
            return self.dropout(h)

        return {ntype: _apply(ntype, h) for ntype, h in hs.items()}
