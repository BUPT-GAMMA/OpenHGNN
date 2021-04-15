"""
NSHE[IJCAI2020]
Network Schema Preserving Heterogeneous Information Network Embedding
Paper: http://www.shichuan.org/doc/87.pdf
Code: https://github.com/Andy-Border/NSHE

We use two dataset acm imdb

"""
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
import dgl
from . import BaseModel, register_model

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
    @classmethod
    def build_model_from_args(cls, args, hg):
        return cls(hg, 'GCN', project_dim=args.dim_size['project'],
                 emd_dim=args.dim_size['emd'], context_dim=args.dim_size['context'])

    def __init__(self, g, gnn_model, project_dim, emd_dim, context_dim):
        super(NSHE, self).__init__()
        self.gnn_model = gnn_model
        self.norm_emb = True
        # dimension of transform: after projecting, after aggregation, after CE_encoder
        self.project_dim = project_dim
        self.emd_dim = emd_dim
        self.context_dim = context_dim
        # * ================== encoder config==================
        linear_list1 = []
        linear_list2 = []
        linear_list3 = []
        cla_dim = self.emd_dim + self.context_dim * (len(g.ntypes) - 1)
        for ntype in g.ntypes:
            in_dim = g.nodes[ntype].data['h'].shape[1]
            linear_list1.append((ntype, in_dim, self.project_dim))
            linear_list2.append((ntype, self.emd_dim, self.context_dim))
            linear_list3.append((ntype, cla_dim, 1))
        # * ================== Project feature Layer==================
        self.feature_proj = hetero_linear(linear_list1)
        # * ================== Neighborhood Agg(gnn_model)==================
        if self.gnn_model == "GCN":
            self.gnn1 = GraphConv(self.project_dim, self.emd_dim, norm="none", activation=F.relu)
            self.gnn2 = GraphConv(self.emd_dim, self.emd_dim, norm="none", activation=None)
        elif self.gnn_model == "GAT":
            self.gnn1 = GraphConv(self.project_dim, self.emd_dim, activation=F.relu, )
            self.gnn2 = GraphConv(self.emd_dim, self.emd_dim, activation=None)

        # * ================== Context encoder(called CE in the paper)=================
        self.context_encoder = hetero_linear(linear_list2)
        # * ================== NSI Classification================
        self.linear_classifier = multi_2Linear(linear_list3)

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
            hg.ndata['h'] = emd

        return emd

    def full_train(self, hg):
        h = hg.ndata['h']
        x = self.__call__(hg, h)
        neg_edges, ns_samples = get_epoch_samples(g, epoch, config.dataset, config.num_ns_neg, config.device)
        h_context = self.context_encoder(x)
        p_list = self.pre_ns(ns_samples, x, h_context, g.ntypes)
        x = th.sigmoid(th.cat([p for p in p_list])).flatten()

        node_emb, ns_prediction, eva_h = model(g, ns_samples)
        # compute loss
        pos_edges = g_homo.edges()
        pairwise_loss = cal_node_pairwise_loss(node_emb, pos_edges, neg_edges)
        ns_label = th.cat([ns['label'] for ns in ns_samples]).type(th.float32).to(config.device)
        cla_loss = th.nn.BCELoss(ns_prediction, ns_label)
        loss = pairwise_loss + cla_loss * config.beta
        return loss

    def pre_ns(self, ns_samples, h, h_context, ntypes):
        p_list = []
        for ns_type in ns_samples:
            target = ns_type['target_type']
            index_h = ns_type[target]
            h_tar_type = h[target]
            h_tar = h_tar_type[index_h]
            for type in ntypes:
                if type != target:
                    index_h = ns_type[type]
                    h_con_type = h_context[type]
                    h_con = h_con_type[index_h]
                    h_tar = th.cat((h_tar, h_con), dim=1)
            p = self.linear_classifier(target, h_tar)
            p_list.append(p)
        return p_list

    def h2dict(self, h, hdict):
        pre = 0
        for i, value in hdict.items():
            hdict[i] = h[pre:value.shape[0]+pre]
            pre += value.shape[0]
        return hdict
    @staticmethod
    def cal_node_pairwise_loss(node_emd, edge, neg_edge):
        # cross entropy loss from LINE
        # pos loss
        inner_product = cal_inner_product(node_emd, edge)
        pos_loss = - th.mean(F.logsigmoid(inner_product))
        # neg loss
        inner_product = cal_inner_product(node_emd, neg_edge)
        neg_loss = - th.mean(F.logsigmoid(-1 * inner_product))
        loss = pos_loss + neg_loss
        return loss

    @staticmethod
    def cal_inner_product(node_emd, edge):
        emb_u_i = node_emd[edge[0]]
        emb_u_j = node_emd[edge[1]]
        inner_product = th.sum(emb_u_i * emb_u_j, dim=1)
        return inner_product


class multi_Linear(nn.Module):
    def __init__(self, linear_list, bias=False):
        super(multi_Linear, self).__init__()
        self.encoder = nn.ModuleDict({})
        for linear in linear_list:
            self.encoder[linear[0]] = nn.Linear(in_features=linear[1], out_features=linear[2], bias=bias)

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
        for ntype, h in h_dict.items():
            h = self.encoder(ntype, h)
            h_dict[ntype] = h
        return h_dict
