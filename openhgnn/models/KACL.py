import torch

from openhgnn.models.base_model import BaseModel
from openhgnn.models import register_model
import torch.nn.functional as F
import torch as th
from torch import nn
import numpy as np
from dgl import function as fn
from dgl.nn.pytorch import edge_softmax
from dgl._ffi.base import DGLError
from dgl.nn.pytorch.utils import Identity
from dgl.utils import expand_as_pair


@register_model('KACL')
class KACL(BaseModel):
    def build_model_from_args(args, hg):
        g = hg._g
        kg = hg._kg
        n_entity = kg.number_of_nodes('entity')
        n_user = g.number_of_nodes('user')
        n_item = g.number_of_nodes('item')
        meta_graph = kg.metagraph()
        n_relation = len(meta_graph.edges())
        weight_size = args.layer_size
        num_layers = len(weight_size) - 2
        heads = [args.head] * num_layers + [1]

        return KACL(args, n_entity, n_relation, weight_size[-2], weight_size[-1], num_layers, heads, F.elu, args.feat_drop, args.attn_drop,
                   args.negative_slope, n_user, n_item, args.residual, pretrain=None)

    def __init__(self, args, num_entity, num_etypes, num_hidden, num_classes, num_layers,
                 heads, activation, feat_drop, attn_drop, negative_slope, user_size, item_size, residual,
                 pretrain=None):
        super(KACL, self).__init__()
        if args.gpu == -1:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(args.gpu)
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.sub_gat_layers = nn.ModuleList()
        self.kg_gat_layers = nn.ModuleList()

        self.drop_learner = False

        self.activation = activation

        self.cfe_size = args.embed_size
        self.kge_size = args.kge_size
        self.edge_dim = self.kge_size
        self.cl_alpha = args.cl_alpha
        alpha = args.alpha
        cl_dim = self.cfe_size

        tau = args.temperature
        self.weight_decay = args.weight_decay
        self.kg_weight_decay = args.kg_weight_decay
        self.batch_size = args.batch_size

        if pretrain is not None:
            user_embed = pretrain['user_embed']
            item_embed = pretrain['item_embed']
            self.user_size = user_embed.shape[0]
            self.item_size = item_embed.shape[0]
            self.ret_num = self.user_size + self.item_size
            self.embed = nn.Parameter(th.zeros((self.ret_num, self.cfe_size)))
            self.cl_embed = nn.Parameter(th.zeros((self.ret_num, self.cfe_size)))
            nn.init.xavier_normal_(self.embed, gain=1.414)
            nn.init.xavier_normal_(self.cl_embed, gain=1.414)
            self.ini = th.FloatTensor(np.concatenate([user_embed, item_embed], axis=0))
        else:
            self.user_size = user_size
            self.item_size = item_size
            self.ret_num = self.user_size + self.item_size
            self.embed = nn.Parameter(th.zeros((self.ret_num, self.cfe_size)))
            self.cl_embed = nn.Parameter(th.zeros((self.ret_num, self.cfe_size)))
            nn.init.xavier_normal_(self.embed, gain=1.414)
            nn.init.xavier_normal_(self.cl_embed, gain=1.414)

        self.kg_embed = nn.Parameter(th.zeros((num_entity, args.kge_size)))
        self.user_embed = nn.Parameter(th.zeros((self.user_size, args.kge_size + 48)))

        nn.init.xavier_normal_(self.kg_embed, gain=1.414)
        # nn.init.xavier_normal_(self.user_embed, gain=1.414)
        # input projection (no residual)
        self.gat_layers.append(myGATConv(self.cfe_size, num_hidden, heads[0],
                                         feat_drop, attn_drop, negative_slope, False, self.activation, bias=True,
                                         alpha=alpha))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(myGATConv(num_hidden * heads[l - 1],
                                             num_hidden, heads[l],
                                             feat_drop, attn_drop, negative_slope, residual, self.activation, bias=True,
                                             alpha=alpha))
        # output projection
        self.gat_layers.append(myGATConv(num_hidden * heads[-2],
                                         num_classes, heads[-1],
                                         feat_drop, attn_drop, negative_slope, residual, None, bias=True, alpha=alpha))

        # input projection (no residual)
        self.sub_gat_layers.append(myGATConv(self.cfe_size, num_hidden, heads[0],
                                             feat_drop, attn_drop, negative_slope, False, self.activation, bias=True,
                                             alpha=alpha))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.sub_gat_layers.append(myGATConv(num_hidden * heads[l - 1],
                                                 num_hidden, heads[l],
                                                 feat_drop, attn_drop, negative_slope, residual, self.activation,
                                                 bias=True, alpha=alpha))
        # output projection
        self.sub_gat_layers.append(myGATConv(num_hidden * heads[-2],
                                             num_classes, heads[-1],
                                             feat_drop, attn_drop, negative_slope, residual, None, bias=True,
                                             alpha=alpha))

        # input projection (no residual)
        self.kg_gat_layers.append(myGATConv(self.kge_size, num_hidden, heads[0],
                                            feat_drop, attn_drop, negative_slope, False, self.activation, bias=True,
                                            alpha=alpha))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.kg_gat_layers.append(myGATConv(num_hidden * heads[l - 1],
                                                num_hidden, heads[l],
                                                feat_drop, attn_drop, negative_slope, residual, self.activation,
                                                bias=True, alpha=alpha))
        # output projection
        self.kg_gat_layers.append(myGATConv(num_hidden * heads[-2],
                                            num_classes, heads[-1],
                                            feat_drop, attn_drop, negative_slope, residual, None, bias=True,
                                            alpha=alpha))
        self.epsilon = th.FloatTensor([1e-12]).to(self.device)
        # self.epsilon = th.FloatTensor([1e-12])
        self.contrast = Contrast_2view(self.cfe_size + 48, self.kge_size + 48, cl_dim, tau, args.batch_size_cl, self.device)
        self.decoder = DistMult(num_etypes, self.kge_size + 48)
        self.learner1 = DropLearner(self.cfe_size, self.cfe_size)
        self.learner2 = DropLearner(self.kge_size, self.kge_size, self.edge_dim)
        self.cf_edge_weight = None
        self.kg_edge_weight = None

    def calc_ui_emb(self, g):
        all_embed = []
        h = self.embed
        tmp = (h / (th.max(th.norm(h, dim=1, keepdim=True), self.epsilon)))
        all_embed.append(tmp)
        res_attn = None
        for l in range(self.num_layers):
            h, res_attn = self.gat_layers[l](g, h, res_attn=res_attn)
            h = h.flatten(1)
            tmp = (h / (th.max(th.norm(h, dim=1, keepdim=True), self.epsilon)))
            all_embed.append(tmp)
        # output projection
        logits, _ = self.gat_layers[-1](g, h, res_attn=res_attn)
        logits = logits.mean(1)
        all_embed.append(logits / (th.max(th.norm(logits, dim=1, keepdim=True), self.epsilon)))
        all_embed = th.cat(all_embed, 1)
        return all_embed

    def calc_cl_emb(self, g, drop_learn=False):
        all_embed = []
        h = self.cl_embed
        tmp = (h / (th.max(th.norm(h, dim=1, keepdim=True), self.epsilon)))
        edge_weight = None
        reg = 0
        if drop_learn:
            reg, edge_weight = self.learner1(tmp, g, temperature=0.7)
            self.cf_edge_weight = edge_weight.detach()
        else:
            edge_weight = self.cf_edge_weight
        all_embed.append(tmp)
        res_attn = None
        for l in range(self.num_layers):
            h, res_attn = self.sub_gat_layers[l](g, h, res_attn=res_attn, edge_weight=edge_weight)
            h = h.flatten(1)
            tmp = (h / (th.max(th.norm(h, dim=1, keepdim=True), self.epsilon)))
            all_embed.append(tmp)
        # output projection
        logits, _ = self.sub_gat_layers[-1](g, h, res_attn=res_attn, edge_weight=edge_weight)
        logits = logits.mean(1)
        all_embed.append(logits / (th.max(th.norm(logits, dim=1, keepdim=True), self.epsilon)))
        all_embed = th.cat(all_embed, 1)
        if drop_learn:
            return all_embed, reg
        else:
            return all_embed

    def calc_kg_emb(self, g, drop_learn=False):
        all_embed = []
        h = self.kg_embed
        tmp = (h / (th.max(th.norm(h, dim=1, keepdim=True), self.epsilon)))
        edge_weight = None
        reg = 0
        if drop_learn:
            reg, edge_weight = self.learner2(tmp, g, temperature=0.7)
            self.kg_edge_weight = edge_weight.detach()
        else:
            edge_weight = self.kg_edge_weight
        all_embed.append(tmp)
        res_attn = None
        for l in range(self.num_layers):
            h, res_attn = self.kg_gat_layers[l](g, h, res_attn=res_attn, edge_weight=edge_weight)
            h = h.flatten(1)
            tmp = (h / (th.max(th.norm(h, dim=1, keepdim=True), self.epsilon)))
            all_embed.append(tmp)
        # output projection
        logits, _ = self.kg_gat_layers[-1](g, h, res_attn=res_attn, edge_weight=edge_weight)
        logits = logits.mean(1)
        all_embed.append(logits / (th.max(th.norm(logits, dim=1, keepdim=True), self.epsilon)))
        all_embed = th.cat(all_embed, 1)
        if drop_learn:
            return all_embed, reg
        else:
            return all_embed

    def calc_cf_loss(self, g, sub_g, kg, user_id, pos_item, neg_item):  # (self, g, user_id, item_id, pos_mat):
        embedding_cf = self.calc_ui_emb(g)
        # embedding_cf = self.calc_cl_emb(g)
        # reg_cl, reg_kg = 0, 0

        reg_cl, reg_kg = 0, 0
        embedding_cl, reg_cl = self.calc_cl_emb(sub_g, True)
        # embedding_cl = self.calc_cl_emb(sub_g, False)

        # embedding_kg = self.calc_kg_emb(kg, e_feat)[:self.item_size]
        embedding_kg, reg_kg = self.calc_kg_emb(kg, True)
        # embedding_kg = self.calc_kg_emb(kg, False)

        embedding_kg = th.cat([self.user_embed, embedding_kg[:self.item_size]], 0)

        embedding = th.cat([embedding_cf, embedding_cl, embedding_kg], 1)

        # embedding = th.cat([embedding_cf, self.ini], 1)
        # embedding = th.cat([embedding_cl, embedding_kg, self.ini], 1)
        # embedding = th.cat([embedding_cf, embedding_kg, self.ini], 1)

        u_emb = embedding[user_id]
        p_emb = embedding[pos_item]
        n_emb = embedding[neg_item]
        pos_scores = (u_emb * p_emb).sum(dim=1)
        neg_scores = (u_emb * n_emb).sum(dim=1)
        base_loss = F.softplus(neg_scores - pos_scores).mean()
        reg_loss = self.weight_decay * (
                    (u_emb * u_emb).sum() / 2 + (p_emb * p_emb).sum() / 2 + (n_emb * n_emb).sum() / 2) / self.batch_size
        loss = base_loss + reg_loss
        return loss, reg_cl, reg_kg

    def calc_cl_loss(self, g, kg, item):
        embedding = self.calc_cl_emb(g)
        # kg_embedding = self.calc_kg_emb(kg, e_feat)
        kg_embedding = self.calc_kg_emb(kg)
        kg_emb = kg_embedding[item]
        item = item + np.array([self.user_size])
        cf_emb = embedding[item]
        cl_loss = self.contrast(cf_emb, kg_emb)
        loss = self.cl_alpha * cl_loss
        return loss

    def calc_kg_loss(self, g, h, r, pos_t, neg_t):
        # embedding = self.calc_kg_emb(g, e_feat)
        weight = False
        embedding = self.calc_kg_emb(g)

        h_emb = embedding[h]
        pos_t_emb = embedding[pos_t]
        neg_t_emb = embedding[neg_t]

        pos_score = self.decoder(h_emb, pos_t_emb, r)
        neg_score = self.decoder(h_emb, neg_t_emb, r)
        aug_edge_weight = 1
        if weight:
            emb = self.kg_embed
            emb = (emb / (th.max(th.norm(emb, dim=1, keepdim=True), self.epsilon)))
            _, aug_edge_weight = self.learner2.get_weight(emb[h], emb[pos_t], temperature=0.7)
        # loss
        base_loss = (aug_edge_weight * F.softplus(-neg_score + pos_score)).mean()
        return base_loss

    def forward(self, mode, *input):
        if mode == "cf":
            return self.calc_cf_loss(*input)
        elif mode == "kg":
            return self.calc_kg_loss(*input)
        elif mode == "cl":
            return self.calc_cl_loss(*input)
        elif mode == "test":
            # g, kg, e_feat = input
            g, kg = input
            self.kg_edge_weight = None
            self.cf_edge_weight = None
            embedding_cf = self.calc_ui_emb(g)
            # embedding_cf = self.calc_cl_emb(g)

            embedding_cl = self.calc_cl_emb(g)
            embedding_kg = self.calc_kg_emb(kg)

            embedding_kg = th.cat([self.user_embed, embedding_kg[:self.item_size]], 0)
            embedding = th.cat([embedding_cf, embedding_cl, embedding_kg], 1)

            # embedding = th.cat([embedding_cf, self.ini], 1)

            return embedding


class Contrast_2view(nn.Module):
    def __init__(self, cf_dim, kg_dim, hidden_dim, tau, cl_size, device):
        super(Contrast_2view, self).__init__()
        self.projcf = nn.Sequential(
            nn.Linear(cf_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.projkg = nn.Sequential(
            nn.Linear(kg_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.pos = th.eye(cl_size).to(device)
        # self.pos = th.eye(cl_size)
        self.tau = tau
        for model in self.projcf:
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight, gain=1.414)
        for model in self.projkg:
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight, gain=1.414)

    def sim(self, z1, z2):
        z1_norm = th.norm(z1, dim=-1, keepdim=True)
        z2_norm = th.norm(z2, dim=-1, keepdim=True)
        dot_numerator = th.mm(z1, z2.t())
        dot_denominator = th.mm(z1_norm, z2_norm.t())
        sim_matrix = th.exp(dot_numerator / dot_denominator / self.tau)
        sim_matrix = sim_matrix / (th.sum(sim_matrix, dim=1).view(-1, 1) + 1e-8)
        assert sim_matrix.size(0) == sim_matrix.size(1)
        lori_mp = -th.log(sim_matrix.mul(self.pos).sum(dim=-1)).mean()
        return lori_mp

    def forward(self, z1, z2):
        multi_loss = False
        z1_proj = self.projcf(z1)
        z2_proj = self.projkg(z2)
        if multi_loss:
            loss1 = self.sim(z1_proj, z2_proj)
            loss2 = self.sim(z1_proj, z1_proj)
            loss3 = self.sim(z2_proj, z2_proj)
            return (loss1 + loss2 + loss3) / 3
        else:
            return self.sim(z1_proj, z2_proj)


class DistMult(nn.Module):
    def __init__(self, num_rel, dim):
        super(DistMult, self).__init__()
        self.W = nn.Parameter(th.FloatTensor(size=(num_rel, dim, dim)))
        nn.init.xavier_normal_(self.W, gain=1.414)

    def forward(self, left_emb, right_emb, r_id):
        thW = self.W[r_id]
        left_emb = th.unsqueeze(left_emb, 1)
        right_emb = th.unsqueeze(right_emb, 2)
        return th.bmm(th.bmm(left_emb, thW), right_emb).squeeze()


class DropLearner(nn.Module):
    def __init__(self, node_dim, edge_dim=None, mlp_edge_model_dim=64):
        super(DropLearner, self).__init__()

        self.mlp_src = nn.Sequential(
            nn.Linear(node_dim, mlp_edge_model_dim),
            nn.ReLU(),
            nn.Linear(mlp_edge_model_dim, 1)
        )
        self.mlp_dst = nn.Sequential(
            nn.Linear(node_dim, mlp_edge_model_dim),
            nn.ReLU(),
            nn.Linear(mlp_edge_model_dim, 1)
        )
        self.mlp_con = nn.Sequential(
            nn.Linear(node_dim, mlp_edge_model_dim),
            nn.ReLU(),
            nn.Linear(mlp_edge_model_dim, 1)
        )

        self.concat = False

        if edge_dim is not None:
            self.mlp_edge = nn.Sequential(
                nn.Linear(edge_dim, mlp_edge_model_dim),
                nn.ReLU(),
                nn.Linear(mlp_edge_model_dim, 1)
            )
        else:
            self.mlp_edge = None
        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def get_weight(self, head_emb, tail_emb, temperature=0.5, relation_emb=None, edge_type=None):
        if self.concat:
            weight = self.mlp_con(head_emb + tail_emb)
            w_src = self.mlp_src(head_emb)
            w_dst = self.mlp_dst(tail_emb)
            weight += w_src + w_dst
        else:
            w_src = self.mlp_src(head_emb)
            w_dst = self.mlp_dst(tail_emb)
            weight = w_src + w_dst
        if relation_emb is not None and self.mlp_edge is not None:
            e_weight = self.mlp_edge(relation_emb)
            weight += e_weight
        weight = weight.squeeze()
        bias = 0.0 + 0.0001  # If bias is 0, we run into problems
        eps = (bias - (1 - bias)) * th.rand(weight.size()) + (1 - bias)
        gate_inputs = th.log(eps) - th.log(1 - eps)
        gate_inputs = gate_inputs.to(head_emb.device)
        gate_inputs = (gate_inputs + weight) / temperature
        aug_edge_weight = th.sigmoid(gate_inputs).squeeze()
        edge_drop_out_prob = 1 - aug_edge_weight
        reg = edge_drop_out_prob.mean()
        # print(aug_edge_weight.size())
        return reg.detach(), aug_edge_weight.detach()

    def forward(self, node_emb, graph, temperature=0.5, relation_emb=None, edge_type=None):
        if self.concat:
            w_con = node_emb
            graph.srcdata.update({'in': w_con})
            graph.apply_edges(fn.u_add_v('in', 'in', 'con'))
            n_weight = graph.edata.pop('con')
            weight = self.mlp_con(n_weight)
            w_src = self.mlp_src(node_emb)
            w_dst = self.mlp_dst(node_emb)
            graph.srcdata.update({'inl': w_src})
            graph.dstdata.update({'inr': w_dst})
            graph.apply_edges(fn.u_add_v('inl', 'inr', 'ine'))
            weight += graph.edata.pop('ine')
            # print(weight.size())
        else:
            w_src = self.mlp_src(node_emb)
            w_dst = self.mlp_dst(node_emb)
            graph.srcdata.update({'inl': w_src})
            graph.dstdata.update({'inr': w_dst})
            graph.apply_edges(fn.u_add_v('inl', 'inr', 'ine'))
            n_weight = graph.edata.pop('ine')
            weight = n_weight
        if relation_emb is not None and self.mlp_edge is not None:
            w_edge = self.mlp_edge(relation_emb)
            graph.edata.update({'ee': w_edge})
            e_weight = graph.edata.pop('ee')
            weight += e_weight
        weight = weight.squeeze()
        bias = 0.0 + 0.0001  # If bias is 0, we run into problems
        eps = (bias - (1 - bias)) * th.rand(weight.size()) + (1 - bias)
        gate_inputs = th.log(eps) - th.log(1 - eps)
        gate_inputs = gate_inputs.to(node_emb.device)
        gate_inputs = (gate_inputs + weight) / temperature
        aug_edge_weight = th.sigmoid(gate_inputs).squeeze()
        edge_drop_out_prob = 1 - aug_edge_weight
        reg = edge_drop_out_prob.mean()
        aug_edge_weight = aug_edge_weight.unsqueeze(-1).unsqueeze(-1)
        # print(aug_edge_weight.size())
        return reg, aug_edge_weight


# pylint: enable=W0235
class myGATConv(nn.Module):
    """
    Adapted from
    https://docs.dgl.ai/_modules/dgl/nn/pytorch/conv/gatconv.html#GATConv
    """

    def __init__(self, in_feats, out_feats, num_heads, feat_drop=0., attn_drop=0.,
                 negative_slope=0.2, residual=False, activation=None, allow_zero_in_degree=False, bias=False, alpha=0.):
        super(myGATConv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation
        self.bias = bias
        if bias:
            self.bias_param = nn.Parameter(th.zeros((1, num_heads, out_feats)))
        self.alpha = alpha

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, res_attn=None, edge_weight=None):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')
            if isinstance(feat, tuple):
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, 'fc_src'):
                    self.fc_src, self.fc_dst = self.fc, self.fc
                feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
                feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
            else:
                h_src = h_dst = self.feat_drop(feat)
                feat_src = feat_dst = self.fc(h_src).view(
                    -1, self._num_heads, self._out_feats)
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({'ft': feat_src, 'el': el})
            graph.dstdata.update({'er': er})
            graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
            e = self.leaky_relu(graph.edata.pop('e'))
            # compute softmax
            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
            if edge_weight is not None:
                graph.edata['a'] = graph.edata['a'] * edge_weight
            if res_attn is not None:
                graph.edata['a'] = graph.edata['a'] * (1 - self.alpha) + res_attn * self.alpha
            # message passing
            graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                             fn.sum('m', 'ft'))
            rst = graph.dstdata['ft']
            # residual
            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
                rst = rst + resval
            if self.bias:
                rst = rst + self.bias_param
            # activation
            if self.activation:
                rst = self.activation(rst)
            return rst, graph.edata.pop('a').detach()
