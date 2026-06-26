import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from torch.nn import init
from openhgnn.models import BaseModel, register_model
from dgl.utils import expand_as_pair


class HEROHomoEvaluation:
    def __init__(self, embs, labels, train_idx, val_idx, test_idx):
        self.embs = self._to_numpy(embs)
        self.labels = self._to_numpy(labels).reshape(-1)

        self.train_idx = self._to_numpy(train_idx).reshape(-1).astype(np.int64)
        self.val_idx = self._to_numpy(val_idx).reshape(-1).astype(np.int64)
        self.test_idx = self._to_numpy(test_idx).reshape(-1).astype(np.int64)

        self.trX = self.embs[self.train_idx]
        self.trY = self.labels[self.train_idx]
        self.valX = self.embs[self.val_idx]
        self.valY = self.labels[self.val_idx]
        self.tsX = self.embs[self.test_idx]
        self.tsY = self.labels[self.test_idx]

    def _to_numpy(self, x):
        if hasattr(x, "detach"):
            return x.detach().cpu().numpy()
        return x

    def evaluate(self):
        fis, fas = 0.0, 0.0
        for rs in [0, 1, 2, 3, 4]:
            lr = LogisticRegression(max_iter=500, random_state=rs, solver="sag")
            lr.fit(self.trX, self.trY)
            y_pred = lr.predict(self.tsX)
            fis += metrics.f1_score(self.tsY, y_pred, average="micro")
            fas += metrics.f1_score(self.tsY, y_pred, average="macro")
        return fis / 5.0, fas / 5.0


class MLP(nn.Module):
    def __init__(self, dim, dropprob=0.0):
        super(MLP, self).__init__()
        self.net = nn.ModuleList()
        self.dropout = nn.Dropout(dropprob)

        struc = list(dim)
        for i in range(len(struc) - 1):
            self.net.append(nn.Linear(struc[i], struc[i + 1]))

    def forward(self, x):
        for i in range(len(self.net) - 1):
            x = F.relu(self.net[i](x))
            x = self.dropout(x)
        y = self.net[-1](x)
        return y



class GraphConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 norm='both',
                 weight=True,
                 bias=True,
                 activation=None,
                 allow_zero_in_degree=False):
        super(GraphConv, self).__init__()

        if norm not in ('none', 'both', 'right', 'left'):
            raise DGLError(
                'Invalid norm value. Must be either "none", "both", "right" or "left". '
                f'But got "{norm}".'
            )

        self._in_feats = in_feats
        self._out_feats = out_feats
        self._norm = norm
        self._allow_zero_in_degree = allow_zero_in_degree

        if weight:
            self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))
        else:
            self.register_parameter('weight', None)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_feats))
        else:
            self.register_parameter('bias', None)

        self._activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        if self.weight is not None:
            init.xavier_uniform_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, weight=None, edge_weight=None):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError(
                        'There are 0-in-degree nodes in the graph. '
                        'Adding self-loop by `g = dgl.add_self_loop(g)` will resolve this.'
                    )

            aggregate_fn = fn.copy_u('h', 'm')
            if edge_weight is not None:
                assert edge_weight.shape[0] == graph.number_of_edges()
                graph.edata['_edge_weight'] = edge_weight
                aggregate_fn = fn.u_mul_e('h', '_edge_weight', 'm')

            feat_src, feat_dst = expand_as_pair(feat, graph)

            if self._norm in ['left', 'both']:
                degs = graph.out_degrees().float().clamp(min=1)
                if self._norm == 'both':
                    norm = torch.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat_src.dim() - 1)
                norm = torch.reshape(norm, shp)
                feat_src = feat_src * norm

            if weight is not None:
                if self.weight is not None:
                    raise DGLError(
                        'External weight is provided while module already has its own weight.'
                    )
            else:
                weight = self.weight

            if self._in_feats > self._out_feats:
                if weight is not None:
                    feat_src = torch.matmul(feat_src, weight)
                graph.srcdata['h'] = feat_src
                graph.update_all(aggregate_fn, fn.sum(msg='m', out='h'))
                rst = graph.dstdata['h']
            else:
                graph.srcdata['h'] = feat_src
                graph.update_all(aggregate_fn, fn.sum(msg='m', out='h'))
                rst = graph.dstdata['h']
                if weight is not None:
                    rst = torch.matmul(rst, weight)

            if self._norm in ['right', 'both']:
                degs = graph.in_degrees().float().clamp(min=1)
                if self._norm == 'both':
                    norm = torch.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat_dst.dim() - 1)
                norm = torch.reshape(norm, shp)
                rst = rst * norm

            if self.bias is not None:
                rst = rst + self.bias

            if self._activation is not None:
                rst = self._activation(rst)

            return rst


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, activation=None):
        super().__init__()
        self.conv = GraphConv(
            in_dim,
            out_dim,
            norm='both',
            weight=True,
            bias=True,
            activation=activation,
            allow_zero_in_degree=False
        )

    def forward(self, g, feat):
        return self.conv(g, feat)


class SimpleGCN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers=2):
        super().__init__()

        assert n_layers >= 2, "n_layers must be >= 2"

        self.n_layers = n_layers
        self.layers = nn.ModuleList()

        self.layers.append(GCNLayer(in_dim, hid_dim, activation=nn.ReLU()))

        for _ in range(n_layers - 2):
            self.layers.append(GCNLayer(hid_dim, hid_dim, activation=nn.ReLU()))

        self.layers.append(GCNLayer(hid_dim, out_dim, activation=None))

    def forward(self, g, feat):
        h = feat
        for layer in self.layers:
            h = layer(g, h)
        return h


@register_model("HERO_homo")
class HEROHomo(BaseModel):
    @classmethod
    def build_model_from_args(cls, args):
        required_args = [
            "ft_size", "hid_units", "out_ft",
            "beta", "alpha",
            "g_dim", "g_equidim", "p_equidim",
            "gamma", "eta", "lambbda"
        ]
        for key in required_args:
            if not hasattr(args, key):
                raise AttributeError(f"args.{key} is required for HERO_homo.")
        return cls(args)

    def __init__(self, args):
        super(HEROHomo, self).__init__()
        self.args = args

        self.gcn = SimpleGCN(self.args.ft_size, self.args.hid_units, self.args.out_ft)
        self.mlp = MLP([self.args.ft_size, self.args.out_ft])

        self.g_proj = nn.Sequential(
            nn.Linear(self.args.out_ft, self.args.g_dim, bias=False),
            nn.ReLU(inplace=True)
        )
        self.g_1 = nn.Sequential(
            nn.Linear(self.args.out_ft, self.args.g_equidim, bias=False),
            nn.ReLU(inplace=True)
        )
        self.p_1 = nn.Sequential(
            nn.Linear(self.args.g_equidim, self.args.p_equidim, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, graph, features, distance):
        embs_ori = self.gcn(graph, features[0])

        emb_f = self.mlp(features[1])
        coe2 = 1.0 / self.args.beta
        res = torch.mm(emb_f.T, emb_f)
        inv = torch.inverse(torch.eye(emb_f.shape[1], device=emb_f.device) + coe2 * res)
        res = torch.mm(inv, res)
        res = coe2 * emb_f - coe2 * coe2 * torch.mm(emb_f, res)
        tmp = torch.mm(emb_f.T, res)
        part1 = torch.mm(emb_f, tmp)
        part2 = (- self.args.alpha / 2) * torch.mm(distance, res)
        embs_hom = part1 + part2

        return embs_ori, embs_hom

    def get_concat_embeddings(self, graph, features, distance):
        embs_ori, embs_hom = self.forward(graph, features, distance)
        return torch.cat([embs_ori, embs_hom], dim=1)

    def compute_loss(self, graph, features, distance):
        embs_ori, embs_hom = self.forward(graph, features, distance)

        embs_P1 = self.g_proj(embs_ori)
        embs_P2 = self.g_proj(embs_hom)

        intra_c = embs_P1.T @ embs_P1.contiguous()
        intra_c = torch.exp(F.normalize(intra_c, p=2, dim=1)).sum()
        loss_uni = torch.log(intra_c).mean()

        intra_c_2 = embs_P2.T @ embs_P2.contiguous()
        intra_c_2 = torch.exp(F.normalize(intra_c_2, p=2, dim=1)).sum()
        loss_uni += torch.log(intra_c_2).mean()

        inter_c = embs_P1.T @ embs_P2
        inter_c = F.normalize(inter_c, p=2, dim=1)
        loss_inv = -torch.diagonal(inter_c).sum()

        embs_Q2 = self.g_1(embs_ori)
        embs_Q1 = self.g_1(embs_hom)
        embs_Q1_trans = self.p_1(embs_Q1)

        inter_c = embs_Q1_trans.T @ embs_Q2
        inter_c = F.normalize(inter_c, p=2, dim=1)
        loss_spe_inv = -torch.diagonal(inter_c).sum()

        inter_c = embs_Q1_trans.T @ embs_Q1
        inter_c = F.normalize(inter_c, p=2, dim=1)
        loss_spe_nontrival_1 = torch.diagonal(inter_c).sum()

        inter_c_1 = embs_Q1_trans.T @ embs_P2
        inter_c_1 = F.normalize(inter_c_1, p=2, dim=1)
        loss_spe_nontrival_2 = torch.diagonal(inter_c_1).sum()

        loss_consistency = loss_inv + self.args.gamma * loss_uni
        loss_specificity = loss_spe_inv + self.args.eta * (
            loss_spe_nontrival_1 + loss_spe_nontrival_2
        )
        loss = loss_consistency + self.args.lambbda * loss_specificity

        return {
            "loss": loss,
            "loss_consistency": loss_consistency.detach(),
            "loss_specificity": loss_specificity.detach(),
        }