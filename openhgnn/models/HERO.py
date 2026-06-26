import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from openhgnn.models import BaseModel, register_model
from ..layers import FullyConnect, Discriminator, Linear_layer, SemanticAttention
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
VERY_SMALL_NUMBER = 1e-12
INF = 1e20


class HEROEvaluation:

    def __init__(self, embs, labels, args, train_idx=None, val_idx=None, test_idx=None):
        self.embs = self._to_numpy(embs)
        self.args = args

        if train_idx is not None and val_idx is not None and test_idx is not None:
            labels = self._to_numpy(labels).reshape(-1)

            train_idx = self._to_numpy(train_idx).reshape(-1).astype(np.int64)
            val_idx = self._to_numpy(val_idx).reshape(-1).astype(np.int64)
            test_idx = self._to_numpy(test_idx).reshape(-1).astype(np.int64)

            self.trX, self.trY = self.embs[train_idx], np.array(labels[train_idx])
            self.valX, self.valY = self.embs[val_idx], np.array(labels[val_idx])
            self.tsX, self.tsY = self.embs[test_idx], np.array(labels[test_idx])

            # 过滤无效标签（如 -1）
            tr_mask = self.trY >= 0
            val_mask = self.valY >= 0
            ts_mask = self.tsY >= 0

            self.trX, self.trY = self.trX[tr_mask], self.trY[tr_mask]
            self.valX, self.valY = self.valX[val_mask], self.valY[val_mask]
            self.tsX, self.tsY = self.tsX[ts_mask], self.tsY[ts_mask]

            valid_all = labels[labels >= 0]
            if len(valid_all) == 0:
                raise ValueError("No valid labels found for HERO evaluation.")
            self.n_label = int(len(np.unique(valid_all)))


    def _to_numpy(self, x):
        if x is None:
            return None
        if hasattr(x, "detach"):
            return x.detach().cpu().numpy()
        if isinstance(x, np.ndarray):
            return x
        return np.array(x)

    def evaluate(self):
        if len(self.trX) == 0 or len(self.tsX) == 0:
            raise ValueError("Train or test split is empty in HEROEvaluation.")
        fis, fas = 0.0, 0.0
        for rs in [0, 1, 2, 3, 4]:
            lr = LogisticRegression(max_iter=500,random_state=rs,solver='sag')
            lr.fit(self.trX, self.trY)
            y_pred = lr.predict(self.tsX)
            f1_micro = metrics.f1_score(self.tsY, y_pred, average='micro')
            f1_macro = metrics.f1_score(self.tsY, y_pred, average='macro')
            fis += f1_micro
            fas += f1_macro
        fis /= 5.0
        fas /= 5.0
        return fis, fas


class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()
    def forward(self, seq):
        return torch.mean(seq, dim=1)

class MLP(nn.Module):
    def __init__(self, dim, dropprob=0.0):
        super(MLP, self).__init__()
        self.net = nn.ModuleList()
        self.dropout = nn.Dropout(dropprob)

        # struc = list(dim)
        struc = []
        for i in range(len(dim)):
            struc.append(dim[i])
        for i in range(len(struc) - 1):
            self.net.append(nn.Linear(struc[i], struc[i + 1]))

    def forward(self, x):
        for i in range(len(self.net) - 1):
            x = F.relu(self.net[i](x))
            x = self.dropout(x)
        y = self.net[-1](x)
        return y


@register_model('HERO')
class HERO(BaseModel):

    @classmethod
    def build_model_from_args(cls, args, hg):
        target_ntype = getattr(args, "category", "paper")

        if not isinstance(hg, dgl.DGLHeteroGraph):
            raise TypeError("HERO expects a DGLHeteroGraph as input.")

        if target_ntype not in hg.ntypes:
            raise KeyError(f"Target node type '{target_ntype}' not found in hg.ntypes: {hg.ntypes}")

        if "feature_distance" not in hg.nodes[target_ntype].data:
            raise KeyError(
                f"'feature_distance' not found in g.nodes['{target_ntype}'].data. "
                "Please ensure HERODataset provides it."
            )

        dataset_name = getattr(args, "dataset_name", None)
        if dataset_name is None:
            dataset_name = getattr(args, "dataset", None)

        if dataset_name not in ["ACM4HERO", "Aminer4HERO", "DBLP4HERO","Yelp4HERO"]:
            raise ValueError(
                f"HERO currently only supports ACM4HERO / Aminer4HERO / DBLP4HERO, but got {dataset_name}"
            )

        # ntype_order = cls._get_ntype_order(dataset_name)

        if dataset_name == "Yelp4HERO":
            info_node_order = getattr(args, "node_order", None)
            if info_node_order is None:
                # 退化为图中顺序
                ntype_order = list(hg.ntypes)
            else:
                ntype_order = info_node_order
        else:
            ntype_order = cls._get_ntype_order(dataset_name)
        node_slices, global_features = cls._build_global_feature_tensor_from_graph(
            hg, h_dict=None, ntype_order=ntype_order
        )
        nt_rel = cls._build_nt_rel(hg, ntype_order)

        args.dataset_name = dataset_name
        args.target_ntype = target_ntype
        args.node_num = hg.num_nodes(target_ntype)
        args.node_size = global_features.shape[0]
        args.ft_size = global_features.shape[1]
        args.nt_rel = nt_rel
        args.ntype_order = ntype_order

        required_args = [
            "out_ft", "hid_units", "hid_units2",
            "beta", "alpha",
            "g_dim", "g_equidim", "p_equidim",
            "gamma", "eta", "lambbda"
        ]
        for key in required_args:
            if not hasattr(args, key):
                raise AttributeError(f"args.{key} is required for HERO.")

        return cls(args=args)

    def __init__(self, args):
        super(HERO, self).__init__()
        self.args = args
        self.target_ntype = args.target_ntype

        self.bnn = nn.ModuleDict()
        self.disc2 = nn.ModuleDict()
        self.fc = nn.ModuleDict()
        self.mlp = MLP([self.args.ft_size, self.args.out_ft])
        self.semanticatt = nn.ModuleDict()

        for t, rels in self.args.nt_rel.items():
            self.fc[t] = FullyConnect(
                self.args.hid_units2 + self.args.ft_size,
                self.args.out_ft
            )

            self.disc2[t] = Discriminator(
                self.args.ft_size,
                self.args.out_ft
            )

            for rel in rels:
                self.bnn["0" + rel] = Linear_layer(
                    self.args.ft_size,
                    self.args.hid_units,
                    act=nn.ReLU(),
                    isBias=False
                )
                self.bnn["1" + rel] = Linear_layer(
                    self.args.hid_units,
                    self.args.hid_units2,
                    act=nn.ReLU(),
                    isBias=False
                )

            self.semanticatt["0" + t] = SemanticAttention(
                self.args.hid_units,
                self.args.hid_units // 4
            )
            self.semanticatt["1" + t] = SemanticAttention(
                self.args.hid_units2,
                self.args.hid_units2 // 4
            )

    @staticmethod
    def _get_ntype_order(dataset_name):
        if dataset_name == "ACM4HERO":
            return ["paper", "author", "subject"]
        elif dataset_name == "Aminer4HERO":
            return ["paper", "author", "reference"]
        elif dataset_name == "DBLP4HERO":
            return ["author", "paper", "conference"]
        else:
            raise ValueError(f"Unsupported dataset_name: {dataset_name}")


    @staticmethod
    def _freeze_module(module):
        for p in module.parameters():
            p.requires_grad = False


    @classmethod
    def _build_global_feature_tensor_from_graph(cls, hg, h_dict=None, ntype_order=None):
        if ntype_order is None:
            raise ValueError("ntype_order must be explicitly provided for HERO.")

        features_list = []
        node_slices = {}
        start = 0

        for ntype in ntype_order:
            feat = cls._get_node_feature(hg, ntype, h_dict)
            num_nodes = feat.shape[0]
            end = start + num_nodes
            node_slices[ntype] = slice(start, end)
            features_list.append(feat)
            start = end

        all_features = torch.cat(features_list, dim=0)
        return node_slices, all_features


    @staticmethod
    def _build_nt_rel(hg, ntype_order):
        nt_rel = {}
        for ntype in ntype_order:
            rels = []
            for src_type, etype, dst_type in hg.canonical_etypes:
                if src_type == ntype:
                    rels.append(f"{src_type}__{etype}__{dst_type}")
            nt_rel[ntype] = rels
        return nt_rel


    @staticmethod
    def _mean_neighbor_aggregate_to_src(hg, canonical_etype, dst_feat, num_src_nodes):
        src, dst = hg.edges(etype=canonical_etype, order="eid")

        if src.numel() == 0:
            return torch.zeros(
                num_src_nodes,
                dst_feat.shape[1],
                device=dst_feat.device,
                dtype=dst_feat.dtype
            )

        src_max = int(src.max().item())
        src_min = int(src.min().item())
        dst_max = int(dst.max().item())
        dst_min = int(dst.min().item())

        if src_min < 0 or src_max >= num_src_nodes:
            raise RuntimeError(
                f"[HERO] src index out of range for etype={canonical_etype}: "
                f"src_min={src_min}, src_max={src_max}, num_src_nodes={num_src_nodes}"
            )

        if dst_min < 0 or dst_max >= dst_feat.shape[0]:
            raise RuntimeError(
                f"[HERO] dst index out of range for etype={canonical_etype}: "
                f"dst_min={dst_min}, dst_max={dst_max}, dst_feat_rows={dst_feat.shape[0]}"
            )

        device = dst_feat.device
        feat_dim = dst_feat.shape[1]

        src_sum = torch.zeros(num_src_nodes, feat_dim, device=device, dtype=dst_feat.dtype)
        src_deg = torch.zeros(num_src_nodes, 1, device=device, dtype=dst_feat.dtype)

        gathered = dst_feat[dst]
        src_sum.index_add_(0, src, gathered)

        ones = torch.ones(src.shape[0], 1, device=device, dtype=dst_feat.dtype)
        src_deg.index_add_(0, src, ones)

        src_mean = src_sum / torch.clamp(src_deg, min=1.0)
        return src_mean


    @staticmethod
    def _get_node_feature(hg, ntype, h_dict=None):
        node_data = hg.nodes[ntype].data
        if 'feat' in node_data:
            # print(f"[INPUT] {ntype}: use graph['feat'], shape={node_data['feat'].shape}")
            return node_data['feat'].float()

    @classmethod
    def _build_global_feature_tensor_from_graph(cls, hg, h_dict=None, ntype_order=None):
        if ntype_order is None:
            raise ValueError("ntype_order must be explicitly provided for HERO.")

        features_list = []
        node_slices = {}
        start = 0

        for ntype in ntype_order:
            feat = cls._get_node_feature(hg, ntype, h_dict)
            num_nodes = feat.shape[0]
            end = start + num_nodes

            node_slices[ntype] = slice(start, end)
            features_list.append(feat)
            start = end

        all_features = torch.cat(features_list, dim=0)
        return node_slices, all_features

  

    def _compute_embeddings(self, hg, features, distance, node_slices):
        device = features.device

        embs1 = torch.zeros((self.args.node_size, self.args.hid_units), device=device)
        embs2 = torch.zeros((self.args.node_size, self.args.out_ft), device=device)

        for n, rels in self.args.nt_rel.items():
            if len(rels) == 0:
                continue

            vec = []
            for rel in rels:
                src_type, etype, dst_type = rel.split("__")
                canonical_etype = (src_type, etype, dst_type)

                # 原文：mean_neighbor = A_(n,t) @ features[t]
                dst_feat = features[node_slices[dst_type]]

                mean_neighbor = self._mean_neighbor_aggregate_to_src(
                    hg,
                    canonical_etype,
                    dst_feat=dst_feat,
                    num_src_nodes=hg.num_nodes(src_type)
                )

                v = self.bnn["0" + rel](mean_neighbor)

                vec.append(v)

            vec = torch.stack(vec, dim=0)
            v_summary = torch.mean(vec, dim=0)

           
            embs1[node_slices[n]] = v_summary
      
        for n, rels in self.args.nt_rel.items():
            if len(rels) == 0:
                continue

            vec = []
            for rel in rels:
                src_type, etype, dst_type = rel.split("__")
                canonical_etype = (src_type, etype, dst_type)

                dst_emb = embs1[node_slices[dst_type]]

                mean_neighbor = self._mean_neighbor_aggregate_to_src(
                    hg,
                    canonical_etype,
                    dst_feat=dst_emb,
                    num_src_nodes=hg.num_nodes(src_type)
                )

                v = self.bnn["1" + rel](mean_neighbor)

                vec.append(v)

            vec = torch.stack(vec, dim=0)
            v_summary = torch.mean(vec, dim=0)

            self_feat = features[node_slices[n]]
            v_cat = torch.hstack((v_summary, self_feat))
            v_summary = self.fc[n](v_cat)

            embs2[node_slices[n]] = v_summary

     
        if self.args.dataset_name == "ACM4HERO":
            embs_het = embs1
        else:
            embs_het = embs2

        emb_f = self.mlp(features)
        emb_f = emb_f[0:self.args.node_num]

        coe2 = 1.0 / self.args.beta
        res = torch.mm(emb_f.T, emb_f)
        inv = torch.inverse(torch.eye(emb_f.shape[1], device=device) + coe2 * res)
        res = torch.mm(inv, res)
        res = coe2 * emb_f - coe2 * coe2 * torch.mm(emb_f, res)
        tmp = torch.mm(emb_f.T, res)
        part1 = torch.mm(emb_f, tmp)

        part2 = (-self.args.alpha / 2) * torch.mm(distance, res)

        embs_all = part1 + part2
        embs_hom = embs_all[0:self.args.node_num]
        embs_het = embs_het[0:self.args.node_num]

        return embs_het, embs_hom

    def forward(self, hg, h_dict=None):

        node_slices, features = self._build_global_feature_tensor_from_graph(hg, h_dict, ntype_order=self.args.ntype_order)
        device = next(self.parameters()).device

        features = features.to(device)
        distance = hg.nodes[self.target_ntype].data['feature_distance'].to(device)

        emb_het, emb_hom = self._compute_embeddings(
            hg=hg,
            features=features,
            distance=distance,
            node_slices=node_slices
        )

        return {"emb_het": emb_het,"emb_hom": emb_hom}

    # ==============================================================
    # 给 trainerflow 调用的接口
    # ==============================================================

    def get_embeddings(self, hg, h_dict=None):
        out = self.forward(hg, h_dict)
        return out["emb_het"], out["emb_hom"]

    def get_concat_embeddings(self, hg, h_dict=None):
  
        emb_het, emb_hom = self.get_embeddings(hg, h_dict)
        return torch.cat([emb_het, emb_hom], dim=1)

