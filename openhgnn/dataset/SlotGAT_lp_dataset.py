import os
import random
from collections import Counter, defaultdict

import dgl
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from . import BaseDataset, register_dataset


@register_dataset('slotgat_lp')
@register_dataset('SlotGAT_LP')
class SlotGATLPDataset(BaseDataset):
    _local_datasets = {
        'slotgat_lp': 'LastFM',
        'SlotGAT_LP': 'LastFM',
        'LastFM': 'LastFM',
        'PubMed_LP': 'PubMed_LP',
    }

    def __init__(self, dataset_name, *args, **kwargs):
        super(SlotGATLPDataset, self).__init__(*args, **kwargs)
        args = kwargs['args']
        self.args = args
        self.name = dataset_name
        self.data_root = getattr(args, 'slotgat_lp_data_root', './openhgnn/dataset/SlotGAT/LP')
        self.path = getattr(args, 'slotgat_data_path', None)
        if self.path is None:
            self.path = os.path.join(self.data_root, self._local_datasets.get(dataset_name, dataset_name))
        self.path = self._resolve_data_path(self.path)
        self.has_feature = True
        self.meta_paths_dict = None
        self.target_link = []
        self.target_link_r = None
        self._build(args.device)

    def _resolve_data_path(self, explicit_path):
        required = ('node.dat', 'link.dat', 'link.dat.test')
        candidates = [explicit_path, os.path.join(explicit_path, self.name), os.path.join(explicit_path, 'data')]
        for path in candidates:
            if path and all(os.path.exists(os.path.join(path, file_name)) for file_name in required):
                return path
        raise FileNotFoundError(f'Cannot find SlotGAT LP files {required} under {explicit_path}.')

    def _build(self, device):
        self.nodes = self.load_nodes()
        self.links = self.load_links('link.dat')
        self.links_test = self.load_links('link.dat.test')
        self.test_types = list(self.links_test['data'].keys())
        self.train_pos, self.valid_pos = self.get_train_valid_pos()
        self.features_list = self.get_feature_list(device)
        self.in_dim = [feature.shape[1] for feature in self.features_list]
        self.g = self.load_graph(device)
        self.e_feat = self.get_e_feat()
        self.num_ntype = len(self.features_list)
        self.num_etypes = int(self.e_feat.max().item()) + 1 if self.e_feat.numel() > 0 else 0
        self.num_classes = getattr(self.args, 'hid_dim', getattr(self.args, 'hidden_dim', 64))
        self.process_g()
        for key, value in [('in_dim', self.in_dim), ('num_ntype', self.num_ntype), ('num_etypes', self.num_etypes), ('num_classes', self.num_classes)]:
            if not hasattr(self.args, key) or getattr(self.args, key) is None:
                setattr(self.args, key, value)

    def load_graph(self, device):
        adj_m = sum(self.links['data'].values())
        g = dgl.from_scipy((adj_m + adj_m.T).tocsr())
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)
        return g.to(device)

    def get_feature_list(self, device):
        features = [sp.eye(self.nodes['count'][i]) if self.nodes['attr'][i] is None else self.nodes['attr'][i]
                    for i in range(len(self.nodes['count']))]
        features_list = [self.mat2tensor(feature).to(device) for feature in features]
        feats_type = getattr(self.args, 'feats_type', 2)
        if feats_type in (1, 5):
            save = 0 if feats_type == 1 else 2
            for i in range(len(features_list)):
                if i != save:
                    features_list[i] = torch.zeros((features_list[i].shape[0], 10), device=device)
        elif feats_type in (2, 3, 4):
            save = feats_type - 2 if feats_type in (2, 4) else -1
            for i in range(len(features_list)):
                if i == save:
                    continue
                dim = features_list[i].shape[0]
                indices = torch.LongTensor(np.vstack((np.arange(dim), np.arange(dim))))
                values = torch.FloatTensor(np.ones(dim))
                features_list[i] = torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device)
        return features_list

    def mat2tensor(self, mat):
        if isinstance(mat, np.ndarray):
            return torch.from_numpy(mat).float()
        coo = mat.tocoo()
        indices = torch.LongTensor(np.vstack((coo.row, coo.col)))
        values = torch.FloatTensor(coo.data)
        return torch.sparse_coo_tensor(indices, values, torch.Size(coo.shape), dtype=torch.float32)

    def get_e_feat(self):
        edge2type = {}
        rel_count = len(self.links['count'])
        for rel_id, mat in self.links['data'].items():
            for u, v in zip(*mat.nonzero()):
                edge2type[(u, v)] = rel_id
        for node_id in range(self.nodes['total']):
            edge2type.setdefault((node_id, node_id), rel_count)
        for rel_id, mat in self.links['data'].items():
            for u, v in zip(*mat.nonzero()):
                edge2type.setdefault((v, u), rel_id + 1 + rel_count)
        e_feat = [edge2type[(u.item(), v.item())] for u, v in zip(*self.g.cpu().edges())]
        return torch.tensor(e_feat, dtype=torch.long, device=self.g.device)

    def process_g(self):
        self.g.edge_type_indexer = F.one_hot(self.e_feat).to(self.g.device)
        self.g.node_idx_by_ntype = []
        self.g.num_ntypes = len(self.features_list)
        self.g.node_ntype_indexer = torch.zeros(self.nodes['total'], len(self.features_list), device=self.g.device)
        idx = 0
        for ntype, feature in enumerate(self.features_list):
            nodes = []
            for _ in range(feature.shape[0]):
                nodes.append(idx)
                self.g.node_ntype_indexer[idx][ntype] = 1
                idx += 1
            self.g.node_idx_by_ntype.append(nodes)

    def get_train_valid_pos(self, train_ratio=0.9):
        train_pos, valid_pos = {}, {}
        for rel_id, mat in self.links['data'].items():
            train_pos[rel_id], valid_pos[rel_id] = [[], []], [[], []]
            last_h_id = -1
            for h_id, t_id in zip(*mat.nonzero()):
                if h_id != last_h_id or random.random() < train_ratio:
                    train_pos[rel_id][0].append(h_id); train_pos[rel_id][1].append(t_id)
                    last_h_id = h_id
                else:
                    valid_pos[rel_id][0].append(h_id); valid_pos[rel_id][1].append(t_id)
                    mat[h_id, t_id] = 0
            mat.eliminate_zeros()
        return train_pos, valid_pos

    def sample_negative(self, positive_edges, edge_types=None):
        edge_types = self.test_types if edge_types is None else edge_types
        neg = {}
        for rel_id in edge_types:
            _, t_type = self.links['meta'][rel_id]
            low, high = self.nodes['shift'][t_type], self.nodes['shift'][t_type] + self.nodes['count'][t_type]
            neg[rel_id] = [[], []]
            for h_id in positive_edges[rel_id][0]:
                neg[rel_id][0].append(h_id)
                neg[rel_id][1].append(random.randrange(low, high))
        return neg

    def get_eval_data(self, mode='valid'):
        if mode == 'valid':
            return self.merge_pos_neg(self.valid_pos, self.sample_negative(self.valid_pos))
        return self.get_test_neigh_w_random()

    def merge_pos_neg(self, pos, neg):
        neigh, label = {}, {}
        for rel_id in self.test_types:
            neigh[rel_id] = [list(pos[rel_id][0]) + list(neg[rel_id][0]), list(pos[rel_id][1]) + list(neg[rel_id][1])]
            label[rel_id] = [1] * len(pos[rel_id][0]) + [0] * len(neg[rel_id][0])
        return neigh, label

    def get_test_neigh_w_random(self):
        random.seed(1)
        all_edges = defaultdict(set)
        pos_links = sum(self.links['data'].values()) + sum(self.links_test['data'].values())
        row, col = (pos_links + pos_links.T).nonzero()
        for h_id, t_id in zip(row, col):
            all_edges[h_id].add(t_id)
        neigh, label = {}, {}
        for rel_id, mat in self.links_test['data'].items():
            _, t_type = self.links_test['meta'][rel_id]
            low, high = self.nodes['shift'][t_type], self.nodes['shift'][t_type] + self.nodes['count'][t_type]
            neigh[rel_id], label[rel_id] = [[], []], []
            for h_id, t_id in zip(*mat.nonzero()):
                neigh[rel_id][0].append(h_id); neigh[rel_id][1].append(t_id); label[rel_id].append(1)
                neg_t = random.randrange(low, high)
                while neg_t in all_edges[h_id]:
                    neg_t = random.randrange(low, high)
                neigh[rel_id][0].append(h_id); neigh[rel_id][1].append(neg_t); label[rel_id].append(0)
        return neigh, label

    @staticmethod
    def evaluate(edge_list, confidence, labels):
        labels = np.array(labels); confidence = np.array(confidence)
        roc_auc = roc_auc_score(labels, confidence) if len(set(labels)) > 1 else 0.0
        mrrs = []
        by_head = defaultdict(list)
        for h_id, score, label in zip(edge_list[0], confidence, labels):
            by_head[h_id].append((score, label))
        for values in by_head.values():
            ranked = sorted(values, key=lambda x: -x[0])
            for rank, (_, label) in enumerate(ranked, start=1):
                if label == 1:
                    mrrs.append(1 / rank); break
        return {'roc_auc': roc_auc, 'MRR': float(np.mean(mrrs)) if mrrs else 0.0}

    def load_links(self, name):
        links = {'total': 0, 'count': Counter(), 'meta': {}, 'data': defaultdict(list)}
        with open(os.path.join(self.path, name), 'r', encoding='utf-8') as f:
            for line in f:
                h_id, t_id, rel_id, weight = line.split('\t')[:4]
                h_id, t_id, rel_id, weight = int(h_id), int(t_id), int(rel_id), float(weight)
                links['meta'].setdefault(rel_id, (self.get_node_type(h_id), self.get_node_type(t_id)))
                links['data'][rel_id].append((h_id, t_id, weight)); links['count'][rel_id] += 1; links['total'] += 1
        links['data'] = {rel_id: self.list_to_sp_mat(edges) for rel_id, edges in links['data'].items()}
        return links

    def list_to_sp_mat(self, edges):
        return sp.coo_matrix(([x[2] for x in edges], ([x[0] for x in edges], [x[1] for x in edges])),
                             shape=(self.nodes['total'], self.nodes['total'])).tocsr()

    def load_nodes(self):
        nodes = {'total': 0, 'count': Counter(), 'attr': {}, 'shift': {}}
        with open(os.path.join(self.path, 'node.dat'), 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.rstrip('\n').split('\t')
                node_id, node_type = int(parts[0]), int(parts[2])
                if len(parts) == 4:
                    nodes['attr'][node_id] = list(map(float, parts[3].split(',')))
                nodes['count'][node_type] += 1; nodes['total'] += 1
        shift, attr = 0, {}
        for ntype in range(len(nodes['count'])):
            nodes['shift'][ntype] = shift
            attr[ntype] = np.array([nodes['attr'][i] for i in range(shift, shift + nodes['count'][ntype])]) if shift in nodes['attr'] else None
            shift += nodes['count'][ntype]
        nodes['attr'] = attr
        return nodes

    def get_node_type(self, node_id):
        for ntype in range(len(self.nodes['shift'])):
            if node_id < self.nodes['shift'][ntype] + self.nodes['count'][ntype]:
                return ntype
        raise ValueError(f'Unknown node id {node_id}')

    def get_split(self):
        return self.g, None, None, None, None
