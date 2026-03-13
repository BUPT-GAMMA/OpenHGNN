"""
HTGformer dataset
==================
Paper: HTGformer: Heterogeneous Temporal Graph Transformer (SIGIR 2025)

Support the datasets in Table 1 of the paper:
  1. OGBN-MAG  — Link Prediction (AUC, AP)
     data sources: HTGNN (SDM'2022), need ogbn_graphs*.bin + mp2vec/g0~g9.vector
  2. Aminer    — Link Prediction (AUC, AP)
     data sources: DHGAS (AAAI'2023), PyG HeteroData form (processed-False-32.pt)
  3. YELP      — Node Classification (Macro-F1, Recall)
     data sources: DHGAS (AAAI'2023), PyG HeteroData form (True-32.pt)
  4. COVID-19  — Node Classification (MAE)
     data sources: HTGNN (SDM'2022), need covid_graphs.bin
"""

import os
import torch
import numpy as np
import dgl
from collections import defaultdict


# ══════════════════════════════════════════════════════════════════════════════
# base class
# ══════════════════════════════════════════════════════════════════════════════
class HTGDatasetBase:
    """HTGformer 数据集基类"""
    def __init__(self):
        self.graphs = []
        self.feat_dicts = []
        self.labels = None
        self.train_idx = None
        self.val_idx = None
        self.test_idx = None
        self.category = None
        self.task = None
        self.num_classes = None
        self.in_dim_dict = {}
        # Link prediction multi-sample mode (OGBN-MAG, Aminer)
        self.train_samples = []
        self.val_samples = []
        self.test_samples = []


# ══════════════════════════════════════════════════════════════════════════════
# OGBN-MAG tool function
# ══════════════════════════════════════════════════════════════════════════════
def _mp2vec_feat(path, g):
    from gensim.models import KeyedVectors
    wordvec = KeyedVectors.load(path, mmap='r')
    for ntype in g.ntypes:
        prefix = {'author': 'a_', 'institution': 'i_', 'field_of_study': 't_'}.get(ntype)
        if prefix is None:
            continue
        feat = torch.zeros(g.num_nodes(ntype), 128)
        for j in range(g.num_nodes(ntype)):
            try:
                feat[j] = torch.from_numpy(np.array(wordvec[f'{prefix}{j}']))
            except KeyError:
                continue
        g.nodes[ntype].data['feat'] = feat
    return g


def _generate_APA(graph, device):
    AP = graph.adj(etype=('author', 'writes', 'paper')).to_dense()
    PA = AP.t()
    APA = torch.mm(AP.to(device), PA.to(device)).detach().cpu()
    APA[torch.eye(APA.shape[0]).bool()] = 0.5
    return APA


def _construct_htg_label_mag(glist, idx, device):
    APA_cur = _generate_APA(glist[idx], device)
    APA_pre = _generate_APA(glist[idx - 1], device)
    APA_pre = (APA_pre > 0.5).float()
    APA_cur = (APA_cur > 0.5).float()
    APA_sub = APA_cur - APA_pre
    APA_add = APA_cur + APA_pre
    APA_add[torch.eye(APA_add.shape[0]).bool()] = 0.5
    pos_src, pos_dst = (APA_sub == 1).nonzero(as_tuple=True)
    neg_src, neg_dst = (APA_add == 0).nonzero(as_tuple=True)
    size = max(1, int(pos_src.shape[0] * 0.1))
    pos_idx = torch.randperm(pos_src.shape[0])[:size]
    neg_idx = torch.randperm(neg_src.shape[0])[:size]
    n = APA_cur.shape[0]
    return (dgl.graph((pos_src[pos_idx], pos_dst[pos_idx]), num_nodes=n),
            dgl.graph((neg_src[neg_idx], neg_dst[neg_idx]), num_nodes=n))


# ══════════════════════════════════════════════════════════════════════════════
# OGBN-MAG dataset
# ══════════════════════════════════════════════════════════════════════════════
class OGBNMAGHTGDataset(HTGDatasetBase):
    """
    OGBN-MAG Link prediction dataset: author - Work prediction
    output: AUC 94.61±0.35%, AP 93.98±0.51% (paper: 92.56/91.64)

    data directory：
      raw_dir/ogbn_graphs*.bin + raw_dir/mp2vec/g0~g9.vector
    """
    def __init__(self, raw_dir='./data/ogbn_mag', use_synthetic=False,
                 num_timestamps=10, time_window=3, device=None):
        super().__init__()
        self.category = 'author'
        self.task = 'link_prediction'
        self.time_window = time_window
        self.device = device or torch.device('cpu')
        if use_synthetic or not os.path.exists(raw_dir):
            print("[OGBNMAGHTGDataset] 使用合成数据")
            self._build_synthetic()
        else:
            self._load_real(raw_dir)

    def _load_real(self, raw_dir):
        from dgl.data.utils import load_graphs
        for name in ['ogbn_graphs.bin', 'ogbn_graphs_001.bin']:
            path = os.path.join(raw_dir, name)
            if os.path.exists(path):
                glist, _ = load_graphs(path)
                break
        else:
            raise FileNotFoundError(f"在 {raw_dir} 找不到 ogbn_graphs*.bin")
        mp2vec_dir = os.path.join(raw_dir, 'mp2vec')
        print(f"  加载 {len(glist)} 个快照 + mp2vec...")
        glist = [_mp2vec_feat(f'{mp2vec_dir}/g{i}.vector', g) for i, g in enumerate(glist)]
        tw = self.time_window
        for i in range(len(glist)):
            if i < tw:
                continue
            wg = glist[i - tw:i]
            fds = [{ntype: g.nodes[ntype].data.get('feat',
                    torch.zeros(g.num_nodes(ntype), 128)).to(self.device)
                    for ntype in g.ntypes} for g in wg]
            pos_g, neg_g = _construct_htg_label_mag(glist, i, self.device)
            sample = (wg, fds, pos_g, neg_g)
            if i == len(glist) - 1:
                self.test_samples.append(sample)
            elif i == len(glist) - 2:
                self.val_samples.append(sample)
            else:
                self.train_samples.append(sample)
        self.in_dim_dict = {k: v.shape[-1] for k, v in self.train_samples[0][1][0].items()}
        print(f"  train:{len(self.train_samples)} val:{len(self.val_samples)} test:{len(self.test_samples)}")

    def _build_synthetic(self):
        self.in_dim_dict = {'author': 128, 'paper': 128, 'field_of_study': 128, 'institution': 128}


# ══════════════════════════════════════════════════════════════════════════════
# Aminer tool function
# ══════════════════════════════════════════════════════════════════════════════
def _build_coauthor_samples(wri_ei, wri_et, t, num_authors):
    mask = wri_et == t
    papers = wri_ei[0, mask].numpy()
    authors = wri_ei[1, mask].numpy()
    p2a = defaultdict(list)
    for p, a in zip(papers, authors):
        p2a[p].append(a)
    edge_set = set()
    for p, a_list in p2a.items():
        for i in range(len(a_list)):
            for j in range(i + 1, len(a_list)):
                if a_list[i] != a_list[j]:
                    edge_set.add((a_list[i], a_list[j]))
                    edge_set.add((a_list[j], a_list[i]))
    if not edge_set:
        return None
    edges = list(edge_set)
    pos_src = torch.tensor([e[0] for e in edges], dtype=torch.long)
    pos_dst = torch.tensor([e[1] for e in edges], dtype=torch.long)
    neg_dst = torch.randint(0, num_authors, pos_dst.shape)
    return pos_src, pos_dst, pos_src.clone(), neg_dst


# ══════════════════════════════════════════════════════════════════════════════
# Aminer dataset
# ══════════════════════════════════════════════════════════════════════════════
class AminerHTGDataset(HTGDatasetBase):
    """
    Aminer Link prediction dataset (author 共作预测)。
    output: AUC 85.98±0.89%, AP 80.83±0.90% (paper: 89.78/88.03)

    data file: aminer_processed.pt (DHGAS 仓库 Cross-Domain_data/processed-False-32.pt)
    """
    NUM_PAPER = 18464
    NUM_AUTHOR = 23035
    NUM_VENUE = 22

    def __init__(self, raw_dir='./data', use_synthetic=False, time_window=5):
        super().__init__()
        self.category = 'author'
        self.task = 'link_prediction'
        self.time_window = time_window
        pt_path = os.path.join(raw_dir, 'aminer_processed.pt')
        if use_synthetic or not os.path.exists(pt_path):
            print("[AminerHTGDataset] 使用合成数据")
            self._build_synthetic()
        else:
            self._load_real(pt_path)

    def _load_real(self, pt_path):
        data = torch.load(pt_path, map_location='cpu')
        paper_x = data['paper'].x
        author_x = data['author'].x.float()
        venue_x = data['venue'].x.float()
        pub_ei = data['paper', 'published', 'venue'].edge_index
        pub_et = data['paper', 'published', 'venue'].edge_time.squeeze()
        wri_ei = data['paper', 'written', 'author'].edge_index
        wri_et = data['paper', 'written', 'author'].edge_time.squeeze()

        self.snapshots, self.all_feat_dicts = [], []
        for t in range(16):
            pm, wm = pub_et == t, wri_et == t
            ps, pd = pub_ei[0, pm], pub_ei[1, pm]
            ws, wd = wri_ei[0, wm], wri_ei[1, wm]
            if ps.shape[0] == 0: ps = pd = torch.zeros(1, dtype=torch.long)
            if ws.shape[0] == 0: ws = wd = torch.zeros(1, dtype=torch.long)
            g = dgl.heterograph({
                ('paper', 'published', 'venue'): (ps, pd),
                ('venue', 'published_by', 'paper'): (pd, ps),
                ('paper', 'written', 'author'): (ws, wd),
                ('author', 'writes', 'paper'): (wd, ws),
            }, num_nodes_dict={'paper': self.NUM_PAPER, 'author': self.NUM_AUTHOR, 'venue': self.NUM_VENUE})
            self.snapshots.append(g)
            self.all_feat_dicts.append({'paper': paper_x, 'author': author_x, 'venue': venue_x})

        self.in_dim_dict = {'paper': 32, 'author': 1, 'venue': 1}
        self.coauthor_samples = {}
        for t in range(16):
            result = _build_coauthor_samples(wri_ei, wri_et, t, self.NUM_AUTHOR)
            if result:
                self.coauthor_samples[t] = result
        self.train_ts = [t for t in range(14) if t in self.coauthor_samples]
        self.val_ts = [14] if 14 in self.coauthor_samples else []
        self.test_ts = [15] if 15 in self.coauthor_samples else []
        print(f"  16 快照, Train:{len(self.train_ts)} Val:{len(self.val_ts)} Test:{len(self.test_ts)}")

    def _build_synthetic(self):
        self.in_dim_dict = {'paper': 32, 'author': 1, 'venue': 1}
        self.snapshots, self.all_feat_dicts = [], []
        self.coauthor_samples = {}
        self.train_ts = self.val_ts = self.test_ts = []


# ══════════════════════════════════════════════════════════════════════════════
# YELP dataset
# ══════════════════════════════════════════════════════════════════════════════
class YELPHTGDataset(HTGDatasetBase):
    """
    YELP node classification dataset (item 3 class)。
    output: F1 35.91±1.59%, Recall 40.91±1.57% (paper: 43.24/43.86, w/o_LLM)

    data file: yelp_processed.pt (DHGAS repository yelp/processed/True-32.pt)
    node classification : 随机 80:10:10 (consistent with DHGAS)
    """
    NUM_USER = 55702
    NUM_ITEM = 12524

    def __init__(self, raw_dir='./data', use_synthetic=False, seed=22):
        super().__init__()
        self.category = 'item'
        self.task = 'node_classification'
        self.num_classes = 3
        pt_path = os.path.join(raw_dir, 'yelp_processed.pt')
        if use_synthetic or not os.path.exists(pt_path):
            print("[YELPHTGDataset] 使用合成数据")
            self._build_synthetic()
        else:
            self._load_real(pt_path, seed)

    def _load_real(self, pt_path, seed=22):
        data = torch.load(pt_path, map_location='cpu')
        user_x, item_x = data['user'].x, data['item'].x
        rev_ei = data['user', 'review', 'item'].edge_index
        rev_et = data['user', 'review', 'item'].edge_time.squeeze()
        tip_ei = data['user', 'tip', 'item'].edge_index
        tip_et = data['user', 'tip', 'item'].edge_time.squeeze()

        for t in range(12):
            rm, tm = rev_et == t, tip_et == t
            rs, rd = rev_ei[0, rm], rev_ei[1, rm]
            ts, td = tip_ei[0, tm], tip_ei[1, tm]
            if rs.shape[0] == 0: rs = rd = torch.zeros(1, dtype=torch.long)
            if ts.shape[0] == 0: ts = td = torch.zeros(1, dtype=torch.long)
            g = dgl.heterograph({
                ('user', 'review', 'item'): (rs, rd),
                ('item', 'reviewed_by', 'user'): (rd, rs),
                ('user', 'tip', 'item'): (ts, td),
                ('item', 'tipped_by', 'user'): (td, ts),
            }, num_nodes_dict={'user': self.NUM_USER, 'item': self.NUM_ITEM})
            self.graphs.append(g)
            self.feat_dicts.append({'user': user_x, 'item': item_x})

        self.labels = data['item'].y
        self.num_classes = int(self.labels.max().item()) + 1
        self.in_dim_dict = {'user': user_x.shape[-1], 'item': item_x.shape[-1]}

        np.random.seed(seed)
        n = self.NUM_ITEM
        val_num = int(np.ceil(0.1 * n))
        test_num = int(np.ceil(0.1 * n))
        perm = torch.from_numpy(np.random.permutation(n))
        self.test_idx = perm[:test_num]
        self.val_idx = perm[test_num:test_num + val_num]
        self.train_idx = perm[test_num + val_num:]
        print(f"  12 快照, {self.num_classes} 类, split: {len(self.train_idx)}/{len(self.val_idx)}/{len(self.test_idx)}")

    def _build_synthetic(self):
        for t in range(12):
            g = dgl.heterograph({
                ('user', 'review', 'item'): (torch.randint(0, self.NUM_USER, (3000,)), torch.randint(0, self.NUM_ITEM, (3000,))),
                ('item', 'reviewed_by', 'user'): (torch.randint(0, self.NUM_ITEM, (3000,)), torch.randint(0, self.NUM_USER, (3000,))),
            }, num_nodes_dict={'user': self.NUM_USER, 'item': self.NUM_ITEM})
            self.graphs.append(g)
            self.feat_dicts.append({'user': torch.randn(self.NUM_USER, 32), 'item': torch.randn(self.NUM_ITEM, 32)})
        self.labels = torch.randint(0, 3, (self.NUM_ITEM,))
        self.in_dim_dict = {'user': 32, 'item': 32}
        n = self.NUM_ITEM; perm = torch.randperm(n)
        self.train_idx = perm[:int(0.8*n)]; self.val_idx = perm[int(0.8*n):int(0.9*n)]; self.test_idx = perm[int(0.9*n):]


# ══════════════════════════════════════════════════════════════════════════════
# COVID-19 dataset
# ══════════════════════════════════════════════════════════════════════════════
class COVID19HTGDataset(HTGDatasetBase):
    """COVID-19 node regression (MAE), hidden_dim=8, predict_type='state'"""
    def __init__(self, raw_dir='./data', use_synthetic=False, time_window=7):
        super().__init__()
        self.category = 'state'
        self.task = 'node_regression'
        self.time_window = time_window
        pt_path = os.path.join(raw_dir, 'covid_graphs.bin')
        if use_synthetic or not os.path.exists(pt_path):
            print("[COVID19HTGDataset] 使用合成数据")
            self._build_synthetic()
        else:
            self._load_real(raw_dir)

    def _load_real(self, raw_dir):
        from dgl.data.utils import load_graphs
        glist, _ = load_graphs(os.path.join(raw_dir, 'covid_graphs.bin'))
        tw = self.time_window
        for i in range(len(glist)):
            if i < tw:
                continue
            wg = glist[i - tw:i]
            # The features of all time steps within the stitched time window are multi-dimensional features.
            # Each node transforms from 1-dimensional to time_window-dimensional, providing more comprehensive temporal information.
            fds = []
            for t_idx, g in enumerate(wg):
                fd = {}
                for ntype in g.ntypes:
                    feat_list = [wg_t.nodes[ntype].data.get('feat',
                                 torch.zeros(wg_t.num_nodes(ntype), 1))
                                 for wg_t in wg]
                    fd[ntype] = torch.cat(feat_list, dim=-1)  # [N, tw]
                fds.append(fd)
            lg = glist[i]
            sample = (wg, fds, lg)
            if i >= len(glist) - 30:
                self.test_samples.append(sample)
            elif i >= len(glist) - 60:
                self.val_samples.append(sample)
            else:
                self.train_samples.append(sample)
        self.in_dim_dict = {k: v.shape[-1] for k, v in self.train_samples[0][1][0].items()}
        print(f"  {len(glist)} 快照, train:{len(self.train_samples)} val:{len(self.val_samples)} test:{len(self.test_samples)}")

    def _build_synthetic(self):
        self.in_dim_dict = {'state': 7, 'county': 7}  # time_window=7 Vertex joining feature
