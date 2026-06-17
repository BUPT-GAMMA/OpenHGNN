"""
HTGformer Dataset
==================
Paper: HTGformer: Heterogeneous Temporal Graph Transformer (SIGIR 2025)

Registered as an OpenHGNN dataset via @register_dataset('htgformer_dataset').
Entry point: build_dataset(dataset, task) in openhgnn/dataset/__init__.py

Supports datasets from Table 1:
  1. ogbn_mag4HGformer  - Link Prediction (AUC, AP)
  2. aminer4HGformer    - Link Prediction (AUC, AP)
  3. yelp4HGformer      - Node Classification (Macro-F1, Recall)
  4. covid4HGformer     - Node Regression (MAE)

Data is automatically downloaded from S3 storage on first use.
The `use_synthetic` flag is ONLY for unit-test fixtures, never the default
training path.
"""

import os
import torch
import numpy as np
import dgl
from collections import defaultdict
from dgl.data.utils import download, extract_archive

from openhgnn.dataset import register_dataset, BaseDataset


# S3 URLs
_S3_PREFIX = 'https://dgl-data.s3.cn-north-1.amazonaws.com.cn/dataset/openhgnn/'
_S3_URLS = {
    'aminer': _S3_PREFIX + 'aminer4HGformer.pt',
    'ogbn_graphs': _S3_PREFIX + 'ogbn4HGformer.bin',
    'mp2vec': _S3_PREFIX + 'mp2vec.zip',
    'yelp': _S3_PREFIX + 'yelp4HGformer.pt',
    'covid': _S3_PREFIX + 'covid4HGformer.bin',
}
_DEFAULT_DATA_DIR = './openhgnn/dataset'

# Mapping from OpenHGNN dataset name -> internal key + task
_DATASET_INFO = {
    'ogbn_mag4HGformer': ('ogbn_mag', 'link_prediction'),
    'aminer4HGformer':   ('aminer',   'link_prediction'),
    'yelp4HGformer':     ('yelp',     'node_classification'),
    'covid4HGformer':    ('covid',    'node_regression'),
}


def _ensure_file(key, filename, data_dir=_DEFAULT_DATA_DIR):
    """Download a single file from S3 if it does not exist locally."""
    filepath = os.path.join(data_dir, filename)
    if not os.path.exists(filepath):
        print(f"  Downloading {filename} from S3...")
        os.makedirs(data_dir, exist_ok=True)
        download(_S3_URLS[key], path=data_dir)
        downloaded = os.path.join(data_dir, os.path.basename(_S3_URLS[key]))
        if downloaded != filepath and os.path.exists(downloaded):
            os.rename(downloaded, filepath)
    return filepath


def _ensure_mp2vec(data_dir=_DEFAULT_DATA_DIR):
    """Download and extract mp2vec embeddings (OGBN-MAG)."""
    mp2vec_dir = os.path.join(data_dir, 'mp2vec')
    if not os.path.exists(mp2vec_dir) or len(os.listdir(mp2vec_dir)) == 0:
        zip_path = os.path.join(data_dir, 'mp2vec.zip')
        if not os.path.exists(zip_path):
            print("  Downloading mp2vec.zip from S3...")
            os.makedirs(data_dir, exist_ok=True)
            download(_S3_URLS['mp2vec'], path=data_dir)
        extract_archive(zip_path, data_dir)
    return mp2vec_dir


# OGBN-MAG utility functions
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


# Registered HTGformer dataset
@register_dataset('htgformer_dataset')
class HTGformerDataset(BaseDataset):
    """
    Unified HTGformer dataset for OpenHGNN.

    Loads heterogeneous temporal graph (HTG) data for one of four sub-datasets,
    each corresponding to a different task:
      - ogbn_mag4HGformer / aminer4HGformer -> link_prediction
      - yelp4HGformer                       -> node_classification
      - covid4HGformer                      -> node_regression

    The loaded object exposes time-step snapshots and per-task training
    structures consumed by HTGformerTrainer.
    """

    NUM = {
        'aminer': dict(paper=18464, author=23035, venue=22),
        'yelp':   dict(user=55702, item=12524),
    }

    def __init__(self, dataset_name, logger=None, args=None, use_synthetic=False):
        # BaseDataset expects a logger kwarg
        super().__init__(logger=logger)
        self.dataset_name = dataset_name
        self.args = args
        key, task = _DATASET_INFO.get(dataset_name, (None, None))
        if key is None:
            raise ValueError(f"Unknown HTGformer dataset: {dataset_name}")
        self.key = key
        self.task = task

        # data_dir resolution: args.data_dir overrides default
        self.data_dir = getattr(args, 'data_dir', _DEFAULT_DATA_DIR) if args else _DEFAULT_DATA_DIR

        # containers
        self.graphs = []
        self.feat_dicts = []
        self.labels = None
        self.train_idx = self.val_idx = self.test_idx = None
        self.category = None
        self.num_classes = None
        self.in_dim_dict = {}
        self.train_samples = []
        self.val_samples = []
        self.test_samples = []
        self.snapshots = []
        self.all_feat_dicts = []
        self.coauthor_samples = {}
        self.train_ts = self.val_ts = self.test_ts = []

        # `use_synthetic` is a TEST-ONLY fixture, never the default training path
        if use_synthetic:
            self._build_synthetic()
        else:
            self._load_real()

    # Dispatch
    def _load_real(self):
        if self.key == 'ogbn_mag':
            self._load_ogbn_mag()
        elif self.key == 'aminer':
            self._load_aminer()
        elif self.key == 'yelp':
            self._load_yelp()
        elif self.key == 'covid':
            self._load_covid()

    def _build_synthetic(self):
        # Minimal fixtures for unit tests only
        if self.key == 'ogbn_mag':
            self.category = 'author'
            self.in_dim_dict = {'author': 128, 'paper': 128, 'field_of_study': 128, 'institution': 128}
        elif self.key == 'aminer':
            self.category = 'author'
            self.in_dim_dict = {'paper': 32, 'author': 1, 'venue': 1}
        elif self.key == 'yelp':
            self.category = 'item'
            self.num_classes = 3
            self.in_dim_dict = {'user': 32, 'item': 32}
        elif self.key == 'covid':
            self.category = 'state'
            self.in_dim_dict = {'state': 7, 'county': 7}

    # OGBN-MAG (link prediction)
    def _load_ogbn_mag(self):
        from dgl.data.utils import load_graphs
        self.category = 'author'
        time_window = getattr(self.args, 'time_window', 3) if self.args else 3
        device = torch.device(getattr(self.args, 'device', 'cpu')) if self.args else torch.device('cpu')

        graph_path = _ensure_file('ogbn_graphs', 'ogbn4HGformer.bin', self.data_dir)
        mp2vec_dir = _ensure_mp2vec(self.data_dir)

        glist, _ = load_graphs(graph_path)
        glist = [_mp2vec_feat(f'{mp2vec_dir}/g{i}.vector', g) for i, g in enumerate(glist)]
        tw = time_window
        for i in range(len(glist)):
            if i < tw:
                continue
            wg = glist[i - tw:i]
            fds = [{ntype: g.nodes[ntype].data.get('feat',
                    torch.zeros(g.num_nodes(ntype), 128)).to(device)
                    for ntype in g.ntypes} for g in wg]
            pos_g, neg_g = _construct_htg_label_mag(glist, i, device)
            sample = (wg, fds, pos_g, neg_g)
            if i == len(glist) - 1:
                self.test_samples.append(sample)
            elif i == len(glist) - 2:
                self.val_samples.append(sample)
            else:
                self.train_samples.append(sample)
        self.in_dim_dict = {k: v.shape[-1] for k, v in self.train_samples[0][1][0].items()}

    # Aminer (link prediction)
    def _load_aminer(self):
        self.category = 'author'
        num = self.NUM['aminer']
        pt_path = _ensure_file('aminer', 'aminer4HGformer.pt', self.data_dir)
        data = torch.load(pt_path, map_location='cpu')
        paper_x = data['paper'].x
        author_x = data['author'].x.float()
        venue_x = data['venue'].x.float()
        pub_ei = data['paper', 'published', 'venue'].edge_index
        pub_et = data['paper', 'published', 'venue'].edge_time.squeeze()
        wri_ei = data['paper', 'written', 'author'].edge_index
        wri_et = data['paper', 'written', 'author'].edge_time.squeeze()

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
            }, num_nodes_dict={'paper': num['paper'], 'author': num['author'], 'venue': num['venue']})
            self.snapshots.append(g)
            self.all_feat_dicts.append({'paper': paper_x, 'author': author_x, 'venue': venue_x})

        self.in_dim_dict = {'paper': 32, 'author': 1, 'venue': 1}
        for t in range(16):
            result = _build_coauthor_samples(wri_ei, wri_et, t, num['author'])
            if result:
                self.coauthor_samples[t] = result
        self.train_ts = [t for t in range(14) if t in self.coauthor_samples]
        self.val_ts = [14] if 14 in self.coauthor_samples else []
        self.test_ts = [15] if 15 in self.coauthor_samples else []

    # YELP (node classification)
    def _load_yelp(self):
        self.category = 'item'
        num = self.NUM['yelp']
        seed = getattr(self.args, 'seed', 22) if self.args else 22
        pt_path = _ensure_file('yelp', 'yelp4HGformer.pt', self.data_dir)
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
            }, num_nodes_dict={'user': num['user'], 'item': num['item']})
            self.graphs.append(g)
            self.feat_dicts.append({'user': user_x, 'item': item_x})

        self.labels = data['item'].y
        self.num_classes = int(self.labels.max().item()) + 1
        self.in_dim_dict = {'user': user_x.shape[-1], 'item': item_x.shape[-1]}

        np.random.seed(seed)
        n = num['item']
        val_num = test_num = int(np.ceil(0.1 * n))
        perm = torch.from_numpy(np.random.permutation(n))
        self.test_idx = perm[:test_num]
        self.val_idx = perm[test_num:test_num + val_num]
        self.train_idx = perm[test_num + val_num:]

    # COVID-19 (node regression)
    def _load_covid(self):
        from dgl.data.utils import load_graphs
        self.category = 'state'
        time_window = getattr(self.args, 'time_window', 7) if self.args else 7
        graph_path = _ensure_file('covid', 'covid4HGformer.bin', self.data_dir)
        glist, _ = load_graphs(graph_path)
        tw = time_window
        for i in range(len(glist)):
            if i < tw:
                continue
            wg = glist[i - tw:i]
            # Key optimization: concatenate features from all time steps (1-dim -> tw-dim)
            fds = []
            for t_idx, g in enumerate(wg):
                fd = {}
                for ntype in g.ntypes:
                    feat_list = [wg_t.nodes[ntype].data.get('feat',
                                 torch.zeros(wg_t.num_nodes(ntype), 1))
                                 for wg_t in wg]
                    fd[ntype] = torch.cat(feat_list, dim=-1)
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
