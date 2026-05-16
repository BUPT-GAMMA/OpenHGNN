"""HGDL dataset for OpenHGNN.

Loads the heterogeneous graphs used by the HGDL paper (NeurIPS 2024)
for the DBLP and ACM datasets, and exposes them via the OpenHGNN
node_classification dataset interface.

The preprocessing logic is ported from the upstream ``preprocess_DBLP.py``
in https://github.com/Listener-Watcher/HGDL. It has been refactored into
pure functions so that importing this module does not touch disk.

Raw data layout (place under ``openhgnn/dataset/DBLP/``):
    APA.npz                  - precomputed label distributions y
    author.txt               - author id list
    author_label.txt         - author -> label
    conf.txt                 - conference id list
    conf_label.txt           - conference -> label
    paper.txt                - paper id list (unused)
    paper_author.txt         - paper-author edges
    paper_conf.txt           - paper-conference edges
    paper_label.txt          - paper -> label (unused)
    paper_term.txt           - paper-term edges
    term.txt                 - term id list

For numerical alignment with the reference implementation, ``_load()``
reseeds ``random / numpy / torch / dgl / cuda`` in the exact order used
by the upstream pipeline so the 40/10/50 author split is bit-perfect
identical across runs.

This dataset inherits ``NodeClassificationDataset`` so it can be driven
by the standard ``node_classification`` task. HGDL is a per-node
label-distribution problem; the trainerflow handles the
label-distribution loss and metrics directly (the standard task's
argmax-based F1 evaluator is not used).
"""
from __future__ import annotations

import copy
import os
import random
from collections import defaultdict

import dgl
import numpy as np
import torch

from . import register_dataset
from .NodeClassificationDataset import NodeClassificationDataset


# ---------------------------------------------------------------------------
# File parsing helpers (ported verbatim from upstream preprocess_DBLP.py)
# ---------------------------------------------------------------------------
def _parse_node(path, sep='\t'):
    """Parse a node file into ``(index_list, id_to_index_dict)``."""
    index = []
    id_to_idx = {}
    with open(path) as fh:
        for i, line in enumerate(fh.readlines()):
            line = line.strip('\n').split(sep)
            index.append(int(line[0]))
            id_to_idx[int(line[0])] = i
    return index, id_to_idx


def _parse_two_col(path, unique=True, sep='\t'):
    """Parse a two-column file into forward / backward dicts."""
    if unique:
        forward, backward = {}, {}
    else:
        forward, backward = defaultdict(list), defaultdict(list)
    with open(path) as fh:
        for line in fh.readlines():
            line = line.strip('\n').split(sep)
            first, second = int(line[0]), int(line[1])
            if unique:
                forward[first] = second
                backward[second] = first
            else:
                forward[first].append(second)
                backward[second].append(first)
    return forward, backward


def _edge_dict_to_index(edge_dict):
    """Convert ``{src: [dst, ...]}`` into a ``2 x num_edges`` tensor."""
    edge_index = [[], []]
    for start, end_list in edge_dict.items():
        for end in end_list:
            edge_index[0].append(start)
            edge_index[1].append(end)
    return torch.tensor(edge_index)


def _edge_index_to_dict(edge_index):
    """Convert a ``2 x num_edges`` edge_index to ``{src: [dst, ...]}``."""
    edge_dict = defaultdict(list)
    num_edges = len(edge_index[0])
    for i in range(num_edges):
        edge_dict[edge_index[0][i].item()].append(edge_index[1][i].item())
    return edge_dict


def _id_to_index(edge_index, row_map, col_map):
    """Remap raw ids in an edge_index using two id->index dicts."""
    new_ei = copy.deepcopy(edge_index)
    for i in range(edge_index.shape[1]):
        new_ei[0][i] = row_map[edge_index[0][i].item()][0]
        new_ei[1][i] = col_map[edge_index[1][i].item()][0]
    return new_ei


# ---------------------------------------------------------------------------
# Build the DBLP heterogeneous graph from raw .txt + APA.npz
# ---------------------------------------------------------------------------
def _build_dblp_hg(data_dir):
    """Reproduce upstream preprocess_DBLP.py as a pure function.

    Returns
    -------
    g           : dgl.DGLHeteroGraph
    features    : torch.Tensor  shape (n_author, n_terms)
    labels      : torch.Tensor  shape (n_author, 4)  label distributions
    num_classes : int           always 4 for DBLP
    """
    p = lambda f: os.path.join(data_dir, f)

    # ------- node lists / id maps -------
    author_arr, author_dict = _parse_node(p("author.txt"))
    author_label = _parse_two_col(p("author_label.txt"))[0]
    term_arr, term_dict = _parse_node(p("term.txt"))
    conf_arr, conf_dict = _parse_node(p("conf.txt"))
    conf_label = _parse_two_col(p("conf_label.txt"))[0]

    # ------- raw edges -------
    paper_author, author_paper = _parse_two_col(p("paper_author.txt"), unique=False)
    paper_term, term_paper = _parse_two_col(p("paper_term.txt"), unique=False)
    paper_conf, conf_paper = _parse_two_col(p("paper_conf.txt"), unique=False)

    # ------- filter authors to those with labels -------
    paper_author_new = copy.deepcopy(paper_author)
    author_paper_new = copy.deepcopy(author_paper)
    author_index = [author_dict[aid_] for aid_ in author_label.keys()]
    author_arr_new = [author_arr[i] for i in range(len(author_arr))
                      if i in author_index]

    author_dict_new = defaultdict(list)
    for i, a in enumerate(author_arr_new):
        author_dict_new[a].append(i)

    for key, val in paper_author.items():
        for v in val:
            if v not in author_arr_new:
                paper_author_new[key].remove(v)
            if len(paper_author_new[key]) == 0:
                paper_author_new.pop(key)
    paper_id = list(paper_author_new.keys())

    # ------- prune term / conf edges to surviving papers -------
    paper_term_new = copy.deepcopy(paper_term)
    for key in paper_term.keys():
        if key not in paper_id:
            paper_term_new.pop(key)
    ei_pt = _edge_dict_to_index(paper_term_new)
    term_paper_new = _edge_index_to_dict([ei_pt[1], ei_pt[0]])

    paper_conf_new = copy.deepcopy(paper_conf)
    for key in paper_conf.keys():
        if key not in paper_id:
            paper_conf_new.pop(key)
    ei_pc = _edge_dict_to_index(paper_conf_new)
    conf_paper_new = _edge_index_to_dict([ei_pc[1], ei_pc[0]])

    term_id = list(term_paper_new.keys())
    conf_id = list(conf_paper_new.keys())

    paper_dict_new = defaultdict(list)
    for i, pid in enumerate(paper_id):
        paper_dict_new[pid].append(i)
    term_dict_new = defaultdict(list)
    for i, tid in enumerate(term_id):
        term_dict_new[tid].append(i)
    conf_dict_new = defaultdict(list)
    for i, cid in enumerate(conf_id):
        conf_dict_new[cid].append(i)

    for key, val in author_paper.items():
        if key not in author_arr_new:
            author_paper_new.pop(key)

    # ------- author features: bag-of-terms -------
    num_authors = len(author_arr_new)
    num_terms = len(term_arr)
    features = torch.zeros(num_authors, num_terms)
    for author, papers in author_paper_new.items():
        for paper in papers:
            for term in paper_term_new[paper]:
                features[author_dict_new[author][0]][term_dict[term]] += 1

    # ------- rebuild author->paper / paper->author with new indices -------
    ei_ap = _edge_dict_to_index(author_paper_new)
    ei_pa = torch.stack([ei_ap[1], ei_ap[0]], dim=0)
    paper_author_new = _edge_index_to_dict(ei_pa)

    # ------- assemble DGL hetero graph (replaces upstream PyG HeteroData) -------
    ap = _id_to_index(_edge_dict_to_index(author_paper_new),
                      author_dict_new, paper_dict_new)
    pa = _id_to_index(_edge_dict_to_index(paper_author_new),
                      paper_dict_new, author_dict_new)
    tp = _id_to_index(_edge_dict_to_index(term_paper_new),
                      term_dict_new, paper_dict_new)
    pt = _id_to_index(_edge_dict_to_index(paper_term_new),
                      paper_dict_new, term_dict_new)
    cp = _id_to_index(_edge_dict_to_index(conf_paper_new),
                      conf_dict_new, paper_dict_new)
    pc = _id_to_index(_edge_dict_to_index(paper_conf_new),
                      paper_dict_new, conf_dict_new)

    data_dict = {
        ('author', 'to', 'paper'):     (ap[0], ap[1]),
        ('paper',  'to', 'author'):    (pa[0], pa[1]),
        ('term',   'to', 'paper'):     (tp[0], tp[1]),
        ('paper',  'to', 'term'):      (pt[0], pt[1]),
        ('conference', 'to', 'paper'): (cp[0], cp[1]),
        ('paper',  'to', 'conference'):(pc[0], pc[1]),
    }
    num_nodes_dict = {
        'author': num_authors,
        'paper':  len(paper_id),
        'term':   num_terms,
        'conference': len(conf_id),
    }
    g = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict)

    # ------- attach features and labels -------
    g.nodes['author'].data['x'] = features
    y = np.load(p("APA.npz"))['y']
    labels = torch.from_numpy(y).float()
    g.nodes['author'].data['y'] = labels
    # OpenHGNN convention: features should also be reachable via ndata['h']
    g.nodes['author'].data['h'] = features

    return g, features, labels, 4


# ---------------------------------------------------------------------------
# Helpers used by the trainerflow to build normalised metapath adjacencies
# ---------------------------------------------------------------------------
def _from_edge_index_to_adj(edge_index, num_nodes):
    """Dense (n, n) adjacency from a 2 x m edge index, no self-loops added."""
    adj = torch.zeros(num_nodes, num_nodes)
    adj[edge_index[0], edge_index[1]] = 1.0
    return adj


def _gcn_norm(adj):
    """Symmetric normalisation D^{-1/2}(A + I)D^{-1/2}.

    Matches utils_.gcn_norm in the upstream HGDL repo exactly.
    """
    n = adj.shape[0]
    eye = torch.eye(n, device=adj.device, dtype=adj.dtype)
    adj = adj + eye
    deg = torch.sum(adj, dim=1).pow(-0.5)
    deg = torch.where(torch.isinf(deg), torch.zeros_like(deg), deg)
    D = torch.diag(deg)
    return D @ adj @ D


def _build_adj_list(g, meta_paths):
    """Build a list of gcn_norm-ed dense adjacencies, one per metapath."""
    adj_list = []
    n = g.num_nodes('author')
    for mp in meta_paths:
        sub = dgl.metapath_reachable_graph(g, mp)
        src, dst = sub.edges()
        ei = torch.stack([src, dst], dim=0)
        adj = _from_edge_index_to_adj(ei, n)
        adj = _gcn_norm(adj)
        adj_list.append(adj)
    return adj_list


# ---------------------------------------------------------------------------
# OpenHGNN dataset class
# ---------------------------------------------------------------------------
@register_dataset('hgdl_node_classification')
class HGDLDataset(NodeClassificationDataset):
    """DBLP dataset for the HGDL paper.

    Inherits ``NodeClassificationDataset`` so it conforms to the standard
    ``node_classification`` task interface. Beyond the base attributes
    (``g`` / ``category`` / ``num_classes`` / ``has_feature`` /
    ``multi_label`` / ``meta_paths_dict``), this dataset exposes:

    - ``features``    : (n_author, n_term) bag-of-terms features
    - ``labels``      : (n_author, 4) label distributions
    - ``adj_list``    : list of k gcn_norm-ed (n, n) dense adjacencies
    - ``train_idx`` / ``valid_idx`` / ``test_idx`` : long tensors
    - ``train_mask`` / ``val_mask`` / ``test_mask`` : bool tensors

    The labels are real-valued distributions (each row sums to 1), so the
    standard cross-entropy / F1 evaluation pipeline does not apply. The
    HGDL trainerflow handles loss (KL) and evaluation (six LDL metrics)
    directly and does not call task.get_loss_fn / task.evaluate.
    """

    # DBLP metapaths used by HGDL: APCPA and APTPA.
    _META_PATHS = [
        [('author', 'to', 'paper'),
         ('paper', 'to', 'conference'),
         ('conference', 'to', 'paper'),
         ('paper', 'to', 'author')],
        [('author', 'to', 'paper'),
         ('paper', 'to', 'term'),
         ('term', 'to', 'paper'),
         ('paper', 'to', 'author')],
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_name = kwargs.get('dataset_name', 'dblp4HGDL')
        # Default seed matches the upstream HGDL config (seed=0).
        self.seed = kwargs.get('seed', 0)
        # Set node_classification-style attributes consumed by BaseFlow.
        self.category = 'author'
        self.num_classes = 4
        self.has_feature = True
        self.multi_label = False
        self.meta_paths_dict = {
            'APCPA': self._META_PATHS[0],
            'APTPA': self._META_PATHS[1],
        }
        self._load()

    # ------------------------------------------------------------------
    def _load(self):
        """Load DBLP and build everything the trainerflow needs.

        Reseeds RNGs in the exact order used by upstream
        ``load_data_transformer.load_data`` so the 40/10/50 author split
        is bit-perfect identical across runs.
        """
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        dgl.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)

        data_dir = os.path.join(os.path.dirname(__file__), 'DBLP')
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(
                f"DBLP data directory not found at {data_dir}. "
                f"Place APA.npz and the 8 .txt files under {data_dir}/.")

        g, features, labels, num_classes = _build_dblp_hg(data_dir)
        self.g = g
        self.features = features
        self.labels = labels
        self.num_classes = num_classes

        # 40 / 10 / 50 split on authors, identical to upstream load_dblp().
        num_nodes = features.shape[0]
        float_mask = np.random.permutation(np.linspace(0, 1, num_nodes))
        train_idx_np = np.where(float_mask <= 0.4)[0]
        val_idx_np = np.where((float_mask > 0.4) & (float_mask <= 0.5))[0]
        test_idx_np = np.where(float_mask > 0.5)[0]

        self.train_idx = torch.from_numpy(train_idx_np).long()
        self.valid_idx = torch.from_numpy(val_idx_np).long()
        self.test_idx = torch.from_numpy(test_idx_np).long()
        self.train_mask = self._idx_to_bool_mask(train_idx_np, num_nodes)
        self.val_mask = self._idx_to_bool_mask(val_idx_np, num_nodes)
        self.test_mask = self._idx_to_bool_mask(test_idx_np, num_nodes)

        # Precompute metapath adjacencies (gcn_norm-ed, dense, on CPU).
        self.adj_list = _build_adj_list(self.g, self._META_PATHS)

    @staticmethod
    def _idx_to_bool_mask(idx, total):
        m = torch.zeros(total, dtype=torch.bool)
        m[idx] = True
        return m

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------
    # The base class's get_labels expects integer labels stored under
    # ndata['labels'] / ndata['label']; HGDL's labels are real-valued
    # distributions in ndata['y']. Return the cached tensor directly.
    def get_labels(self):
        return self.labels

    def get_split(self, validation=True):
        return self.train_idx, self.valid_idx, self.test_idx
# ===================================================================
# ACM dataset for HGDL
# ===================================================================

def _build_acm_hg(data_dir):
    """Build ACM heterogeneous graph from pre-saved npy/npz files."""
    features = np.load(os.path.join(data_dir, 'features.npy'))
    labels = np.load(os.path.join(data_dir, 'labels.npy'))
    edges_data = np.load(os.path.join(data_dir, 'edges.npz'))

    features = torch.from_numpy(features).float()
    labels = torch.from_numpy(labels).float()

    # Parse edge arrays back to DGL format
    edge_dict = {}
    # Map from saved key format "src__to__dst" to DGL tuple format
    key_map = {
        'author__to__paper': ('author', 'to', 'paper'),
        'paper__to__author': ('paper', 'to', 'author'),
        'paper__to__conference': ('paper', 'to', 'conference'),
        'conference__to__paper': ('conference', 'to', 'paper'),
        'paper__to__subjects': ('paper', 'to', 'subjects'),
        'subjects__to__paper': ('subjects', 'to', 'paper'),
        'paper__to__proceedings': ('paper', 'to', 'proceedings'),
        'proceedings__to__paper': ('proceedings', 'to', 'paper'),
        'author__to__affiliation': ('author', 'to', 'affiliation'),
        'affiliation__to__author': ('affiliation', 'to', 'author'),
    }
    data_dict = {}
    for saved_key, dgl_key in key_map.items():
        ei = edges_data[saved_key]
        data_dict[dgl_key] = (torch.from_numpy(ei[0]).long(),
                              torch.from_numpy(ei[1]).long())

    g = dgl.heterograph(data_dict)
    g.nodes['author'].data['x'] = features
    g.nodes['author'].data['y'] = labels
    g.nodes['author'].data['h'] = features

    return g, features, labels, labels.shape[1]


@register_dataset('hgdl_acm')
class HGDLACMDataset(NodeClassificationDataset):
    """ACM dataset for the HGDL paper (NeurIPS 2024).

    Same interface as HGDLDataset but for ACM: 5810 authors, 14-class
    conference label distributions, 3 metapaths (APA, AAfA, APSPA),
    70/10/20 train/val/test split.
    """

    _META_PATHS = [
        [('author', 'to', 'paper'), ('paper', 'to', 'author')],
        [('author', 'to', 'affiliation'), ('affiliation', 'to', 'author')],
        [('author', 'to', 'paper'), ('paper', 'to', 'subjects'),
         ('subjects', 'to', 'paper'), ('paper', 'to', 'author')],
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_name = kwargs.get('dataset_name', 'acm4HGDL')
        self.seed = kwargs.get('seed', 0)
        self.category = 'author'
        self.num_classes = 14
        self.has_feature = True
        self.multi_label = False
        self.meta_paths_dict = {
            'APA': self._META_PATHS[0],
            'AAfA': self._META_PATHS[1],
            'APSPA': self._META_PATHS[2],
        }
        self._load()

    def _load(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        dgl.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)

        data_dir = os.path.join(os.path.dirname(__file__), 'ACM')
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(
                f"ACM data directory not found at {data_dir}.")

        g, features, labels, num_classes = _build_acm_hg(data_dir)
        self.g = g
        self.features = features
        self.labels = labels
        self.num_classes = num_classes

        # 70 / 10 / 20 split, matching upstream load_acm3()
        num_nodes = features.shape[0]
        float_mask = np.random.permutation(np.linspace(0, 1, num_nodes))
        train_idx_np = np.where(float_mask <= 0.7)[0]
        val_idx_np = np.where((float_mask > 0.7) & (float_mask <= 0.8))[0]
        test_idx_np = np.where(float_mask > 0.8)[0]

        self.train_idx = torch.from_numpy(train_idx_np).long()
        self.valid_idx = torch.from_numpy(val_idx_np).long()
        self.test_idx = torch.from_numpy(test_idx_np).long()
        self.train_mask = self._idx_to_bool_mask(train_idx_np, num_nodes)
        self.val_mask = self._idx_to_bool_mask(val_idx_np, num_nodes)
        self.test_mask = self._idx_to_bool_mask(test_idx_np, num_nodes)

        self.adj_list = _build_adj_list(self.g, self._META_PATHS)

    @staticmethod
    def _idx_to_bool_mask(idx, total):
        m = torch.zeros(total, dtype=torch.bool)
        m[idx] = True
        return m

    def get_labels(self):
        return self.labels

    def get_split(self, validation=True):
        return self.train_idx, self.valid_idx, self.test_idx
