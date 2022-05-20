import pickle
import torch as th
import numpy as np
import dgl
import os
from dgl.data import DGLBuiltinDataset
from dgl.data.utils import idx2mask, load_graphs, save_graphs

__all__ = ['IMDB4GTNDataset', 'ACM4GTNDataset', 'DBLP4GTNDataset']


class _GTNDataset(DGLBuiltinDataset):
    r"""GTN Dataset.

    It contains three datasets used in a NeurIPS'19 paper Graph Transformer Networks <https://arxiv.org/abs/1911.06455>,
    which includes two citation network datasets DBLP and ACM, and a movie dataset
    IMDB. DBLP contains three types of nodes (papers (P), authors (A), conferences (C)), four types of edges
    (PA, AP, PC, CP), and research areas of authors as labels. ACM contains three types of nodes (papers
    (P), authors (A), subject (S)), four types of edges (PA, AP, PS, SP), and categories of papers as labels.
    Each node in the two datasets is represented as bag-of-words of keywords. On the other hand, IMDB
    contains three types of nodes (movies (M), actors (A), and directors (D)) and labels are genres of
    movies. Node features are given as bag-of-words representations of plots.

    Dataset statistics:

    Dataset Nodes Edges Edge type Features Training Validation Test

    DBLP 18405 67946 4 334 800 400 2857

    ACM 8994 25922 4 1902 600 300 2125

    IMDB 12772 37288 4 1256 300 300 2339

    Data source link: <https://drive.google.com/file/d/1qOZ3QjqWMIIvWjzrIdRe3EA4iKzPi6S5/view?usp=sharing>

    Examples
    --------
    >>> dataset = IMDB4GTNDataset()
    >>> graph = dataset[0]
    """

    def __init__(self, name, canonical_etypes, target_ntype, raw_dir=None, force_reload=False, verbose=False,
                 transform=None):
        assert name in ['dblp4GTN', 'acm4GTN', 'imdb4GTN']
        self._canonical_etypes = canonical_etypes
        self._target_ntype = target_ntype
        if raw_dir is None:
            raw_dir = './openhgnn/dataset'
        super(_GTNDataset, self).__init__(
            name,
            url='https://s3.cn-north-1.amazonaws.com.cn/dgl-data/dataset/openhgnn/{}.zip'.format(name),
            raw_dir=raw_dir,
            force_reload=force_reload, verbose=verbose, transform=transform)

    def process(self):
        target_ntype = self.target_ntype
        canonical_etypes = self._canonical_etypes

        with open(self.raw_path + '/node_features.pkl', 'rb') as f:
            node_features = pickle.load(f)
        with open(self.raw_path + '/edges.pkl', 'rb') as f:
            edges = pickle.load(f)
        with open(self.raw_path + '/labels.pkl', 'rb') as f:
            labels = pickle.load(f)

        num_nodes = edges[0].shape[0]
        assert len(canonical_etypes) == len(edges)

        ntype_mask = dict()
        ntype_idmap = dict()
        ntypes = set()
        data_dict = {}

        # create dgl graph
        for etype in canonical_etypes:
            ntypes.add(etype[0])
            ntypes.add(etype[2])
        for ntype in ntypes:
            ntype_mask[ntype] = np.zeros(num_nodes, dtype=bool)
            ntype_idmap[ntype] = np.full(num_nodes, -1, dtype=int)
        for i, etype in enumerate(canonical_etypes):
            src_nodes = edges[i].nonzero()[0]
            dst_nodes = edges[i].nonzero()[1]
            src_ntype = etype[0]
            dst_ntype = etype[2]
            ntype_mask[src_ntype][src_nodes] = True
            ntype_mask[dst_ntype][dst_nodes] = True
        for ntype in ntypes:
            ntype_idx = ntype_mask[ntype].nonzero()[0]
            ntype_idmap[ntype][ntype_idx] = np.arange(ntype_idx.size)
        for i, etype in enumerate(canonical_etypes):
            src_nodes = edges[i].nonzero()[0]
            dst_nodes = edges[i].nonzero()[1]
            src_ntype = etype[0]
            dst_ntype = etype[2]
            data_dict[etype] = \
                (th.from_numpy(ntype_idmap[src_ntype][src_nodes]).type(th.int64),
                 th.from_numpy(ntype_idmap[dst_ntype][dst_nodes]).type(th.int64))
        g = dgl.heterograph(data_dict)

        # split and label
        all_label = np.full(g.num_nodes(target_ntype), -1, dtype=int)
        for i, split in enumerate(['train', 'val', 'test']):
            node = np.array(labels[i])[:, 0]
            label = np.array(labels[i])[:, 1]
            all_label[node] = label
            g.nodes[target_ntype].data['{}_mask'.format(split)] = \
                th.from_numpy(idx2mask(node, g.num_nodes(target_ntype))).type(th.bool)
        g.nodes[target_ntype].data['label'] = th.from_numpy(all_label).type(th.long)

        # node feature
        node_features = th.from_numpy(node_features).type(th.FloatTensor)
        for ntype in ntypes:
            idx = ntype_mask[ntype].nonzero()[0]
            g.nodes[ntype].data['h'] = node_features[idx]

        self._g = g
        self._num_classes = len(th.unique(self._g.nodes[self.target_ntype].data['label']))
        self._in_dim = self._g.ndata['h'][self.target_ntype].shape[1]

    def save(self):
        graph_path = os.path.join(self.save_path, 'graph.bin')
        save_graphs(graph_path, self._g)

    def load(self):
        graph_path = os.path.join(self.save_path, 'graph.bin')
        gs, _ = load_graphs(graph_path)
        self._g = gs[0]
        self._num_classes = len(th.unique(self._g.nodes[self.target_ntype].data['label']))
        self._in_dim = self._g.ndata['h'][self.target_ntype].shape[1]

    @property
    def target_ntype(self):
        return self._target_ntype

    @property
    def category(self):
        return self._target_ntype

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def in_dim(self):
        return self._in_dim

    def __getitem__(self, idx):
        assert idx == 0
        return self._g

    def __len__(self):
        return 1


class DBLP4GTNDataset(_GTNDataset):
    def __init__(self, raw_dir=None, force_reload=False, verbose=False, transform=None):
        name = 'dblp4GTN'
        canonical_etypes = [('paper', 'paper-author', 'author'), ('author', 'author-paper', 'paper'),
                            ('paper', 'paper-conference', 'conference'), ('conference', 'conference-paper', 'paper')]
        target_ntype = 'author'
        super(DBLP4GTNDataset, self).__init__(name, canonical_etypes, target_ntype, raw_dir=raw_dir,
                                              force_reload=force_reload, verbose=verbose, transform=transform)

    @property
    def meta_paths_dict(self):
        return {'APCPA': [('author', 'author-paper', 'paper'),
                          ('paper', 'paper-conference', 'conference'),
                          ('conference', 'conference-paper', 'paper'),
                          ('paper', 'paper-author', 'author')],
                'APA': [('author', 'author-paper', 'paper'),
                        ('paper', 'paper-author', 'author')],
                }


class ACM4GTNDataset(_GTNDataset):
    def __init__(self, raw_dir=None, force_reload=False, verbose=False, transform=None):
        name = 'acm4GTN'
        canonical_etypes = [('paper', 'paper-author', 'author'), ('author', 'author-paper', 'paper'),
                            ('paper', 'paper-subject', 'subject'), ('subject', 'subject-paper', 'paper')]
        target_ntype = 'paper'
        super(ACM4GTNDataset, self).__init__(name, canonical_etypes, target_ntype, raw_dir=raw_dir,
                                             force_reload=force_reload, verbose=verbose, transform=transform)

    @property
    def meta_paths_dict(self):
        return {'PAPSP': [('paper', 'paper-author', 'author'),
                          ('author', 'author-paper', 'paper'),
                          ('paper', 'paper-subject', 'subject'),
                          ('subject', 'subject-paper', 'paper')],
                'PAP': [('paper', 'paper-author', 'author'),
                        ('author', 'author-paper', 'paper')],
                'PSP': [('paper', 'paper-subject', 'subject'),
                        ('subject', 'subject-paper', 'paper')]
                }


class IMDB4GTNDataset(_GTNDataset):
    def __init__(self, raw_dir=None, force_reload=False, verbose=False, transform=None):
        name = 'imdb4GTN'
        canonical_etypes = [('movie', 'movie-director', 'director'), ('director', 'director-movie', 'movie'),
                            ('movie', 'movie-actor', 'actor'), ('actor', 'actor-movie', 'movie')]
        target_ntype = 'movie'
        super(IMDB4GTNDataset, self).__init__(name, canonical_etypes, target_ntype, raw_dir=raw_dir,
                                              force_reload=force_reload, verbose=verbose, transform=transform)

    @property
    def meta_paths_dict(self):
        return {'MAM': [('movie', 'movie-actor', 'actor'),
                        ('actor', 'actor-movie', 'movie')],
                'MDM': [('movie', 'movie-director', 'subject'),
                        ('subject', 'subject-movie', 'movie')]
                }
