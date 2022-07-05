import torch
import torch as th
import os
from dgl.data import DGLBuiltinDataset
from dgl.data.utils import load_graphs, save_graphs, save_info
import pickle, csv
import tqdm
import numpy as np

__all__ = ['ICDM2022Dataset']


class ICDM2022Dataset(DGLBuiltinDataset):
    r"""ICDM 2022 Dataset.

    Parameters
    ----------
    session : str
        'small', 'session1' or 'session2'. The small one is only used for debug.
    load_features : bool
        Whether to load features to the graph. Default: True
    load_labels : bool
        Whether to load labels to the graph. For session2, there is no labels. Default: True
    raw_dir : str
        Specifying the directory that will store the
        downloaded data or the directory that
        already stores the input data.
        Default: ~/.dgl/
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose : bool
        Whether to print out progress information. Default: False
    transform : callable, optional
        A transform that takes in a :class:`~dgl.DGLGraph` object and returns
        a transformed version. The :class:`~dgl.DGLGraph` object will be
        transformed before every access.
    """

    def __init__(self, session='small', load_features=True, load_labels=True, raw_dir=None, force_reload=False,
                 verbose=False, transform=None):
        name = 'icdm2022_{}'.format(session)
        self.load_features = load_features
        self.load_labels = load_labels

        super(ICDM2022Dataset, self).__init__(
            name,
            url='https://s3.cn-north-1.amazonaws.com.cn/dgl-data/dataset/openhgnn/{}.zip'.format(name),
            raw_dir=raw_dir,
            force_reload=force_reload, verbose=verbose, transform=transform)

    def process(self):

        # load graph
        if self.verbose:
            print('loading dataset...')
        orig_graph_path = os.path.join(self.save_path, '{}.graph.dgl'.format(self.name))
        gs, _ = load_graphs(orig_graph_path)
        g = gs[0]

        nodes_info, self._item_map, self._rev_item_map = self._load_map()
        if self.load_features:
            for ntype, embedding_dict in nodes_info['embeds'].items():
                dim = embedding_dict[0].shape[0]
                g.nodes[ntype].data['h'] = torch.rand(g.num_nodes(ntype), dim)
                for nid, embedding in tqdm.tqdm(embedding_dict.items()):
                    g.nodes[ntype].data['h'][nid] = torch.from_numpy(embedding)

        # load label
        num_nodes = g.num_nodes(self.category)
        if self.load_labels:
            labels_path = os.path.join(self.save_path, '{}_labels.csv'.format(self.name))
            labels = th.tensor([float('nan')] * g.num_nodes(self.category))
            with open(labels_path, 'r') as f:
                csvreader = csv.reader(f)
                for row in csvreader:
                    orig_id = int(row[0])
                    new_id = self._item_map.get(orig_id)
                    if new_id is not None:
                        labels[new_id] = int(row[1])
            label_mask = ~th.isnan(labels)
            label_idx = th.nonzero(label_mask, as_tuple=False).squeeze()
            g.nodes[self.category].data['label'] = labels.type(th.int64)

            # label_idx = np.random.permutation(np.array(label_idx))  # shuffle the label index
            split_ratio = [0.8, 0.2]
            num_labels = len(label_idx)
            train_mask = th.zeros(num_nodes).bool()
            train_mask[label_idx[0: int(split_ratio[0] * num_labels)]] = True
            val_mask = th.zeros(num_nodes).bool()
            val_mask[
                label_idx[int(split_ratio[0] * num_labels): int((split_ratio[0] + split_ratio[1]) * num_labels)]] = True
            g.nodes[self.category].data['train_mask'] = train_mask
            g.nodes[self.category].data['val_mask'] = val_mask

        # load test_idx
        test_idx_path = os.path.join(self.save_path, '{}_test_ids.csv'.format(self.name))
        test_mask = th.zeros(num_nodes).bool()
        with open(test_idx_path, 'r') as f:
            csvreader = csv.reader(f)
            for row in csvreader:
                orig_id = int(row[0])
                new_id = self._item_map.get(orig_id)
                if new_id is not None:
                    test_mask[new_id] = True
        g.nodes[self.category].data['test_mask'] = test_mask
        self._g = g
        if self.verbose:
            print(self._g)
            print('finish loading dataset')

    def has_cache(self):
        graph_path = os.path.join(self.save_path, 'graph.bin')
        return os.path.exists(graph_path)

    def save(self):
        graph_path = os.path.join(self.save_path, 'graph.bin')
        save_graphs(graph_path, self._g)

    def load(self):
        graph_path = os.path.join(self.save_path, 'graph.bin')
        gs, _ = load_graphs(graph_path)
        self._g = gs[0]
        _, self._item_map, self._rev_item_map = self._load_map()
        if self.verbose:
            print(self._g)

    def _load_map(self):
        # load node map
        nodes_path = os.path.join(self.save_path, '{}.nodes.dgl'.format(self.name))
        with open(nodes_path, 'rb') as f:
            nodes_info = pickle.load(f)

        # item map and reversed item map
        item_map = nodes_info['maps']['item']
        rev_item_map = {}
        for k, v in item_map.items():
            rev_item_map[v] = k
        return nodes_info, item_map, rev_item_map

    @property
    def item_map(self):
        return self._item_map

    @property
    def rev_item_map(self):
        return self._rev_item_map

    @property
    def category(self):
        return 'item'

    @property
    def target_ntype(self):
        return 'item'

    @property
    def num_classes(self):
        return 2

    def __getitem__(self, idx):
        assert idx == 0
        return self._g

    def __len__(self):
        return 1
