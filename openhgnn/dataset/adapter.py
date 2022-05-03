"""Dataset adapters for re-purposing a dataset for a different kind of training task."""

import os
import json
import numpy as np
from dgl.data import utils, DGLDataset
from dgl import backend as F
import dgl
from dgl.dataloading.negative_sampler import GlobalUniform, PerSourceUniform
import torch as th

__all__ = ['AsNodeClassificationDataset', 'AsLinkPredictionDataset']


class AsNodeClassificationDataset(DGLDataset):
    """Repurpose a dataset for a standard semi-supervised transductive
    node prediction task.

    The class converts a given dataset into a new dataset object that:

      - Contains only one heterogeneous graph, accessible from ``dataset[0]``.
      - The graph stores:

        - Node labels in ``g.nodes[target_ntype].data['label']``.
        - Train/val/test masks in ``g.nodes[target_ntype].data['train_mask']``, ``g.nodes[target_ntype].data['val_mask']``,
          and ``g.nodes[target_ntype].data['test_mask']`` respectively.

      - In addition, the dataset contains the following attributes:

        - ``num_classes``, the number of classes to predict.
        - ``train_idx``, ``val_idx``, ``test_idx``, train/val/test indexes.

    The class will keep only the first graph in the provided dataset and
    generate train/val/test masks according to the given spplit ratio. The generated
    masks will be cached to disk for fast re-loading. If the provided split ratio
    differs from the cached one, it will re-process the dataset properly.

    Parameters
    ----------
    dataset : DGLDataset
        The dataset to be converted.
    split_ratio : (float, float, float), optional
        Split ratios for training, validation and test sets. Must sum to one.
    target_ntype : str, optional
        The node type to add split mask for.

    Attributes
    ----------
    num_classes : int
        Number of classes to predict.
    train_idx : Tensor
        An 1-D integer tensor of training node IDs.
    val_idx : Tensor
        An 1-D integer tensor of validation node IDs.
    test_idx : Tensor
        An 1-D integer tensor of test node IDs.

    """

    def __init__(self,
                 dataset,
                 split_ratio=None,
                 target_ntype=None,
                 **kwargs):
        self.dataset = dataset
        self.split_ratio = split_ratio
        self.target_ntype = target_ntype
        super().__init__(self.dataset.name + '-as-nodepred',
                         hash_key=(split_ratio, target_ntype, dataset.name, 'nodepred'), **kwargs)

    def process(self):
        is_ogb = hasattr(self.dataset, 'get_idx_split')
        if is_ogb:
            g, label = self.dataset[0]
            self.g = g.clone()
            self.g.ndata['label'] = F.reshape(label, (g.num_nodes(),))
        else:
            self.g = self.dataset[0].clone()

        if 'label' not in self.g.nodes[self.target_ntype].data:
            raise ValueError("Missing node labels. Make sure labels are stored "
                             "under name 'label'.")

        if self.split_ratio is None:
            if is_ogb:
                split = self.dataset.get_idx_split()
                train_idx, val_idx, test_idx = split['train'], split['valid'], split['test']
                n = self.g.num_nodes()
                train_mask = utils.generate_mask_tensor(utils.idx2mask(train_idx, n))
                val_mask = utils.generate_mask_tensor(utils.idx2mask(val_idx, n))
                test_mask = utils.generate_mask_tensor(utils.idx2mask(test_idx, n))
                self.g.ndata['train_mask'] = train_mask
                self.g.ndata['val_mask'] = val_mask
                self.g.ndata['test_mask'] = test_mask
            else:
                assert "train_mask" in self.g.nodes[self.target_ntype].data, \
                    "train_mask is not provided, please specify split_ratio to generate the masks"
                assert "val_mask" in self.g.nodes[self.target_ntype].data, \
                    "val_mask is not provided, please specify split_ratio to generate the masks"
                assert "test_mask" in self.g.nodes[self.target_ntype].data, \
                    "test_mask is not provided, please specify split_ratio to generate the masks"
        else:
            if self.verbose:
                print('Generating train/val/test masks...')
            utils.add_nodepred_split(self, self.split_ratio, self.target_ntype)

        self._set_split_index()

        self.multi_label = getattr(self.dataset, 'multi_label', None)
        if self.multi_label is None:
            self.multi_label = len(self.g.nodes[self.target_ntype].data['label'].shape) == 2

        self.num_classes = getattr(self.dataset, 'num_classes', None)
        if self.num_classes is None:
            if self.multi_label:
                self.num_classes = self.g.nodes[self.target_ntype].data['label'].shape[1]
            else:
                self.num_classes = len(F.unique(self.g.nodes[self.target_ntype].data['label']))

        self.meta_paths = getattr(self.dataset, 'meta_paths', None)
        self.meta_paths_dict = getattr(self.dataset, 'meta_paths_dict', None)

    def has_cache(self):
        return os.path.isfile(os.path.join(self.save_path, 'graph_{}.bin'.format(self.hash)))

    def load(self):
        with open(os.path.join(self.save_path, 'info_{}.json'.format(self.hash)), 'r') as f:
            info = json.load(f)
            if (info['split_ratio'] != self.split_ratio
                    or info['target_ntype'] != self.target_ntype):
                raise ValueError('Provided split ratio is different from the cached file. '
                                 'Re-process the dataset.')
            self.split_ratio = info['split_ratio']
            self.target_ntype = info['target_ntype']
            self.num_classes = info['num_classes']
            self.meta_paths_dict = info['meta_paths_dict']
            self.meta_paths = info['meta_paths']

        gs, _ = utils.load_graphs(os.path.join(self.save_path, 'graph_{}.bin'.format(self.hash)))
        self.g = gs[0]
        self._set_split_index()

    def save(self):
        utils.save_graphs(os.path.join(self.save_path, 'graph_{}.bin'.format(self.hash)), [self.g])
        with open(os.path.join(self.save_path, 'info_{}.json'.format(self.hash)), 'w') as f:
            json.dump({
                'split_ratio': self.split_ratio,
                'target_ntype': self.target_ntype,
                'num_classes': self.num_classes,
                'meta_paths_dict': self.meta_paths_dict,
                'meta_paths': self.meta_paths}, f)

    def __getitem__(self, idx):
        return self.g

    def __len__(self):
        return 1

    def _set_split_index(self):
        """Add train_idx/val_idx/test_idx as dataset attributes according to corresponding mask."""
        ndata = self.g.nodes[self.target_ntype].data
        self.train_idx = F.nonzero_1d(ndata['train_mask'])
        self.val_idx = F.nonzero_1d(ndata['val_mask'])
        self.test_idx = F.nonzero_1d(ndata['test_mask'])

    def get_split(self):
        return self.train_idx, self.val_idx, self.test_idx

    def get_labels(self):
        return self.g.nodes[self.target_ntype].data['label']

    @property
    def category(self):
        return self.target_ntype


class AsLinkPredictionDataset(DGLDataset):
    """Repurpose a dataset for link prediction task.

    The created dataset will include data needed for link prediction.
    It will keep only the first graph in the provided dataset and
    generate train/val/test edges according to the given split ratio,
    and the correspondent negative edges based on the neg_ratio. The generated
    edges will be cached to disk for fast re-loading. If the provided split ratio
    differs from the cached one, it will re-process the dataset properly.

    Parameters
    ----------
    dataset : DGLDataset
        The dataset to be converted.
    split_ratio : (float, float, float), optional
        Split ratios for training, validation and test sets. Must sum to one.
    neg_ratio : int, optional
        Indicate how much negative samples to be sampled
        The number of the negative samples will be equal or less than neg_ratio * num_positive_edges.
    target_link : list[tuple[str, str, str]]
        The edge types on which predictions are make.
    target_link_r : list[tuple[str, str, str]], optional
        The reverse edge types of the target links. Used to remove reverse edges of val/test edges from train graph.
    neg_sampler : str, optional
        Indicate how negative edges of val/test edges are sampled. 'global' or 'per_source'.

    Attributes
    -------
    train_graph: DGLHeteroGraph
        The DGLHeteroGraph for training
    pos_val_graph: DGLHeteroGraph
        The DGLHeteroGraph containing positive validation edges
    pos_test_graph: DGLHeteroGraph
        The DGLHeteroGraph containing positive test edges
    neg_val_graph: DGLHeteroGraph
        The DGLHeteroGraph containing negative validation edges
    neg_test_graph: DGLHeteroGraph
        The DGLHeteroGraph containing negative test edges
    """

    def __init__(self,
                 dataset,
                 target_link,
                 target_link_r,
                 split_ratio=None,
                 neg_ratio=3,
                 neg_sampler='global',
                 **kwargs):
        self.g = dataset[0]
        self.num_nodes = self.g.num_nodes()
        self.dataset = dataset
        self.split_ratio = split_ratio
        self.target_link = target_link
        self.target_link_r = target_link_r
        self.neg_ratio = neg_ratio
        self.neg_sampler = neg_sampler
        super().__init__(dataset.name + '-as-linkpred', hash_key=(
            neg_ratio, target_link, target_link_r, split_ratio, neg_sampler, dataset.name, 'linkpred'), **kwargs)

    def process(self):
        if self.split_ratio is None:
            for etype in self.target_link:
                for mask in ['train_mask', 'val_mask', 'test_mask']:
                    assert mask in self.g.edges[etype].data, \
                        "{} is not provided for edge type {}, please specify split_ratio to generate the masks".format(
                            mask, etype)

        else:
            ratio = self.split_ratio
            for etype in self.target_link:
                n = self.g.num_edges(etype)
                n_train, n_val, n_test = int(n * ratio[0]), int(n * ratio[1]), int(n * ratio[2])
                idx = np.random.permutation(n)
                train_idx = idx[:n_train]
                val_idx = idx[n_train:n_train + n_val]
                test_idx = idx[n_train + n_val:]
                train_mask = th.zeros(n).bool()
                train_mask[train_idx] = True
                val_mask = th.zeros(n).bool()
                val_mask[val_idx] = True
                test_mask = th.zeros(n).bool()
                test_mask[test_idx] = True
                self.g.edges[etype].data['train_mask'] = train_mask
                self.g.edges[etype].data['val_mask'] = val_mask
                self.g.edges[etype].data['test_mask'] = test_mask

        # create val and test graph(pos and neg respectively)
        self.pos_val_graph, self.neg_val_graph = self._get_pos_and_neg_graph('val')
        self.pos_test_graph, self.neg_test_graph = self._get_pos_and_neg_graph('test')

        # create train graph
        train_graph = self.g
        for i, etype in enumerate(self.target_link):
            # remove val and test edges
            train_graph = dgl.remove_edges(train_graph,
                                           th.cat((self.pos_val_graph.edges[etype].data[dgl.EID],
                                                   self.pos_test_graph.edges[etype].data[dgl.EID])),
                                           etype)
            # remove reverse edges of val and test edges
            if self.target_link_r is not None:
                reverse_etype = self.target_link_r[i]
                train_graph = dgl.remove_edges(train_graph, th.arange(train_graph.num_edges(reverse_etype)),
                                               reverse_etype)
                edges = train_graph.edges(etype=etype)
                train_graph = dgl.add_edges(train_graph, edges[1], edges[0], etype=reverse_etype)
        self.train_graph = train_graph

        self.meta_paths = getattr(self.dataset, 'meta_paths', None)
        self.meta_paths_dict = getattr(self.dataset, 'meta_paths_dict', None)

    def _get_pos_and_neg_graph(self, split):
        if self.neg_sampler == 'global':
            neg_sampler = GlobalUniform(self.neg_ratio)
        elif self.neg_sampler == 'per_source':
            neg_sampler = PerSourceUniform(self.neg_ratio)
        else:
            raise ValueError('Unsupported neg_sampler')
        edges = {
            etype: th.nonzero(self.g.edges[etype].data['{}_mask'.format(split)]).squeeze()
            for etype in self.target_link}
        pos_graph = dgl.edge_subgraph(self.g, edges, relabel_nodes=False, store_ids=True)
        neg_edges = getattr(self.dataset, 'neg_{}_edges'.format(split), neg_sampler(self.g, edges))
        neg_graph = dgl.heterograph(neg_edges, {ntype: pos_graph.num_nodes(ntype) for ntype in pos_graph.ntypes})
        return pos_graph, neg_graph

    def has_cache(self):
        return os.path.isfile(os.path.join(self.save_path, 'graph_{}.bin'.format(self.hash)))

    def load(self):
        gs, _ = utils.load_graphs(
            os.path.join(self.save_path, 'graph_{}.bin'.format(self.hash)))

        self.train_graph, self.pos_val_graph, self.pos_test_graph, self.neg_val_graph, self.neg_test_graph = \
            gs[0], gs[1], gs[2], gs[3], gs[4]

        with open(os.path.join(self.save_path, 'info_{}.json'.format(self.hash)), 'r') as f:
            info = json.load(f)
            self.split_ratio = info["split_ratio"]
            self.neg_ratio = info["neg_ratio"]
            self.target_link = info["target_link"]
            self.target_link_r = info["target_link_r"]
            self.neg_sampler = info["neg_sampler"]
            self.meta_paths_dict = info["meta_paths_dict"]
            self.meta_paths = info["meta_paths"]

    def save(self):
        utils.save_graphs(os.path.join(self.save_path, 'graph_{}.bin'.format(self.hash)),
                          [self.train_graph, self.pos_val_graph, self.pos_test_graph, self.neg_val_graph,
                           self.neg_test_graph])
        with open(os.path.join(self.save_path, 'info_{}.json'.format(self.hash)), 'w') as f:
            json.dump({
                'split_ratio': self.split_ratio,
                'neg_ratio': self.neg_ratio,
                'target_link': self.target_link,
                'target_link_r': self.target_link_r,
                'neg_sampler': self.neg_sampler,
                'meta_paths_dict': self.meta_paths_dict,
                'meta_paths': self.meta_paths,
            }, f)

    def get_split(self):
        return self.train_graph, self.pos_val_graph, self.pos_test_graph, self.neg_val_graph, self.neg_test_graph

    def __getitem__(self, idx):
        return self.g

    def __len__(self):
        return 1
