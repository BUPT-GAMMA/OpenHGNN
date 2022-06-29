import torch
import os
from dgl.data import DGLBuiltinDataset
from dgl.data.utils import load_graphs, save_graphs
import pickle, csv
import tqdm
import numpy as np

__all__ = ['ICDM2022Dataset']

class ICDM2022Dataset(DGLBuiltinDataset):
    r"""ICDM 2022 Dataset.

    Parameters
    ----------
    scale : str
        'small' or 'large'.

    """

    def __init__(self, scale='small', load_feature=True, raw_dir=None,
                 force_reload=False,
                 verbose=False,
                 transform=None):
        name = 'icdm2022_{}'.format(scale)
        self.load_feature = load_feature

        super(ICDM2022Dataset, self).__init__(
            name,
            url='https://s3.cn-north-1.amazonaws.com.cn/dgl-data/dataset/openhgnn/{}.zip'.format(name),
            raw_dir=raw_dir,
            force_reload=force_reload, verbose=verbose, transform=transform)

    def process(self):
        self.load()

    def has_cache(self):
        graph_path = os.path.join(self.save_path, 'graph.bin')
        return os.path.exists(graph_path)

    def save(self):
        pass
        # graph_path = os.path.join(self.save_path, 'graph.bin')
        # save_graphs(graph_path, self._g)

    def load(self):
        # load graph
        print('loading dataset...')
        orig_graph_path = os.path.join(self.save_path, '{}.graph.dgl'.format(self.name))
        gs, _ = load_graphs(orig_graph_path)
        g = gs[0]
        # load node map
        nodes_path = os.path.join(self.save_path, '{}.nodes.dgl'.format(self.name))
        with open(nodes_path, 'rb') as f:
            nodes_info = pickle.load(f)

        if self.load_feature:
            for ntype, embedding_dict in nodes_info['embeds'].items():
                dim = embedding_dict[0].shape[0]
                g.nodes[ntype].data['h'] = torch.empty(g.num_nodes(ntype), dim)
                for nid, embedding in tqdm.tqdm(embedding_dict.items()):
                    g.nodes[ntype].data['h'][nid] = torch.from_numpy(embedding)

        # load label
        labels_path = os.path.join(self.save_path, '{}_labels.csv'.format(self.name))
        labels = torch.tensor([float('nan')] * g.num_nodes(self.category))
        with open(labels_path, 'r') as f:
            csvreader = csv.reader(f)
            item_maps = nodes_info['maps']['item']
            for row in csvreader:
                orig_id = int(row[0])
                new_id = item_maps.get(orig_id)
                if new_id is not None:
                    labels[new_id] = int(row[1])

        label_mask = ~torch.isnan(labels)
        label_idx = torch.nonzero(label_mask, as_tuple=False).squeeze()
        g.nodes[self.category].data['label'] = labels.type(torch.int64)
        split_ratio = [0.9, 0.1]
        num_labels = len(label_idx)
        num_nodes = g.num_nodes(self.category)
        train_mask = torch.zeros(num_nodes).bool()
        train_mask[label_idx[0: int(split_ratio[0] * num_labels)]] = True
        val_mask = torch.zeros(num_nodes).bool()
        val_mask[
            label_idx[int(split_ratio[0] * num_labels): int((split_ratio[0] + split_ratio[1]) * num_labels)]] = True
        
        # test_mask = torch.zeros(num_nodes).bool()
        # test_mask[label_idx[int((split_ratio[0] + split_ratio[1]) * num_labels):]] = True
        g.nodes[self.category].data['train_mask'] = train_mask
        g.nodes[self.category].data['val_mask'] = val_mask
        # g.nodes[self.category].data['test_mask'] = test_mask
        # if self.init_emb:
        #     for ntype in g.ntypes:
        #         if ntype == 'item':
        #             g.nodes[ntype].data['h'] = torch.rand(g.num_nodes(ntype), self.item_embedding_dim, dtype=torch.float32)
        #         else:
        #             g.nodes[ntype].data['h'] = torch.rand(g.num_nodes(ntype), self.non_item_embedding_dim,
        #                                                dtype=torch.float32)
        test_idx_path = '/home/icdm/icdm_graph_competition/OpenHGNN/openhgnn/dataset/test_ids.txt'
        test_mask = torch.zeros(num_nodes).bool()
        with open(test_idx_path, 'r') as f:
            txtreader = np.loadtxt(f, dtype = np.int32)
            item_maps = nodes_info['maps']['item']

            for row in txtreader:
                orig_id = int(row)   
                new_id = item_maps.get(orig_id)
                if new_id is not None:
                    # test_mask[label_idx[int((split_ratio[0] + split_ratio[1]) * num_labels):]] = True
                    # print(new_id)
                    test_mask[new_id] = True
        
        g.nodes[self.category].data['test_mask'] = test_mask
        self._g = g

        print('finish loading dataset')

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


