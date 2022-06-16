import torch as th
import os
from dgl.data import DGLBuiltinDataset
from dgl.data.utils import load_graphs, save_graphs

__all__ = ['ICDM2022Dataset']


class ICDM2022Dataset(DGLBuiltinDataset):
    def __init__(self, item_embedding_dim=50, non_item_embedding_dim=30, raw_dir=None, force_reload=False,
                 verbose=False,
                 transform=None):
        name = 'icdm2022'
        self.item_embedding_dim = item_embedding_dim
        self.non_item_embedding_dim = non_item_embedding_dim

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
        graph_path = os.path.join(self.save_path, 'graph.bin')
        save_graphs(graph_path, self._g)

    def load(self):
        graph_path = os.path.join(self.save_path, 'graph.bin')
        gs, _ = load_graphs(graph_path)
        g = gs[0]
        for ntype in g.ntypes:
            if ntype == 'item':
                g.nodes[ntype].data['h'] = th.rand(g.num_nodes(ntype), self.item_embedding_dim, dtype=th.float16)
            else:
                g.nodes[ntype].data['h'] = th.rand(g.num_nodes(ntype), self.non_item_embedding_dim, dtype=th.float16)
        g.nodes[self.category].data['label'] = th.randint(low=0, high=self.num_classes,
                                                          size=(g.num_nodes(self.category),))
        split_ratio = [0.8, 0.1, 0.1]
        num = g.num_nodes(self.category)
        train_mask = th.zeros(num).bool()
        train_mask[0: int(split_ratio[0] * num)] = True
        val_mask = th.zeros(num).bool()
        val_mask[int(split_ratio[0] * num): int((split_ratio[0] + split_ratio[1]) * num)] = True
        test_mask = th.zeros(num).bool()
        test_mask[int((split_ratio[0] + split_ratio[1]) * num):] = True
        g.nodes[self.category].data['train_mask'] = train_mask
        g.nodes[self.category].data['val_mask'] = val_mask
        g.nodes[self.category].data['test_mask'] = test_mask
        self._g = gs[0]

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
