import torch as th
from openhgnn.dataset import AsNodeClassificationDataset, generate_random_hg
from dgl import transforms as T
from dgl import DGLHeteroGraph
from dgl.data import DGLDataset

category = 'author'
meta_paths_dict = {'APA': [('author', 'author-paper', 'paper'), ('paper', 'rev_author-paper', 'author')]}


class MyNCDataset(DGLDataset):
    def __init__(self):
        super().__init__(name='my-nc-dataset', force_reload=True)

    def process(self):
        # Generate a random heterogeneous graph with labels on target node type.
        self._g = generate_random_citation_nc_hg()

    # Some models require meta paths, you can set meta path dict for this dataset.
    @property
    def meta_paths_dict(self):
        return meta_paths_dict

    def __getitem__(self, idx):
        return self._g

    def __len__(self):
        return 1


class MyMultiLabelNCDataset(DGLDataset):
    def __init__(self):
        super().__init__(name='my-multi-label-nc-dataset', force_reload=True)

    def process(self):
        # Generate a random multi-label heterogeneous graph, which indicates that one node can
        # have more than 1 labels.
        self._g = generate_random_citation_nc_hg(multi_label=True)

    @property
    def meta_paths_dict(self):
        return meta_paths_dict

    def __getitem__(self, idx):
        return self._g

    def __len__(self):
        return 1


class MySplitNCDataset(DGLDataset):
    def __init__(self):
        super().__init__(name='my-split-nc-dataset',
                         force_reload=True)

    def process(self):
        hg = generate_random_citation_nc_hg()

        # Optionally you can specify the split masks on your own, otherwise we will
        # automatically split them randomly by the input split ratio to AsNodeClassificationDataset.
        split_ratio = [0.8, 0.1, 0.1]
        num = hg.num_nodes(category)
        train_mask = th.zeros(num).bool()
        train_mask[0: int(split_ratio[0] * num)] = True
        val_mask = th.zeros(num).bool()
        val_mask[int(split_ratio[0] * num): int((split_ratio[0] + split_ratio[1]) * num)] = True
        test_mask = th.zeros(num).bool()
        test_mask[int((split_ratio[0] + split_ratio[1]) * num):] = True
        hg.nodes[category].data['train_mask'] = train_mask
        hg.nodes[category].data['val_mask'] = val_mask
        hg.nodes[category].data['test_mask'] = test_mask
        self._g = hg

    @property
    def meta_paths_dict(self):
        return meta_paths_dict

    def __getitem__(self, idx):
        return self._g

    def __len__(self):
        return 1


def generate_random_citation_nc_hg(multi_label=False) -> DGLHeteroGraph:
    num_classes = 5
    num_edges_dict = {
        ('author', 'author-paper', 'paper'): 10000
    }
    num_nodes_dict = {
        'paper': 1000,
        'author': 100,
    }
    hg = generate_random_hg(num_nodes_dict=num_nodes_dict, num_edges_dict=num_edges_dict)
    transform = T.Compose([T.ToSimple(), T.AddReverse()])
    hg = transform(hg)
    if multi_label:
        hg.nodes[category].data['label'] = th.randint(low=0, high=2, size=(hg.num_nodes(category), num_classes),
                                                      dtype=th.float)
    else:
        hg.nodes[category].data['label'] = th.randint(low=0, high=num_classes, size=(hg.num_nodes(category),))
    return hg


def train_with_custom_nc_dataset(dataset):
    from openhgnn.config import Config
    from openhgnn.start import OpenHGNN
    config_file = ["../../openhgnn/config.ini"]
    config = Config(file_path=config_file, model='HAN', dataset=dataset, task='node_classification', gpu=-1)
    OpenHGNN(args=config)


if __name__ == '__main__':
    myNCDataset = AsNodeClassificationDataset(MyNCDataset(), target_ntype=category, split_ratio=[0.8, 0.1, 0.1],
                                              force_reload=True)
    train_with_custom_nc_dataset(myNCDataset)

    mySplitNCDataset = AsNodeClassificationDataset(MySplitNCDataset(), target_ntype=category, force_reload=True)
    train_with_custom_nc_dataset(mySplitNCDataset)

    myMultiLabelNCDataset = AsNodeClassificationDataset(MyMultiLabelNCDataset(), target_ntype=category,
                                                        split_ratio=[0.8, 0.1, 0.1], force_reload=True)
    train_with_custom_nc_dataset(myMultiLabelNCDataset)
