import torch as th
from openhgnn import Experiment
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


def generate_random_citation_nc_hg() -> DGLHeteroGraph:
    num_classes = 2
    num_edges_dict = {
        ('author', 'author-paper', 'paper'): 1000
    }
    n = 100
    num_nodes_dict = {
        'paper': 10 * n,
        'author': n,
    }

    hg = generate_random_hg(num_nodes_dict=num_nodes_dict, num_edges_dict=num_edges_dict)
    transform = T.Compose([T.ToSimple(), T.AddReverse()])
    hg = transform(hg)
    hg.nodes[category].data['label'] = th.randint(low=0, high=num_classes, size=(hg.num_nodes(category),))
    for ntype in hg.ntypes:
        hg.nodes[ntype].data['h'] = th.ones(hg.num_nodes(ntype), 500).type(th.float16)
    return hg


if __name__ == '__main__':
    myNCDataset = AsNodeClassificationDataset(MyNCDataset(), target_ntype=category, split_ratio=[0.8, 0.1, 0.1],
                                              force_reload=True)
    experiment = Experiment(model='RGCN', dataset=myNCDataset, task='node_classification', gpu=0, mini_batch_flag=True,
                            max_epoch=5,
                            data_cpu=True, batch_size=32)
    experiment.run()
