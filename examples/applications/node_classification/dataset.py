from openhgnn.dataset import generate_random_hg
from dgl import transforms as T
from dgl import DGLHeteroGraph
from dgl.data import DGLDataset
import torch as th

category = 'author'
meta_paths_dict = {'APA': [('author', 'author-paper', 'paper'), ('paper', 'rev_author-paper', 'author')]}


class MyNCDataset(DGLDataset):
    def __init__(self):
        super().__init__(name='my-nc-dataset')

    def process(self):
        # Generate a random heterogeneous graph with labels on target node type.
        self._g = generate_random_citation_hg()

    # Some models require meta paths, you can set meta path dict for this dataset.
    @property
    def meta_paths_dict(self):
        return meta_paths_dict

    def __getitem__(self, idx):
        return self._g

    def __len__(self):
        return 1


def generate_random_citation_hg() -> DGLHeteroGraph:
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

    num = hg.num_nodes(category)
    hg = transform(hg)
    hg.nodes[category].data['label'] = th.randint(low=0, high=num_classes, size=(num,))
    hg.nodes[category].data['label_mask'] = th.zeros(num).bool()
    hg.nodes[category].data['label_mask'][0: int(0.8 * num)] = True

    return hg
