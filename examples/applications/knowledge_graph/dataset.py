from openhgnn.dataset import generate_random_hg
from dgl import transforms as T
from dgl import DGLHeteroGraph
from dgl.data import DGLDataset

meta_paths_dict = {'APA': [('author', 'author-paper', 'paper'), ('paper', 'rev_author-paper', 'author')]}
target_link = [('author', 'author-paper', 'paper')]


class MyDataset(DGLDataset):
    def __init__(self):
        super().__init__(name='my-dataset')

    def process(self):
        self._g = generate_random_citation_hg()

    @property
    def meta_paths_dict(self):
        return meta_paths_dict

    def __getitem__(self, idx):
        return self._g

    def __len__(self):
        return 1


def generate_random_citation_hg() -> DGLHeteroGraph:
    num_edges_dict = {
        ('author', 'author-paper', 'paper'): 10000,
        ('author', 'author-paper1', 'paper'): 10000
    }
    num_nodes_dict = {
        'paper': 1000,
        'author': 100,
    }

    hg = generate_random_hg(num_nodes_dict=num_nodes_dict, num_edges_dict=num_edges_dict)
    transform = T.Compose([T.ToSimple(), T.AddReverse()])
    hg = transform(hg)
    return hg
