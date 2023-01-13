import torch as th
from openhgnn.dataset import generate_random_hg
from dgl import transforms as T
from dgl import DGLHeteroGraph
from dgl.data import DGLDataset

meta_paths_dict = {'APA': [('author', 'author-paper', 'paper'), ('paper', 'rev_author-paper', 'author')]}
target_link = [('author', 'author-paper', 'paper')]
target_link_r = [('paper', 'rev_author-paper', 'author')]


#
# class MyLPDataset(DGLDataset):
#     def __init__(self):
#         super().__init__(name='my-lp-dataset',
#                          force_reload=True)
#
#     def process(self):
#         hg = generate_random_citation_hg()
#         self._g = hg
#
#     @property
#     def meta_paths_dict(self):
#         return meta_paths_dict
#
#     def __getitem__(self, idx):
#         return self._g
#
#     def __len__(self):
#         return 1


class MyLPDatasetWithPredEdges(DGLDataset):
    def __init__(self):
        super().__init__(name='my-lp-dataset',
                         force_reload=True)

    def process(self):
        hg = generate_random_citation_hg()

        self._g = hg
        num = 10
        self._pred_edges = {
            etype: (th.randint(low=0, high=self._g.num_nodes(etype[0]), size=(num,)),
                    th.randint(low=0, high=self._g.num_nodes(etype[2]), size=(num,)))
            for etype in target_link}

    @property
    def meta_paths_dict(self):
        return meta_paths_dict

    @property
    def pred_edges(self):
        return self._pred_edges

    def __getitem__(self, idx):
        return self._g

    def __len__(self):
        return 1


def generate_random_citation_hg() -> DGLHeteroGraph:
    num_edges_dict = {
        ('author', 'author-paper', 'paper'): 100000
    }
    num_nodes_dict = {
        'paper': 1000,
        'author': 100,
    }
    hg = generate_random_hg(num_nodes_dict=num_nodes_dict, num_edges_dict=num_edges_dict)
    transform = T.Compose([T.ToSimple(), T.AddReverse()])
    hg = transform(hg)
    return hg
