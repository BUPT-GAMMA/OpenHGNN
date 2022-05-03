import torch as th
from openhgnn.dataset import AsLinkPredictionDataset, generate_random_hg
from dgl import transforms as T
from dgl import DGLHeteroGraph
from dgl.data import DGLDataset
from dgl.dataloading.negative_sampler import GlobalUniform

meta_paths_dict = {'APA': [('author', 'author-paper', 'paper'), ('paper', 'rev_author-paper', 'author')]}
target_link = [('author', 'author-paper', 'paper')]
target_link_r = [('paper', 'rev_author-paper', 'author')]


class MyLPDataset(DGLDataset):
    def __init__(self):
        super().__init__(name='my-lp-dataset', force_reload=True)

    def process(self):
        # Generate a random heterogeneous graph with labels on target node type.
        hg = generate_random_citation_hg()
        self._g = hg

    # Some models require meta paths, you can set meta path dict for this dataset.
    @property
    def meta_paths_dict(self):
        return meta_paths_dict

    def __getitem__(self, idx):
        return self._g

    def __len__(self):
        return 1


class MySplitLPDatasetWithNegEdges(DGLDataset):
    def __init__(self):
        super().__init__(name='my-split-lp-dataset-with-neg-edges',
                         force_reload=True)

    def process(self):
        hg = generate_random_citation_hg()

        # Optionally you can specify the split masks on your own, otherwise we will
        # automatically split them randomly by the input split ratio to AsLinkPredictionDataset.
        split_ratio = [0.8, 0.1, 0.1]
        for etype in target_link:
            num = hg.num_edges(etype)
            train_mask = th.zeros(num).bool()
            train_mask[0: int(split_ratio[0] * num)] = True
            val_mask = th.zeros(num).bool()
            val_mask[int(split_ratio[0] * num): int((split_ratio[0] + split_ratio[1]) * num)] = True
            test_mask = th.zeros(num).bool()
            test_mask[int((split_ratio[0] + split_ratio[1]) * num):] = True
            hg.edges[etype].data['train_mask'] = train_mask
            hg.edges[etype].data['val_mask'] = val_mask
            hg.edges[etype].data['test_mask'] = test_mask
        # Furthermore, you can also optionally sample the negative edges and process them as 
        # properties neg_val_edges and neg_test_edges. We will first check whether the dataset 
        # has properties named neg_val_edges and neg_test_edges. If no, we will sample negative
        # val/test edges according to neg_ratio and neg_sampler.
        self._neg_val_edges, self._neg_test_edges = self._sample_negative_edges(hg)
        self._g = hg

    def _sample_negative_edges(self, hg):
        negative_sampler = GlobalUniform(1)
        val_edges = {
            etype: th.nonzero(hg.edges[etype].data['val_mask']).squeeze()
            for etype in target_link}
        neg_val_edges = negative_sampler(hg, val_edges)
        test_edges = {
            etype: th.nonzero(hg.edges[etype].data['test_mask']).squeeze()
            for etype in target_link}
        neg_test_edges = negative_sampler(hg, test_edges)
        return neg_val_edges, neg_test_edges

    @property
    def neg_val_edges(self):
        return self._neg_val_edges

    @property
    def neg_test_edges(self):
        return self._neg_test_edges

    @property
    def meta_paths_dict(self):
        return meta_paths_dict

    def __getitem__(self, idx):
        return self._g

    def __len__(self):
        return 1


def generate_random_citation_hg() -> DGLHeteroGraph:
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
    return hg


def train_with_custom_lp_dataset(dataset):
    from openhgnn.config import Config
    from openhgnn.start import OpenHGNN
    config_file = ["../../openhgnn/config.ini"]
    config = Config(file_path=config_file, model='RGCN', dataset=dataset, task='link_prediction', gpu=-1)
    OpenHGNN(args=config)


if __name__ == '__main__':
    myLPDataset = AsLinkPredictionDataset(MyLPDataset(), target_link=target_link, target_link_r=target_link_r,
                                          split_ratio=[0.8, 0.1, 0.1], force_reload=True)
    train_with_custom_lp_dataset(myLPDataset)

    mySplitLPDatasetWithNegEdges = AsLinkPredictionDataset(MySplitLPDatasetWithNegEdges(), target_link=target_link,
                                                           target_link_r=target_link_r,
                                                           force_reload=True)
    train_with_custom_lp_dataset(mySplitLPDatasetWithNegEdges)
