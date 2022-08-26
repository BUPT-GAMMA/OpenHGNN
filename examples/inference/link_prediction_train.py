import argparse
from openhgnn import Experiment
from openhgnn.dataset import AsLinkPredictionDataset, generate_random_hg
from dgl import transforms as T
from dgl import DGLHeteroGraph
from dgl.data import DGLDataset

meta_paths_dict = {'APA': [('author', 'author-paper', 'paper'), ('paper', 'rev_author-paper', 'author')]}
target_link = [('author', 'author-paper', 'paper')]
target_link_r = [('paper', 'rev_author-paper', 'author')]


class MyLPDataset(DGLDataset):
    def __init__(self):
        super().__init__(name='my-lp-dataset',
                         force_reload=True)

    def process(self):
        hg = generate_random_citation_hg()
        self._g = hg

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default='RGCN', type=str, help='name of models')
    parser.add_argument('--gpu', '-g', default='-1', type=int, help='-1 means cpu')
    parser.add_argument('--mini-batch-flag', action='store_true')

    args = parser.parse_args()

    ds = MyLPDataset()
    new_ds = AsLinkPredictionDataset(ds, target_link=target_link, target_link_r=target_link_r,
                                     split_ratio=[0.8, 0.1, 0.1], force_reload=True)

    experiment = Experiment(conf_path='./my_config.ini', max_epoch=20, model=args.model, dataset=new_ds,
                            task='link_prediction', mini_batch_flag=args.mini_batch_flag, gpu=args.gpu,
                            test_flag=False, prediction_flag=False, batch_size=100)
    experiment.run()
