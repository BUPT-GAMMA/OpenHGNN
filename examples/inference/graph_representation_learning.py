import argparse
from openhgnn import Experiment
from openhgnn.dataset import generate_random_hg
from dgl import transforms as T
from dgl import DGLHeteroGraph
from dgl.data import DGLDataset

category = 'author'
meta_paths_dict = {'APA': [('author', 'author-paper', 'paper'), ('paper', 'rev_author-paper', 'author')]}


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
    parser.add_argument('--model', '-m', default='Metapath2vec', type=str, help='name of models')
    parser.add_argument('--gpu', '-g', default='-1', type=int, help='-1 means cpu')
    parser.add_argument('--mini-batch-flag', action='store_true')

    args = parser.parse_args()

    ds = MyDataset()

    experiment = Experiment(conf_path='./my_config.ini', max_epoch=1, model=args.model, dataset=ds,
                            task='demo', meta_path_key='APA', mini_batch_flag=args.mini_batch_flag, gpu=args.gpu,
                            test_flag=False, prediction_flag=False, batch_size=100)
    experiment.run()
