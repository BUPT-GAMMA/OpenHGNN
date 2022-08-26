import argparse
import torch as th
from openhgnn import Experiment
from openhgnn.dataset import AsLinkPredictionDataset, generate_random_hg
from dgl import transforms as T
from dgl import DGLHeteroGraph
from dgl.data import DGLDataset

meta_paths_dict = {'APA': [('author', 'author-paper', 'paper'), ('paper', 'rev_author-paper', 'author')]}
target_link = [('author', 'author-paper', 'paper')]
target_link_r = [('paper', 'rev_author-paper', 'author')]


class MyLPDatasetWithPredEdges(DGLDataset):
    def __init__(self):
        super().__init__(name='my-lp-dataset-with-pred-edges',
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
        'author': 1000,
    }
    hg = generate_random_hg(num_nodes_dict=num_nodes_dict, num_edges_dict=num_edges_dict)
    transform = T.Compose([T.ToSimple(), T.AddReverse()])
    hg = transform(hg)
    return hg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default='RGCN', type=str, help='name of models')
    parser.add_argument('--gpu', '-g', default='-1', type=int, help='-1 means cpu')

    args = parser.parse_args()

    ds = MyLPDatasetWithPredEdges()

    new_ds = AsLinkPredictionDataset(ds, target_link=target_link, target_link_r=target_link_r,
                                     split_ratio=[0.8, 0.1, 0.1], force_reload=True)

    experiment = Experiment(conf_path='./my_config.ini', max_epoch=0, model=args.model, dataset=new_ds,
                            task='link_prediction', gpu=args.gpu, test_flag=False, prediction_flag=True,
                            batch_size=1000, load_from_pretrained=True)

    prediction_res = experiment.run()
    print(prediction_res)
