import torch
from sklearn.metrics import f1_score, accuracy_score

import argparse
from openhgnn import Experiment
from openhgnn.dataset import AsNodeClassificationDataset, generate_random_hg
from dgl import transforms as T
from dgl import DGLHeteroGraph
from dgl.data import DGLDataset
from openhgnn.dataset import ACM4GTNDataset
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default='RGCN', type=str, help='name of models')
    parser.add_argument('--dataset', '-d', default='acm', type=str, help='acm or cora')
    parser.add_argument('--gpu', '-g', default='-1', type=int, help='-1 means cpu')
    parser.add_argument('--mini-batch-flag', action='store_true')

    args = parser.parse_args()

    # acm
    # if args.dataset == 'acm':
    #     ds = ACM4GTNDataset()
    #     ds[0].nodes[ds.target_ntype].data['pred_mask'] = ds[0].nodes[ds.target_ntype].data.pop('test_mask')
    #     ds[0].nodes[ds.target_ntype].data.pop('train_mask')
    #     ds[0].nodes[ds.target_ntype].data.pop('val_mask')
    #
    #     new_ds = AsNodeClassificationDataset(ds, name='acm-pred1', target_ntype=ds.target_ntype)
    # else:
    #     raise ValueError

    # labels = new_ds[0].nodes[new_ds.target_ntype].data['label']

    ds = MyNCDataset()
    new_ds = AsNodeClassificationDataset(ds, target_ntype='author', labeled_nodes_split_ratio=[0.8, 0.1, 0.1],
                                         prediction_ratio=1)

    experiment = Experiment(conf_path='./my_config.ini', max_epoch=0, model=args.model, dataset=new_ds,
                            task='node_classification', mini_batch_flag=args.mini_batch_flag, gpu=args.gpu,
                            test_flag=False, prediction_flag=True, batch_size=100, load_from_pretrained=True)
    prediction_res = experiment.run()
    indices, y_predicts = prediction_res
    # y_predicts = torch.argmax(y_predicts, dim=1)
    # y_true = labels[indices]

    print('indices shape', indices.shape)
    print('y_predicts shape', y_predicts.shape)

    # print('acc', accuracy_score(y_true, y_predicts))
    # print('f1 score macro', f1_score(y_true, y_predicts, average='macro'))
    # print('f1 score micro', f1_score(y_true, y_predicts, average='micro'))
