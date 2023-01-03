import argparse

import dgl
import torch

from openhgnn import Experiment
from openhgnn.dataset import AsLinkPredictionDataset
from dataset import MyDataset, target_link

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default='TransE', type=str, help='name of models')
    parser.add_argument('--gpu', '-g', default='-1', type=int, help='-1 means cpu')
    parser.add_argument('--mini-batch-flag', action='store_true')

    args = parser.parse_args()

    ds = MyDataset()
    new_ds = AsLinkPredictionDataset(ds, target_link=target_link, target_link_r=None,
                                     split_ratio=[0.8, 0.1, 0.1], force_reload=True)


    def get_triples(hg):
        g = dgl.to_homogeneous(hg)
        src, dst = g.edges()
        rel = g.edata[dgl.ETYPE]
        return torch.stack((src, rel, dst)).T


    new_ds.pred_triples = get_triples(new_ds.pred_graph)

    experiment = Experiment(conf_path='./my_config.ini', max_epoch=0, model=args.model, dataset=new_ds,
                            task='link_prediction', mini_batch_flag=args.mini_batch_flag,
                            gpu=args.gpu, test_flag=False, prediction_flag=True,
                            # gpu=args.gpu, test_flag=True, prediction_flag=False,

                            batch_size=100, evaluation_metric='mrr', load_from_pretrained=True)
    prediction_res = experiment.run()
    print(prediction_res)
