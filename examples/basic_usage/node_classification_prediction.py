import dgl
import torch
import argparse

from openhgnn import Experiment
from openhgnn.dataset import AsNodeClassificationDataset, ACM4GTNDataset

from dgl.data import CoraFullDataset, CoraGraphDataset
from sklearn.metrics import f1_score, accuracy_score

# python node_classification_prediction.py -m RGCN -d acm -g -1
# python node_classification_prediction.py -m RGCN -d acm -g -1 --mini-batch-flag
# python node_classification_prediction.py -m RGCN -d cora -g -1
# python node_classification_prediction.py -m RGCN -d cora -g -1 --mini-batch-flag


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default='RGCN', type=str, help='name of models')
    parser.add_argument('--dataset', '-d', default='acm', type=str, help='acm or cora')
    parser.add_argument('--gpu', '-g', default='-1', type=int, help='-1 means cpu')
    parser.add_argument('--mini-batch-flag', action='store_true')

    args = parser.parse_args()
    # acm
    if args.dataset == 'acm':
        ds = ACM4GTNDataset()
        ds[0].nodes[ds.target_ntype].data['pred_mask'] = ds[0].nodes[ds.target_ntype].data.pop('test_mask')
        new_ds = AsNodeClassificationDataset(ds, target_ntype=ds.target_ntype)

    # cora graph
    elif args.dataset == 'cora':
        ds = CoraGraphDataset()
        n = ds[0].num_nodes()
        g = dgl.heterograph(data_dict={('paper', 'cite', 'paper'): ds[0].edges()})
        g.ndata['train_mask'] = ds[0].ndata['train_mask']
        g.ndata['val_mask'] = ds[0].ndata['val_mask']
        g.ndata['test_mask'] = ds[0].ndata['test_mask']
        g.ndata['pred_mask'] = ds[0].ndata['test_mask']
        g.ndata['label'] = ds[0].ndata['label']
        g.ndata['feat'] = ds[0].ndata['feat']
        new_ds = AsNodeClassificationDataset(g, name='cora-hetero', target_ntype='paper', force_reload=True)
        new_ds.num_classes = ds.num_classes

    # cora full

    # ds = CoraFullDataset()
    # n = ds[0].num_nodes()
    # g = dgl.heterograph(data_dict={('author', 'coauthor', 'author'): ds[0].edges()})
    # g.ndata['label_mask'] = torch.zeros(n).bool()
    # g.ndata['label_mask'][0: int(0.8 * n)] = True
    # g.ndata['label'] = ds[0].ndata['label']
    # g.ndata['feat'] = ds[0].ndata['feat']
    # new_ds = AsNodeClassificationDataset(g, name='cora-hetero', target_ntype='author', split_ratio=[0.8, 0.2, 0],
    #                                      label_mask_feat_name='label_mask', force_reload=True)
    # new_ds.num_classes = ds.num_classes

    labels = new_ds[0].nodes[new_ds.target_ntype].data['label']
    experiment = Experiment(conf_path='./my_config.ini', max_epoch=1, epoch=1, model=args.model, dataset=new_ds,
                            task='node_classification', mini_batch_flag=args.mini_batch_flag, gpu=args.gpu,
                            test_flag=False, prediction_flag=True, batch_size=100)
    prediction_res = experiment.run()
    indices, y_predicts = prediction_res
    y_predicts = torch.argmax(y_predicts, dim=1)
    y_true = labels[indices]

    print('indices shape', indices.shape)
    print('y_predicts shape', y_predicts.shape)

    print('acc', accuracy_score(y_true, y_predicts))
    print('f1 score macro', f1_score(y_true, y_predicts, average='macro'))
    print('f1 score micro', f1_score(y_true, y_predicts, average='micro'))
