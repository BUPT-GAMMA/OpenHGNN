import torch
import argparse

from openhgnn import Experiment
from openhgnn.dataset import AsNodeClassificationDataset, ACM4GTNDataset

from sklearn.metrics import f1_score, accuracy_score

# python node_classification_inference.py -m RGCN -d acm -g -1
# python node_classification_inference.py -m RGCN -d acm -g -1 --mini-batch-flag


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
        ds[0].nodes[ds.target_ntype].data.pop('train_mask')
        ds[0].nodes[ds.target_ntype].data.pop('val_mask')

        new_ds = AsNodeClassificationDataset(ds, name='acm-pred1', target_ntype=ds.target_ntype)
    else:
        raise ValueError

    labels = new_ds[0].nodes[new_ds.target_ntype].data['label']
    experiment = Experiment(conf_path='./my_config.ini', max_epoch=0, model=args.model, dataset=new_ds,
                            task='node_classification', mini_batch_flag=args.mini_batch_flag, gpu=args.gpu,
                            test_flag=False, prediction_flag=True, batch_size=100, load_from_pretrained=True)
    prediction_res = experiment.run()
    indices, y_predicts = prediction_res
    y_predicts = torch.argmax(y_predicts, dim=1)
    y_true = labels[indices]

    print('indices shape', indices.shape)
    print('y_predicts shape', y_predicts.shape)

    print('acc', accuracy_score(y_true, y_predicts))
    print('f1 score macro', f1_score(y_true, y_predicts, average='macro'))
    print('f1 score micro', f1_score(y_true, y_predicts, average='micro'))
