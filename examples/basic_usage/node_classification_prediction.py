import dgl
import torch

from openhgnn import Experiment
from openhgnn.dataset import AsNodeClassificationDataset, ACM4GTNDataset

from dgl.data import CoraFullDataset

if __name__ == '__main__':
    # hetero
    # ds = ACM4GTNDataset()
    # ds[0].nodes[ds.target_ntype].data['pred_mask'] = ds[0].nodes[ds.target_ntype].data['test_mask']
    # new_ds = AsNodeClassificationDataset(ds, target_ntype=ds.target_ntype)

    # hetero with one etype type
    ds = CoraFullDataset()
    n = ds[0].num_nodes()
    g = dgl.heterograph(data_dict={('author', 'coauthor', 'author'): ds[0].edges()})
    g.ndata['label_mask'] = torch.zeros(n).bool()
    g.ndata['label_mask'][0: int(0.8 * n)] = True
    g.nodes['author'].data['label'] = ds[0].ndata['label']
    g.nodes['author'].data['feat'] = ds[0].ndata['feat']

    new_ds = AsNodeClassificationDataset(g, name='cora-hetero', target_ntype='author', split_ratio=[0.8, 0.2, 0],
                                         label_mask_feat_name='label_mask')

    experiment = Experiment(model='RGCN', dataset=new_ds, task='node_classification', mini_batch_flag=True, gpu=-1,
                            lr=0.05, hidden_dim=64, max_epoch=1, n_layers=3, test_flag=False, prediction_flag=True)
    prediction_res = experiment.run()

    indices, y_predicts = prediction_res
    y_predicts = torch.argmax(y_predicts, dim=1)
    print(indices.shape)
    print(y_predicts.shape)
