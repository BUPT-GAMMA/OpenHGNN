import torch

from openhgnn import Experiment
from openhgnn.dataset import AsNodeClassificationDataset, ACM4GTNDataset

if __name__ == '__main__':
    ds = ACM4GTNDataset()
    ds[0].nodes[ds.target_ntype].data['pred_mask'] = ds[0].nodes[ds.target_ntype].data['test_mask']
    new_ds = AsNodeClassificationDataset(ds, target_ntype=ds.target_ntype)

    experiment = Experiment(model='RGCN', dataset=new_ds, task='node_classification', mini_batch_flag=True, gpu=-1,
                            lr=0.05, hidden_dim=64, max_epoch=1, n_layers=3, test_flag=False, prediction_flag=True)
    prediction_res = experiment.run()

    indices, y_predicts = prediction_res
    y_predicts = torch.argmax(y_predicts, dim=1)
    print(indices.shape)
    print(y_predicts.shape)
