import torch

from openhgnn import Experiment

if __name__ == '__main__':

    experiment = Experiment(model='RGCN', dataset='acm4GTN', task='node_classification', mini_batch_flag=False, gpu=-1,
                            lr=0.05, hidden_dim=64, max_epoch=1, n_layers=3, test_flag=False, prediction_flag=True)
    prediction_res = experiment.run()
    indices, y_predicts = prediction_res
    y_predicts = torch.argmax(y_predicts, dim=1)

    print(indices.shape)
    print(y_predicts.shape)
