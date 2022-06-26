from openhgnn import Experiment

if __name__ == '__main__':
    experiment = Experiment(model='RGCN', dataset='acm4GTN', task='node_classification', gpu=-1, lr=0.05, hidden_dim=64,
                            max_epoch=30, n_layers=3)
    experiment.run()
