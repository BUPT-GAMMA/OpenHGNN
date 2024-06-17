from openhgnn import Experiment

if __name__ == '__main__':
    experiment = Experiment(model='HGA', dataset='ACM,DBLP', task='node_classification', gpu=0, lr=0.05, hidden_dim=64,
                             max_epoch=10, num_layers=3,num_heads=8,seed=1919810,out_dim=4,gamma=1,use_distributed=False)
    experiment.run()
