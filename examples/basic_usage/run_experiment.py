from openhgnn import Experiment
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

if __name__ == '__main__':
    experiment = Experiment(model='RHINE', dataset='dblp4RHINE', task='node_classification', gpu=0, lr=0.05, hidden_dim=64,
                            max_epoch=30, num_layers=3,use_distributed=False,batch_size=128,graphbolt=False)
    experiment.run()
