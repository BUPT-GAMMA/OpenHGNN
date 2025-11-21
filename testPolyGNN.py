from openhgnn import Experiment
"""available datasets:'MNIST-P-2','Building-2-C','Building-2-R','Building-S'"""
experiment = Experiment(model='PolyGNN', dataset='MNIST-P-2', task='PolyGNN', gpu=0,
                        max_epoch=15,use_distributed=False,graphbolt=False)
experiment.run()