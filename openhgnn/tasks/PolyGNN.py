import torch.nn as nn
from . import BaseTask, register_task
from ..dataset import build_dataset,build_dataset_GB
import torch


@register_task("PolyGNN")
class PolyGNN(BaseTask):
    r"""
    The task for PolyGNN.
    Note : The method of get_graph makes no use in the task,it is implemented due to the request of the framework.
    Attributes
    -----------
    dataset : PolyGNNDataset
        Task-related dataset

    Methods
    ---------
    get_split :
        get the split dataloaders by using the method from PolyGNNDataset.
    get_loss_fn :
        return a loss function
    """
    def __init__(self, args):
        super(PolyGNN, self).__init__()
        self.logger = args.logger
        self.dataset = build_dataset(args.dataset, 'PolyGNNDataset', 
                                      logger=self.logger,args = args)
        self.num_classes = self.dataset.num_classes

    def get_graph(self): # 需要返回一个hg图
        import dgl
        graph_data = {
        ('drug', 'interacts', 'drug'): (torch.tensor([0, 1]), torch.tensor([1, 2])),
        ('drug', 'interacts', 'gene'): (torch.tensor([0, 1]), torch.tensor([2, 3])),
        ('drug', 'treats', 'disease'): (torch.tensor([1]), torch.tensor([2]))
        }
        g = dgl.heterograph(graph_data)
        return g 

    def get_loss_fn(self):
        return nn.CrossEntropyLoss()


    def evaluate(self, logits, mode='test', info=True):
        pass
    
    def get_split(self):
        return self.dataset.get_split()

