import torch.nn.functional as F
import torch.nn as nn
from . import BaseTask, register_task
from ..dataset import build_dataset,build_dataset_GB
from ..utils import Evaluator
import torch
import numpy as np


@register_task("node_classification")
class NodeClassification(BaseTask):
    r"""
    Node classification tasks.

    Attributes
    -----------
    dataset : NodeClassificationDataset
        Task-related dataset

    evaluator : Evaluator
        offer evaluation metric


    Methods
    ---------
    get_graph :
        return a graph
    get_loss_fn :
        return a loss function
    """
    def __init__(self, args):
        super(NodeClassification, self).__init__()
        self.logger = args.logger
        self.dataset = build_dataset(args.dataset, 'node_classification', 
                                     logger=self.logger,args = args)
        if args.graphbolt:
            #   这个就是task.dataset_GB
            self.dataset_GB = build_dataset_GB(args.dataset, 
                                        logger=self.logger, 
                                        args = args)  
        # self.evaluator = Evaluator()
        self.logger = args.logger
        if hasattr(args, 'validation'):
            self.train_idx, self.val_idx, self.test_idx = self.dataset.get_split(args.validation)
        else:
            self.train_idx, self.val_idx, self.test_idx = self.dataset.get_split()
        self.evaluator = Evaluator(args.seed)
        self.labels = self.dataset.get_labels()
        self.multi_label = self.dataset.multi_label
        
        if hasattr(args, 'evaluation_metric'):
            self.evaluation_metric = args.evaluation_metric
        else:
            if args.dataset in ['aifb', 'mutag', 'bgs', 'am']:
                self.evaluation_metric = 'acc'
            else:
                self.evaluation_metric = 'f1'

    def get_graph(self):
        return self.dataset.g

    def get_loss_fn(self):
        if self.multi_label:
            return nn.BCEWithLogitsLoss()
        return F.cross_entropy

    def get_evaluator(self, name):
        if name == 'acc':
            return self.evaluator.cal_acc
        elif name == 'f1_lr':
            return self.evaluator.nc_with_LR
        elif name == 'f1':
            return self.evaluator.f1_node_classification

    def evaluate(self, logits, mode='test', info=True):
        if mode == 'test':
            mask = self.test_idx
        elif mode == 'valid':
            mask = self.val_idx
        elif mode == 'train':
            mask = self.train_idx

        if self.multi_label:
            pred = (logits[mask].cpu().numpy() > 0).astype(int)
        else:
            pred = logits[mask].argmax(dim=1).to('cpu')
            
        if self.evaluation_metric == 'acc':
            acc = self.evaluator.cal_acc(self.labels[mask], pred)
            return dict(Accuracy=acc)
        elif self.evaluation_metric == 'acc-ogbn-mag':
            from ogb.nodeproppred import Evaluator
            evaluator = Evaluator(name='ogbn-mag')
            logits = logits.unsqueeze(dim=1)
            input_dict = {"y_true": logits, "y_pred": self.labels[self.test_idx]}
            result_dict = evaluator.eval(input_dict)
            return result_dict
        elif self.evaluation_metric == 'f1':
            f1_dict = self.evaluator.f1_node_classification(self.labels[mask], pred)
            return f1_dict
        else:
            raise ValueError('The evaluation metric is not supported!')

    def downstream_evaluate(self, logits, evaluation_metric):
        if evaluation_metric == 'f1_lr':
            micro_f1, macro_f1 = self.evaluator.nc_with_LR(logits, self.labels, self.train_idx, self.test_idx)
            return dict(Macro_f1=macro_f1, Mirco_f1=micro_f1)
    
    def get_split(self):
        return self.train_idx, self.val_idx, self.test_idx

    def get_labels(self):
        return self.labels


@register_task("DSSL_trainer")
class DSSL_task(BaseTask):
    r"""
    DSSL_task .

    Attributes
    -----------
    dataset : NodeClassificationDataset
        Task-related dataset

    evaluator : Evaluator
        offer evaluation metric


    Methods
    ---------
    get_graph :
        return a graph
    get_loss_fn :
        return a loss function
    """

    def __init__(self, args):
        super(DSSL_task, self).__init__()
        self.logger = args.logger
        self.dataset = build_dataset(args.dataset, 'node_classification', logger=self.logger)
        # self.evaluator = Evaluator()
        self.logger = args.logger
        if hasattr(args, 'validation'):
            self.train_idx, self.val_idx, self.test_idx = self.dataset.get_split(args.validation)
        else:
            self.train_idx, self.val_idx, self.test_idx = self.dataset.get_split()
        self.evaluator = Evaluator(args.seed)
        self.labels = self.dataset.get_labels()
        self.multi_label = self.dataset.multi_label
        self.train_prop = args.train_prop
        self.valid_prop = args.valid_prop

        if hasattr(args, 'evaluation_metric'):
            self.evaluation_metric = args.evaluation_metric
        else:
            if args.dataset in ['aifb', 'mutag', 'bgs', 'am']:
                self.evaluation_metric = 'acc'
            else:
                self.evaluation_metric = 'f1'

    def get_graph(self):
        return self.dataset.g

    def get_loss_fn(self):
        if self.multi_label:
            return nn.BCEWithLogitsLoss()
        return F.cross_entropy

    def get_evaluator(self, name):
        if name == 'acc':
            return self.evaluator.cal_acc
        elif name == 'f1_lr':
            return self.evaluator.nc_with_LR
        elif name == 'f1':
            return self.evaluator.f1_node_classification

    def evaluate(self, logits, mode='test', info=True):
        if mode == 'test':
            mask = self.test_idx
        elif mode == 'valid':
            mask = self.val_idx
        elif mode == 'train':
            mask = self.train_idx

        if self.multi_label:
            pred = (logits[mask].cpu().numpy() > 0).astype(int)
        else:
            pred = logits[mask].argmax(dim=1).to('cpu')

        if self.evaluation_metric == 'acc':
            acc = self.evaluator.cal_acc(self.labels[mask], pred)
            return dict(Accuracy=acc)
        elif self.evaluation_metric == 'acc-ogbn-mag':
            from ogb.nodeproppred import Evaluator
            evaluator = Evaluator(name='ogbn-mag')
            logits = logits.unsqueeze(dim=1)
            input_dict = {"y_true": logits, "y_pred": self.labels[self.test_idx]}
            result_dict = evaluator.eval(input_dict)
            return result_dict
        elif self.evaluation_metric == 'f1':
            f1_dict = self.evaluator.f1_node_classification(self.labels[mask], pred)
            return f1_dict
        else:
            raise ValueError('The evaluation metric is not supported!')

    def downstream_evaluate(self, logits, evaluation_metric):
        if evaluation_metric == 'f1_lr':
            micro_f1, macro_f1 = self.evaluator.nc_with_LR(logits, self.labels, self.train_idx, self.test_idx)
            return dict(Macro_f1=macro_f1, Mirco_f1=micro_f1)

    def get_rand_split(self):
        self.train_idx, self.val_idx, self.test_idx = self.get_idx_split(train_prop=self.train_prop, valid_prop=self.valid_prop)
        return self.train_idx, self.val_idx, self.test_idx

    def get_split(self):
        return self.train_idx, self.val_idx, self.test_idx

    def get_labels(self):
        return self.labels

    def get_idx_split(self, split_type='random', train_prop=.5, valid_prop=.25):
        """
        train_prop: The proportion of dataset for train split. Between 0 and 1.
        valid_prop: The proportion of dataset for validation split. Between 0 and 1.
        """
        if split_type == 'random':
            ignore_negative = True
            train_idx, valid_idx, test_idx = self.rand_train_test_idx(
                self.labels, train_prop=train_prop, valid_prop=valid_prop, ignore_negative=ignore_negative)

        return train_idx, valid_idx, test_idx

    def rand_train_test_idx(self,label, train_prop=.5, valid_prop=.25, ignore_negative=True):
        """ randomly splits label into train/valid/test splits """
        if ignore_negative:
            labeled_nodes = torch.where(label != -1)[0]
        else:
            labeled_nodes = label

        n = labeled_nodes.shape[0]
        train_num = int(n * train_prop)
        valid_num = int(n * valid_prop)

        perm = torch.as_tensor(np.random.permutation(n))

        train_indices = perm[:train_num]
        val_indices = perm[train_num:train_num + valid_num]
        test_indices = perm[train_num + valid_num:]

        if not ignore_negative:
            return train_indices, val_indices, test_indices
        # print (labeled_nodes)
        # print (train_indices)

        train_idx = labeled_nodes[train_indices.type(torch.LongTensor)]
        valid_idx = labeled_nodes[val_indices.type(torch.LongTensor)]
        test_idx = labeled_nodes[test_indices.type(torch.LongTensor)]

        return train_idx, valid_idx, test_idx
