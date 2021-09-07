import os
import torch
from abc import ABC, abstractmethod

from ..tasks import build_task
from ..layers.HeteroLinear import HeteroFeature
from ..utils import get_nodes_dict


class BaseFlow(ABC):

    def __init__(self, args):
        super(BaseFlow, self).__init__()
        self.evaluator = None
        self.evaluate_interval = 1
        self.load_from_checkpoint = True
        if hasattr(args, '_checkpoint'):
            self._checkpoint = os.path.join(args._checkpoint,
                                            f"{args.model}_{args.dataset}.pt")
        else:
            if self.load_from_checkpoint:
                self._checkpoint = os.path.join("./openhgnn/output/{}".format(args.model),
                                                f"{args.model}_{args.dataset}.pt")
            else:
                self._checkpoint = None
        # if args.dataset[:3] == 'HGB':
        #     args.HGB_results_path = os.path.join("./openhgnn/output/{}/{}.txt".format(args.model, args.dataset[5:]))

        self.args = args
        self.model_name = args.model
        self.device = args.device
        self.task = build_task(args)
        self.hg = self.task.get_graph().to(self.device)
        self.args.meta_paths = self.task.dataset.meta_paths
        self.patience = args.patience
        self.max_epoch = args.max_epoch
        self.optimizer = None
        self.loss_fn = self.task.get_loss_fn()

    def preprocess_feature(self):
        if isinstance(self.hg.ndata['h'], dict):
            self.input_feature = HeteroFeature(self.hg.ndata['h'], get_nodes_dict(self.hg), self.args.hidden_dim).to(self.device)
        elif isinstance(self.hg.ndata['h'], torch.Tensor):
            self.input_feature = HeteroFeature({self.hg.ntypes[0]: self.hg.ndata['h']}, get_nodes_dict(self.hg), self.args.hidden_dim).to(self.device)
        self.optimizer.add_param_group({'params': self.input_feature.parameters()})

    @abstractmethod
    def train(self):
        pass

    def _full_train_step(self):
        # train with a full_batch graph
        raise NotImplementedError

    def _mini_train_step(self):
        # train with a mini_batch seed nodes graph
        raise NotImplementedError

    def _test_step(self):
        raise NotImplementedError

    def load_from_pretrained(self):
        if self.load_from_checkpoint:
            try:
                ck_pt = torch.load(self._checkpoint)
                self.model.load_state_dict(ck_pt)
            except FileNotFoundError:
                print(f"'{self._checkpoint}' doesn't exists")
        return self.model

    def save_checkpoint(self):
        if self._checkpoint and hasattr(self.model, "_parameters()"):
            torch.save(self.model.state_dict(), self._checkpoint)
