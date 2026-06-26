import os
import torch
from abc import ABC, abstractmethod

from ..tasks import build_task
from ..layers.HeteroLinear import HeteroFeature
from ..utils import get_nodes_dict


class BaseFlow(ABC):
    candidate_optimizer = {
        'Adam': torch.optim.Adam,
        'SGD': torch.optim.SGD,
        'Adadelta': torch.optim.Adadelta
    }

    def __init__(self, args):
        """

        Parameters
        ----------
        args

        Attributes
        -------------
        evaluate_interval: int
            the interval of evaluation in validation
        """
        super(BaseFlow, self).__init__()
        self.evaluator = None
        self.evaluate_interval = getattr(args, 'evaluate_interval', 1)
        if hasattr(args, 'model_path'):
            self._checkpoint = args.model_path
        elif hasattr(args, '_checkpoint'):
            self._checkpoint = os.path.join(args._checkpoint, f"{args.model_name}_{args.dataset_name}.pt")
        else:
            if hasattr(args, 'load_from_pretrained'):
                self._checkpoint = os.path.join(args.output_dir,
                                                f"{args.model_name}_{args.dataset_name}_{args.task}.pt")
            else:
                self._checkpoint = None

        if not hasattr(args, 'HGB_results_path') and args.dataset_name[:3] == 'HGB':
            args.HGB_results_path = os.path.join(args.output_dir,
                                                 "{}_{}_{}.txt".format(args.model_name, args.dataset_name[5:],
                                                                       args.seed))

        # Distributed models will check this parameter during the training process to determine whether to use distributed.
        self.use_distributed = args.use_distributed

        # stage flags: whether to run the corresponding stages
        # todo: only take effects in node classification trainer flow

        # args.training_flag = getattr(args, 'training_flag', True)
        # args.validation_flag = getattr(args, 'validation_flag', True)
        args.test_flag = getattr(args, 'test_flag', True)
        args.prediction_flag = getattr(args, 'prediction_flag', False)
        args.use_uva = getattr(args, 'use_uva', False)

        self.args = args
        self.logger = self.args.logger
        self.model_name = args.model_name
        self.model = args.model
        self.device = args.device
        self.task = build_task(args)
        self.max_epoch = args.max_epoch
        self.optimizer = None

        if self.model_name in ["SIAN", "MeiREC", "ExpressGNN", "Ingram", "RedGNN","RedGNNT",  "AdapropI", "AdapropT",
                               "RedGNNT", "Grail", "ComPILE","DisenKGAT","MetaHIN"]:
            return
        if self.model_name == "Ingram":
            return

        if self.args.use_uva:
            self.hg = self.task.get_graph()
        else:
            self.hg = self.task.get_graph().to(self.device)
        self.args.meta_paths_dict = self.task.dataset.meta_paths_dict
        self.patience = args.patience

        self.loss_fn = self.task.get_loss_fn()

    def preprocess(self):
        r"""
        Every trainerflow should run the preprocess_feature if you want to get a feature preprocessing.
        The Parameters in input_feature will be added into optimizer and input_feature will be added into the model.

        Attributes
        -----------
        input_feature : HeteroFeature
            It will return the processed feature if call it.

        """
        if hasattr(self.args, 'activation'):
            if hasattr(self.args.activation, 'weight'):
                import torch.nn as nn
                act = nn.PReLU()
            else:
                act = self.args.activation
        else:
            act = None
        # useful type selection
        if hasattr(self.args, 'feat'):
            pass
        else:
            # Default 0, nothing to do.
            self.args.feat = 0
        self.feature_preprocess(act)
        self.optimizer.add_param_group({'params': self.input_feature.parameters()})
        # for early stop, load the model with input_feature module.
        self.model.add_module('input_feature', self.input_feature)
        self.load_from_pretrained()

    def feature_preprocess(self, act):
        """
        Feat
            0, 1 ,2
        Node feature
            1 node type & more than 1 node types
            no feature

        Returns
        -------

        """

        if self.hg.ndata.get('h', {}) == {} or self.args.feat == 2:
            if self.hg.ndata.get('h', {}) == {}:
                self.logger.feature_info('Assign embedding as features, because hg.ndata is empty.')
            else:
                self.logger.feature_info('feat2, drop features!')
                self.hg.ndata.pop('h')
            self.input_feature = HeteroFeature({}, get_nodes_dict(self.hg), self.args.hidden_dim,
                                            act=act).to(self.device)
        elif self.args.feat == 0:
            self.input_feature = self.init_feature(act)
        elif self.args.feat == 1:
            if self.args.task != 'node_classification':
                self.logger.feature_info('\'feat 1\' is only for node classification task, set feat 0!')
                self.input_feature = self.init_feature(act)
            else:
                h_dict = self.hg.ndata.pop('h')
                self.logger.feature_info('feat1, preserve target nodes!')
                self.input_feature = HeteroFeature({self.category: h_dict[self.category]}, get_nodes_dict(self.hg), self.args.hidden_dim,
                                                act=act).to(self.device)

    def init_feature(self, act):
        self.logger.feature_info("Feat is 0, nothing to do!")
        if isinstance(self.hg.ndata['h'], dict):
            # The heterogeneous contains more than one node type.
            input_feature = HeteroFeature(self.hg.ndata['h'], get_nodes_dict(self.hg),
                                            self.args.hidden_dim, act=act).to(self.device)
        elif isinstance(self.hg.ndata['h'], torch.Tensor):
            # The heterogeneous only contains one node type.
            input_feature = HeteroFeature({self.hg.ntypes[0]: self.hg.ndata['h']}, get_nodes_dict(self.hg),
                                            self.args.hidden_dim, act=act).to(self.device)
        return input_feature

    @abstractmethod
    def train(self):
        pass

    def _full_train_step(self):
        r"""
        Train with a full_batch graph
        """
        raise NotImplementedError

    def _mini_train_step(self):
        r"""
        Train with a mini_batch seed nodes graph
        """
        raise NotImplementedError

    def _full_test_step(self):
        r"""
        Test with a full_batch graph
        """
        raise NotImplementedError

    def _mini_test_step(self):
        r"""
        Test with a mini_batch seed nodes graph
        """
        raise NotImplementedError

    def load_from_pretrained(self):
        if hasattr(self.args, 'load_from_pretrained') and self.args.load_from_pretrained:
            try:
                ck_pt = torch.load(self._checkpoint)
                self.model.load_state_dict(ck_pt)
                self.logger.info('[Load Model] Load model from pretrained model:' + self._checkpoint)
            except FileNotFoundError:
                self.logger.info('[Load Model] Do not load the model from pretrained, '
                                      '{} doesn\'t exists'.format(self._checkpoint))
        # return self.model

    def save_checkpoint(self):
        if self._checkpoint and hasattr(self.model, "_parameters()"):
            torch.save(self.model.state_dict(), self._checkpoint)
