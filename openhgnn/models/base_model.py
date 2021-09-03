from typing import Optional, Type, Any
from abc import ABCMeta, abstractmethod
import torch.nn as nn


class BaseModel(nn.Module, metaclass=ABCMeta):
    @classmethod
    def build_model_from_args(cls, args, hg):
        r"""
        Build the model instance from args and hg.

        So every subclass inheriting it should override the method.
        """
        raise NotImplementedError("Models must implement the build_model_from_args method")

    def __init__(self):
        super(BaseModel, self).__init__()
        self.device = ""
        self.loss_fn = None
        self.evaluator = None

    def forward(self, *args):
        r"""
        Description
        ------------
        The model plays a role of encoder. So the forward will encoder original features into new features.

        Parameters
        -----------
        hg : dgl.DGlHeteroGraph
            the heterogeneous graph
        h_dict : dict
            the dict of heterogeneous feature

        Return
        -------
        out_dic : dict
            A dict of encoded feature. In general, it should ouput all nodes embedding.
            It is allowed that just output the embedding of target nodes which are participated in loss calculation.
        """
        raise NotImplementedError
