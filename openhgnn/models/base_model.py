from typing import Optional, Type, Any
from abc import ABCMeta, abstractmethod
import torch.nn as nn


class BaseModel(nn.Module, metaclass=ABCMeta):
    @classmethod
    def build_model_from_args(cls, args, hg):
        """Build a new models instance."""
        raise NotImplementedError("Models must implement the build_model_from_args method")

    def __init__(self):
        super(BaseModel, self).__init__()
        self.device = ""
        self.loss_fn = None
        self.evaluator = None

    def forward(self, *args):
        raise NotImplementedError

    def predict(self, data):
        raise NotImplementedError