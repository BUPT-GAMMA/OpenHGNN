from typing import Optional, Type, Any
import torch.nn as nn


class BaseModel(nn.Module):
    @staticmethod
    def add_args(parser):
        """Add models-specific arguments to the parser."""
        pass

    @classmethod
    def build_model_from_args(cls, args):
        """Build a new models instance."""
        raise NotImplementedError("Models must implement the build_model_from_args method")

    def __init__(self):
        super(BaseModel, self).__init__()
        self.device = ""
        self.loss_fn = None
        self.evaluator = None

    def _forward_unimplemented(self, *input: Any) -> None:  # abc warning
        pass

    def forward(self, *args):
        raise NotImplementedError

    def predict(self, data):
        return self.forward(data)

    def node_classification_loss(self, data, label, mask, category):
        if mask is None:
            mask = data.train_mask
        assert mask.shape[0] == data.y.shape[0]
        pred = self.forward(data)
        return self.loss_fn(pred[mask], data.y[mask])

    def graph_classification_loss(self, batch):
        pred = self.forward(batch)
        return self.loss_fn(pred, batch.y)

    @staticmethod
    # def get_trainer(task: Any, args: Any) -> Optional[Type[BaseTrainer]]:
    #     return None
    def get_trainer(task: Any, args: Any):
        return None

    def set_device(self, device):
        self.device = device

    def set_loss_fn(self, loss_fn):
        self.loss_fn = loss_fn

    def preprocess(self):
        pass

    def trainer(self):
        pass
