from . import register_flow
from .node_classification import NodeClassification


@register_flow('HGEN_trainer')
class HGENTrainer(NodeClassification):
    r"""
    Trainer flow for HGEN node classification.

    HGEN follows the standard full-batch node classification pipeline, but its
    official training objective adds a model-specific path-correlation
    regularization term. Keep that objective local to HGEN instead of changing
    the shared node_classification flow.
    """

    def __init__(self, args):
        if getattr(args, 'mini_batch_flag', False):
            raise ValueError('HGEN_trainer only supports full-batch training.')
        super(HGENTrainer, self).__init__(args)

    def _full_train_step(self):
        self.model.train()
        h_dict = self.model.input_feature()
        self.hg = self.hg.to(self.device)
        logits = self.model(self.hg, h_dict)[self.category]
        loss = self.loss_fn(logits[self.train_idx], self.labels[self.train_idx])
        loss = loss + self.model.extra_loss()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
