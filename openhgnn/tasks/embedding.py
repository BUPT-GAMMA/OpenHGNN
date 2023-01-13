from . import BaseTask, register_task
from ..dataset import build_dataset


@register_task("embedding")
class Embedding(BaseTask):
    """Embedding task. Generate embeddings for nodes in the graph."""

    def __init__(self, args):
        super(Embedding, self).__init__()
        self.n_dataset = args.dataset
        self.dataset = build_dataset(args.dataset, 'embedding')

    def get_graph(self):
        return getattr(self.dataset, 'g', self.dataset[0])

    def get_loss_fn(self):
        pass

    def evaluate(self):
        pass
