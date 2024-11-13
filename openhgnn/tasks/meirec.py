from . import BaseTask, register_task
from ..dataset import build_dataset


@register_task("meirec")
class MeiREC(BaseTask):
    """Recommendation tasks."""

    def __init__(self, args):
        super().__init__()
        self.logger = args.logger
        self.n_dataset = args.dataset
        self.train_loader, self.test_loader = build_dataset(args.dataset,
                                                            args.task,
                                                            args.batch_num,
                                                            logger=self.logger,
                                                            args = args
                                                           )

    def evaluate(self, y_true, y_score):
        pass

