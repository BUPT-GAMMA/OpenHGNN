from abc import ABC, ABCMeta, abstractmethod


class BaseTask(ABC):

    def __init__(self, ):
        super(BaseTask, self).__init__()
        self.loss_fn = None
        self.evaluator = None

    @abstractmethod
    def evaluate(self, *args, **kwargs):
        pass

