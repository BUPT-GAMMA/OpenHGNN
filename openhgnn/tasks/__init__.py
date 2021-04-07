import os
import importlib
from .base_task import BaseTask

TASK_REGISTRY = {}


def register_task(name):
    """
    New tasks types can be added to cogdl with the :func:`register_task`
    function decorator.

    For example::

        @register_task('node_classification')
        class NodeClassification(BaseTask):
            (...)

    Args:
        name (str): the name of the tasks
    """

    def register_task_cls(cls):
        if name in TASK_REGISTRY:
            raise ValueError("Cannot register duplicate tasks ({})".format(name))
        if not issubclass(cls, BaseTask):
            raise ValueError("Task ({}: {}) must extend BaseTask".format(name, cls.__name__))
        TASK_REGISTRY[name] = cls
        return cls

    return register_task_cls


for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith(".py") and not file.startswith("_"):
        task_name = file[: file.find(".py")]
        module = importlib.import_module("openhgnn.tasks." + task_name)


def build_task(args, dataset=None, model=None):
    if dataset is None and model is None:
        return TASK_REGISTRY[args.task](args)
    elif dataset is not None and model is None:
        return TASK_REGISTRY[args.task](args, dataset=dataset)
    elif dataset is None and model is not None:
        return TASK_REGISTRY[args.task](args, model=model)
    return TASK_REGISTRY[args.task](args, dataset=dataset, model=model)