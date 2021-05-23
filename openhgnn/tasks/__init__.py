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


def build_task(args):
    if not try_import_task(args.task):
        exit(1)
    return TASK_REGISTRY[args.task](args)


def try_import_task(task):
    if task not in TASK_REGISTRY:
        if task in SUPPORTED_TASKS:
            importlib.import_module(SUPPORTED_TASKS[task])
        else:
            print(f"Failed to import {task} task.")
            return False
    return True


SUPPORTED_TASKS = {
    'node_classification': 'openhgnn.tasks.node_classification',
    'link_prediction': 'openhgnn.tasks.link_prediction'
}