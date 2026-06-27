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
            raise ValueError(
                "Task ({}: {}) must extend BaseTask".format(name, cls.__name__)
            )
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
    "coldstart_recommendation": "openhgnn.tasks.coldstart_recommendation",
    "KTN_trainer": "openhgnn.tasks.ktn",
    'demo': 'openhgnn.tasks.demo',
    'node_classification': 'openhgnn.tasks.node_classification',
    'link_prediction': 'openhgnn.tasks.link_prediction',
    'recommendation': 'openhgnn.tasks.recommendation',
    'embedding': 'openhgnn.tasks.embedding',
    'edge_classification': 'openhgnn.tasks.edge_classification',
    'hypergraph': 'openhgnn.tasks.hypergraph',
    'meirec': 'openhgnn.tasks.meirec',
    'pretrain': 'openhgnn.tasks.pretrain',
    'abnorm_event_detection': 'openhgnn.tasks.AbnormEventDetection',
    'DSSL_trainer': 'openhgnn.tasks.node_classification',
    'NBF_link_prediction':'openhgnn.tasks.link_prediction',
    'Ingram': 'openhgnn.tasks.Ingram_task',
    'DisenKGAT_link_prediction':'openhgnn.tasks.link_prediction',
    "node_regression": "openhgnn.tasks.node_regression",
}


_TASK_CLASS_MODULES = {
    "NodeClassification": "openhgnn.tasks.node_classification",
    "LinkPrediction": "openhgnn.tasks.link_prediction",
    "Recommendation": "openhgnn.tasks.recommendation",
    "EdgeClassification": "openhgnn.tasks.edge_classification",
    "hypergraph": "openhgnn.tasks.hypergraph",
    "KTN4MultiLabelNodeClassification": "openhgnn.tasks.ktn",
    "Ingram": "openhgnn.tasks.Ingram_task",
    "NodeRegression": "openhgnn.tasks.node_regression",
}


def __getattr__(name):
    if name in _TASK_CLASS_MODULES:
        module = importlib.import_module(_TASK_CLASS_MODULES[name])
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "BaseTask",
    "TASK_REGISTRY",
    "SUPPORTED_TASKS",
    "build_task",
    "register_task",
    "try_import_task",
]

classes = list(SUPPORTED_TASKS.keys())
