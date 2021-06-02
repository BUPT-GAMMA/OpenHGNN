import importlib
from .base_dataset import BaseDataset
from .utils import load_acm, load_acm_raw
from .academic_graph import AcademicDataset

DATASET_REGISTRY = {}


def register_dataset(name):
    """
    New dataset types can be added to cogdl with the :func:`register_dataset`
    function decorator.

    For example::

        @register_dataset('my_dataset')
        class MyDataset():
            (...)

    Args:
        name (str): the name of the dataset
    """

    def register_dataset_cls(cls):
        if name in DATASET_REGISTRY:
            raise ValueError("Cannot register duplicate dataset ({})".format(name))
        if not issubclass(cls, BaseDataset):
            raise ValueError("Dataset ({}: {}) must extend cogdl.data.Dataset".format(name, cls.__name__))
        DATASET_REGISTRY[name] = cls
        return cls

    return register_dataset_cls


def try_import_task_dataset(task):
    if task not in DATASET_REGISTRY:
        if task in SUPPORTED_DATASETS:
            importlib.import_module(SUPPORTED_DATASETS[task])
        else:
            print(f"Failed to import {task} dataset.")
            return False
    return True


def build_dataset(dataset, task):
    if not try_import_task_dataset(task):
        exit(1)
    if dataset in ['aifb', 'mutag', 'bgs', 'am']:
        _dataset = 'rdf_' + task
    elif dataset in ['acm4NSHE', 'acm4GTN', 'academic4HetGNN', 'acm_han', 'acm_han_raw', 'dblp', 'dblp4MAGNN',
                     'imdb4MAGNN', 'imdb4GTN']:
        _dataset = 'hin_' + task
    elif dataset in ['ogbn-mag']:
        _dataset = 'ogbn_' + task
    elif dataset in ['wn18', 'FB15k', 'FB15k-237']:
        assert task == 'link_prediction'
        _dataset = 'kg_link_prediction'
    return DATASET_REGISTRY[_dataset](dataset)


SUPPORTED_DATASETS = {
    "node_classification": "openhgnn.dataset.NodeClassificationDataset",
    "link_prediction": "openhgnn.dataset.LinkPredictionDataset",
}
