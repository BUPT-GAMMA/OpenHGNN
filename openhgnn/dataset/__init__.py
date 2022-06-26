import importlib
from dgl.data import DGLDataset
from .base_dataset import BaseDataset
from .utils import load_acm, load_acm_raw, generate_random_hg
from .academic_graph import AcademicDataset
from .hgb_dataset import HGBDataset
from .ohgb_dataset import OHGBDataset
from .gtn_dataset import *
from .adapter import AsLinkPredictionDataset, AsNodeClassificationDataset

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


def build_dataset(dataset, task, *args, **kwargs):
    if isinstance(dataset, DGLDataset):
        return dataset
    if dataset in CLASS_DATASETS:
        return build_dataset_v2(dataset, task)
    if not try_import_task_dataset(task):
        exit(1)
    if dataset in ['aifb', 'mutag', 'bgs', 'am']:
        _dataset = 'rdf_' + task
    elif dataset in ['acm4NSHE', 'acm4GTN', 'academic4HetGNN', 'acm_han', 'acm_han_raw', 'acm4HeCo', 'dblp', 'dblp4MAGNN',
                     'imdb4MAGNN', 'imdb4GTN', 'acm4NARS', 'demo_graph', 'yelp4HeGAN', 'DoubanMovie', 'Book-Crossing',
                     'amazon4SLICE', 'MTWM', 'HNE-PubMed', 'HGBl-ACM', 'HGBl-DBLP', 'HGBl-IMDB']:
        _dataset = 'hin_' + task
    elif dataset in ['ohgbl-MTWM', 'ohgbl-yelp1', 'ohgbl-yelp2', 'ohgbl-Freebase',
                     'ohgbn-Freebase', 'ohgbn-yelp2', 'ohgbn-acm', 'ohgbn-imdb']:
        _dataset = 'ohgb_' + task
    elif dataset in ['ogbn-mag']:
        _dataset = 'ogbn_' + task
    elif dataset in ['HGBn-ACM', 'HGBn-DBLP', 'HGBn-Freebase', 'HGBn-IMDB']:
        _dataset = 'HGBn_node_classification'
    elif dataset in ['HGBl-amazon', 'HGBl-LastFM', 'HGBl-PubMed']:
        _dataset = 'HGBl_link_prediction'
    elif dataset in ['wn18', 'FB15k', 'FB15k-237']:
        assert task == 'link_prediction'
        _dataset = 'kg_link_prediction'
    elif dataset in ['LastFM4KGCN']:
        _dataset = 'kgcn_recommendation'
    elif dataset in ['yelp4rec']:
        _dataset = 'hin_' + task
    elif dataset == 'demo':
        _dataset = 'demo_' + task
    return DATASET_REGISTRY[_dataset](dataset, logger=kwargs['logger'])


SUPPORTED_DATASETS = {
    "node_classification": "openhgnn.dataset.NodeClassificationDataset",
    "link_prediction": "openhgnn.dataset.LinkPredictionDataset",
    "recommendation": "openhgnn.dataset.RecommendationDataset"
}

from .NodeClassificationDataset import NodeClassificationDataset
from .LinkPredictionDataset import LinkPredictionDataset
from .RecommendationDataset import RecommendationDataset


def build_dataset_v2(dataset, task):
    if dataset in CLASS_DATASETS:
        path = ".".join(CLASS_DATASETS[dataset].split(".")[:-1])
        module = importlib.import_module(path)
        class_name = CLASS_DATASETS[dataset].split(".")[-1]
        dataset_class = getattr(module, class_name)
        d = dataset_class()
        if task == 'node_classification':
            target_ntype = getattr(d, 'category')
            if target_ntype is None:
                target_ntype = getattr(d, 'target_ntype')
            res = AsNodeClassificationDataset(d, target_ntype=target_ntype)
        elif task == 'link_prediction':
            target_link = getattr(d, 'target_link')
            target_link_r = getattr(d, 'target_link_r')
            res = AsLinkPredictionDataset(d, target_link=target_link, target_link_r=target_link_r)
        return res


CLASS_DATASETS = {
    "dblp4GTN": "openhgnn.dataset.DBLP4GTNDataset",
    "acm4GTN": "openhgnn.dataset.ACM4GTNDataset",
    "imdb4GTN": "openhgnn.dataset.IMDB4GTNDataset",
}

__all__ = [
    'BaseDataset',
    'NodeClassificationDataset',
    'LinkPredictionDataset',
    'RecommendationDataset',
    'AcademicDataset',
    'HGBDataset',
    'OHGBDataset',
    'GTNDataset',
    'AsLinkPredictionDataset',
    'AsNodeClassificationDataset'
]

classes = __all__