import importlib
from .NEW_model import MLP_follow_model
from .base_model import BaseModel
import sys
sys.path.append("..")

MODEL_REGISTRY = {}


def register_model(name):
    """
    New models types can be added to cogdl with the :func:`register_model`
    function decorator.

    For example::

        @register_model('gat')
        class GAT(BaseModel):
            (...)

    Args:
        name (str): the name of the models
    """

    def register_model_cls(cls):
        if name in MODEL_REGISTRY:
            raise ValueError("Cannot register duplicate models ({})".format(name))
        if not issubclass(cls, BaseModel):
            raise ValueError("Model ({}: {}) must extend BaseModel".format(name, cls.__name__))
        MODEL_REGISTRY[name] = cls
        cls.model_name = name
        return cls

    return register_model_cls


def try_import_model(model):
    if model not in MODEL_REGISTRY:
        if model in SUPPORTED_MODELS:
            importlib.import_module(SUPPORTED_MODELS[model])
        else:
            print(f"Failed to import {model} models.")
            return False
    return True


def build_model(model):
    if not try_import_model(model):
        exit(1)
    return MODEL_REGISTRY[model]


SUPPORTED_MODELS = {
    "CompGCN": "openhgnn.models.CompGCN",
    "HetGNN": "openhgnn.models.HetGNN",
    'RGCN': 'openhgnn.models.RGCN',
    "RGAT": 'openhgnn.models.RGAT',
    'RSHN': 'openhgnn.models.RSHN',
    'Metapath2vec': 'openhgnn.models.SkipGram',
    'HERec': 'openhgnn.models.SkipGram',
    'HAN': 'openhgnn.models.HAN',
    #'HGT': 'openhgnn.models.HGT',
    'HeCo': 'openhgnn.models.HeCo',
    'HGT': 'openhgnn.models.HGT_hetero',
    'GTN': 'openhgnn.models.GTN_sparse',
    'fastGTN': 'openhgnn.models.fastGTN',
    'MHNF': 'openhgnn.models.MHNF',
    'MAGNN': 'openhgnn.models.MAGNN',
    'HeGAN':'openhgnn.models.HeGAN',
    'NSHE': 'openhgnn.models.NSHE',
    'NARS': 'openhgnn.models.NARS',
    'RHGNN': 'openhgnn.models.RHGNN',
    'HPN': 'openhgnn.models.HPN',
    'KGCN': 'openhgnn.models.KGCN',
    'SLiCE':'openhgnn.models.SLiCE',
    'HGSL': 'openhgnn.models.HGSL',
    'GCN': 'space4hgnn.homo_models.GCN',
    'GAT': 'space4hgnn.homo_models.GAT',
    'homo_GNN': 'openhgnn.models.homo_GNN',
    'general_HGNN': 'openhgnn.models.general_HGNN',
    'HDE': 'openhgnn.models.HDE',
    'SimpleHGN': 'openhgnn.models.SimpleHGN',
}
