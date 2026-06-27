import importlib
from .NEW_model import MLP_follow_model
from .base_model import BaseModel
from torch import nn
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
            raise ValueError(
                "Model ({}: {}) must extend BaseModel".format(name, cls.__name__)
            )
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
    if isinstance(model, nn.Module):
        if not hasattr(model, "build_model_from_args"):

            def build_model_from_args(args, hg):
                return model

            model.build_model_from_args = build_model_from_args
        return model
    if not try_import_model(model):
        exit(1)
    return MODEL_REGISTRY[model]


_MODEL_CLASS_NAMES = {
    "Metapath2vec": "SkipGram",
    "HERec": "SkipGram",
    "GTN": "GTN",
    "fastGTN": "fastGTN",
    "homo_GNN": "homo_GNN",
    "general_HGNN": "general_HGNN",
    "HERO_homo": "HEROHomo",
    "SHGP": "ATT_HGCN",
}


SUPPORTED_MODELS = {
#####       add models here
    'RelGT': 'openhgnn.models.RelGT',
    'HGSketch': 'openhgnn.models.HGSketch',
    'HGDL':'openhgnn.models.HGDL',
    'MHGCN':'openhgnn.models.MHGCN',
    'BPHGNN' : 'openhgnn.models.BPHGNN',
    "MetaHIN": "openhgnn.models.MetaHIN",
    'HGA':'openhgnn.models.HGA',
    'RHINE': 'openhgnn.models.RHINE',
    'FedHGNN':'openhgnn.models.FedHGNN',
    'HCMGNN':'openhgnn.models.HCMGNN',
####################################
    "SIAN": "openhgnn.models.SIAN",
    "CompGCN": "openhgnn.models.CompGCN",
    "HetGNN": "openhgnn.models.HetGNN",
    "HMPNN": "openhgnn.models.HMPNN",
    'RGCN': 'openhgnn.models.RGCN',
    "RGAT": 'openhgnn.models.RGAT',
    'RSHN': 'openhgnn.models.RSHN',
    'Metapath2vec': 'openhgnn.models.SkipGram',
    'HERec': 'openhgnn.models.SkipGram',
    'HAN': 'openhgnn.models.HAN',
    'RoHe': 'openhgnn.models.RoHe',
    'HeCo': 'openhgnn.models.HeCo',
    'HGT': 'openhgnn.models.HGT',
    'GTN': 'openhgnn.models.GTN_sparse',
    'fastGTN': 'openhgnn.models.fastGTN',
    'MHNF': 'openhgnn.models.MHNF',
    'MAGNN': 'openhgnn.models.MAGNN',
    'HeGAN': 'openhgnn.models.HeGAN',
    'NSHE': 'openhgnn.models.NSHE',
    'NARS': 'openhgnn.models.NARS',
    'RHGNN': 'openhgnn.models.RHGNN',
    'HPN': 'openhgnn.models.HPN',
    'KGCN': 'openhgnn.models.KGCN',
    'SLiCE': 'openhgnn.models.SLiCE',
    'HGSL': 'openhgnn.models.HGSL',
    'GCN': 'space4hgnn.homo_models.GCN',
    'GAT': 'space4hgnn.homo_models.GAT',
    'homo_GNN': 'openhgnn.models.homo_GNN',
    'general_HGNN': 'openhgnn.models.general_HGNN',
    'HDE': 'openhgnn.models.HDE',
    'SimpleHGN': 'openhgnn.models.SimpleHGN',
    'GATNE-T': 'openhgnn.models.GATNE',
    'HetSANN': 'openhgnn.models.HetSANN',
    'HGAT': 'openhgnn.models.HGAT',
    'ieHGCN': 'openhgnn.models.ieHGCN',
    'TransE': 'openhgnn.models.TransE',
    'TransH': 'openhgnn.models.TransH',
    'TransR': 'openhgnn.models.TransR',
    'TransD': 'openhgnn.models.TransD',
    'GIE':'openhgnn.models.GIE',
    'GIN':'openhgnn.models.GIN',
    'Rsage': 'openhgnn.models.Rsage',
    'Mg2vec': 'openhgnn.models.Mg2vec',
    'DHNE': 'openhgnn.models.DHNE',
    'DiffMG': 'openhgnn.models.DiffMG',
    'MeiREC': 'openhgnn.models.MeiREC',
    'HGNN_AC': 'openhgnn.models.HGNN_AC',
    'AEHCL': 'openhgnn.models.AEHCL',
    'KGAT': 'openhgnn.models.KGAT',
    'SHGP': 'openhgnn.models.ATT_HGCN',
    'DSSL': 'openhgnn.models.DSSL',
    'HGCL': 'openhgnn.models.HGCL',
    'lightGCN': 'openhgnn.models.lightGCN',
    'SeHGNN' : 'openhgnn.models.SeHGNN',
    'Grail': 'openhgnn.models.Grail',
    'ComPILE': 'openhgnn.models.ComPILE',
    'AdapropT': 'openhgnn.models.AdapropT',
    'AdapropI':'openhgnn.models.AdapropI',
    'LTE': 'openhgnn.models.LTE',
    'LTE_Transe': 'openhgnn.models.LTE_Transe',
    'SACN':'openhgnn.models.SACN',
    'ExpressGNN': 'openhgnn.models.ExpressGNN',
    'NBF': 'openhgnn.models.NBF', 
    'Ingram': 'openhgnn.models.Ingram',
    'RedGNN': 'openhgnn.models.RedGNN',
    'RedGNNT': 'openhgnn.models.RedGNNT',
    'SEHTGNN': 'openhgnn.models.SEHTGNN',
    'HTGformer': 'openhgnn.models.HTGformer',
    "HERO": "openhgnn.models.HERO",
    "HERO_homo":"openhgnn.models.HERO_homo",

    'RMR':'openhgnn.models.RMR',
    'HGOT': 'openhgnn.models.HGOT',
    'HGEN': 'openhgnn.models.HGEN',
}


def __getattr__(name):
    if name in SUPPORTED_MODELS:
        module = importlib.import_module(SUPPORTED_MODELS[name])
        value = getattr(module, _MODEL_CLASS_NAMES.get(name, name))
        globals()[name] = value
        return value
    if name in _MODEL_CLASS_NAMES:
        model_name = name
        module = importlib.import_module(SUPPORTED_MODELS[model_name])
        value = getattr(module, _MODEL_CLASS_NAMES[model_name])
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "BaseModel",
    "MLP_follow_model",
    "MODEL_REGISTRY",
    "SUPPORTED_MODELS",
    "build_model",
    "register_model",
    "try_import_model",
]
classes = list(SUPPORTED_MODELS.keys())
