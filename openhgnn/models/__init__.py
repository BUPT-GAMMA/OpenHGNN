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


SUPPORTED_MODELS = {
    "CompGCN": "openhgnn.models.CompGCN",
    "HetGNN": "openhgnn.models.HetGNN",
    "RGCN": "openhgnn.models.RGCN",
    "RGAT": "openhgnn.models.RGAT",
    "RSHN": "openhgnn.models.RSHN",
    "Metapath2vec": "openhgnn.models.SkipGram",
    "HERec": "openhgnn.models.SkipGram",
    "HAN": "openhgnn.models.HAN",
    "RoHe": "openhgnn.models.RoHe",
    "HeCo": "openhgnn.models.HeCo",
    "HGT": "openhgnn.models.HGT",
    "GTN": "openhgnn.models.GTN_sparse",
    "fastGTN": "openhgnn.models.fastGTN",
    "MHNF": "openhgnn.models.MHNF",
    "MAGNN": "openhgnn.models.MAGNN",
    "HeGAN": "openhgnn.models.HeGAN",
    "NSHE": "openhgnn.models.NSHE",
    "NARS": "openhgnn.models.NARS",
    "RHGNN": "openhgnn.models.RHGNN",
    "HPN": "openhgnn.models.HPN",
    "KGCN": "openhgnn.models.KGCN",
    "SLiCE": "openhgnn.models.SLiCE",
    "HGSL": "openhgnn.models.HGSL",
    "GCN": "space4hgnn.homo_models.GCN",
    "GAT": "space4hgnn.homo_models.GAT",
    "homo_GNN": "openhgnn.models.homo_GNN",
    "general_HGNN": "openhgnn.models.general_HGNN",
    "HDE": "openhgnn.models.HDE",
    "SimpleHGN": "openhgnn.models.SimpleHGN",
    "GATNE-T": "openhgnn.models.GATNE",
    "HetSANN": "openhgnn.models.HetSANN",
    "HGAT": "openhgnn.models.HGAT",
    "ieHGCN": "openhgnn.models.ieHGCN",
    "TransE": "openhgnn.models.TransE",
    "TransH": "openhgnn.models.TransH",
    "TransR": "openhgnn.models.TransR",
    "TransD": "openhgnn.models.TransD",
    "GIE": "openhgnn.models.GIE",
    "GIN": "openhgnn.models.GIN",
    "Rsage": "openhgnn.models.Rsage",
    "Mg2vec": "openhgnn.models.MG2vec",
    "DHNE": "openhgnn.models.DHNE",
    "DiffMG": "openhgnn.models.DiffMG",
    "MeiREC": "openhgnn.models.MeiREC",
    "HGNN_AC": "openhgnn.models.HGNN_AC",
    "AEHCL": "openhgnn.models.AEHCL",
    "KGAT": "openhgnn.models.KGAT",
    "SHGP": "openhgnn.models.ATT_HGCN",
    "DSSL": "openhgnn.models.DSSL",
    "HGCL": "openhgnn.models.HGCL",
    "lightGCN": "openhgnn.models.lightGCN",
    "HMPNN": "openhgnn.models.HMPNN",
}

from .HGCL import HGCL
from .CompGCN import CompGCN
from .HetGNN import HetGNN
from .RGCN import RGCN
from .RGAT import RGAT
from .RSHN import RSHN
from .SkipGram import SkipGram
from .HAN import HAN
from .RoHe import RoHe
from .HeCo import HeCo
from .HGT import HGT
from .GTN_sparse import GTN
from .fastGTN import fastGTN
from .MHNF import MHNF
from .MAGNN import MAGNN
from .HeGAN import HeGAN
from .NSHE import NSHE
from .NARS import NARS
from .RHGNN import RHGNN
from .HPN import HPN
from .KGCN import KGCN
from .SLiCE import SLiCE
from .HGSL import HGSL
from .homo_GNN import homo_GNN
from .general_HGNN import general_HGNN
from .HDE import HDE
from .SimpleHGN import SimpleHGN
from .HetSANN import HetSANN
from .ieHGCN import ieHGCN
from .HGAT import HGAT
from .GATNE import GATNE
from .Rsage import Rsage
from .Mg2vec import Mg2vec
from .DHNE import DHNE
from .DiffMG import DiffMG
from .MeiREC import MeiREC
from .HGNN_AC import HGNN_AC
from .KGAT import KGAT
from .DSSL import DSSL
from .lightGCN import lightGCN
from .HMPNN import HMPNN

__all__ = [
    "BaseModel",
    "CompGCN",
    "HetGNN",
    "RGCN",
    "RGAT",
    "RSHN",
    "SkipGram",
    "HAN",
    "HeCo",
    "HGT",
    "GTN",
    "fastGTN",
    "MHNF",
    "MAGNN",
    "HeGAN",
    "NSHE",
    "NARS",
    "RHGNN",
    "HPN",
    "KGCN",
    "SLiCE",
    "HGSL",
    "homo_GNN",
    "general_HGNN",
    "HDE",
    "SimpleHGN",
    "GATNE",
    "Rsage",
    "Mg2vec",
    "DHNE",
    "DiffMG",
    "MeiREC",
    "KGAT",
    "ATT_HGCN",
    "KGAT",
    "DSSL",
    "lightGCN",
    "HMPNN",
]
classes = __all__
