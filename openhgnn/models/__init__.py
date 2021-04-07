import importlib

from .base_model import BaseModel
from .EmbedLayer import HeteroEmbedLayer

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


def build_model(args):
    if not try_import_model(args.model):
        exit(1)
    return MODEL_REGISTRY[args.model].build_model_from_args(args)


SUPPORTED_MODELS = {
    "CompGCN": "openhgnn.models.CompGCN",
    "spectral": "cogdl.models.emb.spectral",
    "hin2vec": "cogdl.models.emb.hin2vec",
}
