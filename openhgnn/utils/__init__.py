from __future__ import annotations

from importlib import import_module

from .best_config import BEST_CONFIGS
from .logger import Logger
from .utils import *
from .visualize import *

_LAZY_IMPORTS = {
    "Evaluator": ("openhgnn.utils.evaluator", "Evaluator"),
    "load_HIN": ("openhgnn.utils.dgl_graph", "load_HIN"),
    "load_KG": ("openhgnn.utils.dgl_graph", "load_KG"),
    "load_OGB": ("openhgnn.utils.dgl_graph", "load_OGB"),
    "load_dgl_graph": ("openhgnn.utils.dgl_graph", "load_dgl_graph"),
    "load_link_pred": ("openhgnn.utils.dgl_graph", "load_link_pred"),
    "hgbi": ("openhgnn.utils.hgbi", None),
}


def __getattr__(name: str):
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = _LAZY_IMPORTS[name]
    module = import_module(module_name)
    value = module if attribute_name is None else getattr(module, attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_IMPORTS))
