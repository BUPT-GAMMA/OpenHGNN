from __future__ import annotations

from importlib import import_module

from ._version import __version__

__all__ = [
    "__version__",
    "Experiment",
    "RECOMMENDED_ENVIRONMENTS",
    "get_environment_report",
    "get_supported_registry_items",
    "validate_registry",
]


class Experiment:
    def __new__(cls, *args, **kwargs):
        module = import_module("openhgnn.experiment")
        experiment_cls = getattr(module, "Experiment")
        return experiment_cls(*args, **kwargs)


_LAZY_IMPORTS = {
    "RECOMMENDED_ENVIRONMENTS": ("openhgnn._metadata", "RECOMMENDED_ENVIRONMENTS"),
    "get_environment_report": ("openhgnn._metadata", "get_environment_report"),
    "get_supported_registry_items": ("openhgnn.registry", "get_supported_registry_items"),
    "validate_registry": ("openhgnn.registry", "validate_registry"),
}


def __getattr__(name: str):
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = _LAZY_IMPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
