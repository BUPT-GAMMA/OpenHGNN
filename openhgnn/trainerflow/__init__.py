import importlib
from .base_flow import BaseFlow

FLOW_REGISTRY = {}


def register_flow(name):
    """
    New flow can be added to openhgnn with the :func:`register_flow`
    function decorator.

    For example::

        @register_task('node_classification')
        class NodeClassification(BaseFlow):
            (...)

    Args:
        name (str): the name of the flows
    """

    def register_flow_cls(cls):
        if name in FLOW_REGISTRY:
            raise ValueError("Cannot register duplicate flow ({})".format(name))
        if not issubclass(cls, BaseFlow):
            raise ValueError("Flow ({}: {}) must extend BaseFlow".format(name, cls.__name__))
        FLOW_REGISTRY[name] = cls
        return cls

    return register_flow_cls


def try_import_flow(flow):
    if flow not in FLOW_REGISTRY:
        if flow in SUPPORTED_FLOWS:
            importlib.import_module(SUPPORTED_FLOWS[flow])
        else:
            print(f"Failed to import {flow} flows.")
            return False
    return True


def build_flow(args, flow_name):
    if not try_import_flow(flow_name):
        exit(1)
    return FLOW_REGISTRY[flow_name](args)


SUPPORTED_FLOWS = {
    'entity_classification': 'openhgnn.trainerflow.entity_classification',
    'node_classification': 'openhgnn.trainerflow.node_classification',
    'distmult': 'openhgnn.trainerflow.dist_mult',
    'link_prediction': 'openhgnn.trainerflow.link_prediction',
    'recommendation': 'openhgnn.trainerflow.recommendation',
    'hetgnntrainer': 'openhgnn.trainerflow.hetgnn_trainer',
    'hgttrainer': 'openhgnn.trainerflow.hgt_trainer',
    'nshetrainer': 'openhgnn.trainerflow.nshe_trainer',
    'demo': 'openhgnn.trainerflow.demo',
    'kgcntrainer': 'openhgnn.trainerflow.kgcn_trainer',
    'HeGAN_trainer': 'openhgnn.trainerflow.HeGAN_trainer',
    'HeCo_trainer': 'openhgnn.trainerflow.HeCo_trainer',
    'DMGI_trainer': 'openhgnn.trainerflow.DMGI_trainer'
}