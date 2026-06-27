import importlib
from .base_flow import BaseFlow
from abc import ABC     

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
        if not issubclass(cls, (BaseFlow,ABC)):
            raise ValueError("Flow ({}: {}) must extend BaseFlow or ABC".format(name, cls.__name__))
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
###########     add trainer_flow here. 【register name】 ： 【class name】

    'RelGT_trainer': 'openhgnn.trainerflow.relgt_trainer',
    'HGSketch_trainer': 'openhgnn.trainerflow.HGSketch_trainer',
    'MHGCN_trainer':'openhgnn.trainerflow.MHGCN_trainer',
    'BPHGNN_trainer':'openhgnn.trainerflow.BPHGNN_trainer',    
    'HGMAE':'openhgnn.trainerflow.HGMAE_trainer',
    'hga_trainer':'openhgnn.trainerflow.hga_trainer',
    'rhine_trainer':'openhgnn.trainerflow.RHINE_trainer',
    'FED_REC_trainer':'openhgnn.trainerflow.FED_REC_trainer',
    'HGDL_trainer':'openhgnn.trainerflow.HGDL_trainer',
##########
    "coldstart_recommmendation": "openhgnn.trainerflow.coldstart_recommendation",
    'SIAN_trainer': 'openhgnn.trainerflow.sian_trainer',
    'node_classification': 'openhgnn.trainerflow.node_classification',
    'node_classification_ac': 'openhgnn.trainerflow.node_classification_ac',
    'link_prediction': 'openhgnn.trainerflow.link_prediction',
    'recommendation': 'openhgnn.trainerflow.recommendation',
    'hetgnntrainer': 'openhgnn.trainerflow.hetgnn_trainer',
    'hgttrainer': 'openhgnn.trainerflow.hgt_trainer',
    'nshetrainer': 'openhgnn.trainerflow.nshe_trainer',
    'demo': 'openhgnn.trainerflow.demo',
    'kgcntrainer': 'openhgnn.trainerflow.kgcn_trainer',
    'HeGAN_trainer': 'openhgnn.trainerflow.HeGAN_trainer',
    'mp2vec_trainer': 'openhgnn.trainerflow.mp2vec_trainer',
    'herec_trainer': 'openhgnn.trainerflow.herec_trainer',
    'HeCo_trainer': 'openhgnn.trainerflow.HeCo_trainer',
    'DMGI_trainer': 'openhgnn.trainerflow.DMGI_trainer',
    'slicetrainer': 'openhgnn.trainerflow.slice_trainer',
    'hde_trainer': 'openhgnn.trainerflow.hde_trainer',
    'GATNE_trainer': 'openhgnn.trainerflow.GATNE_trainer',
    'TransX_trainer': 'openhgnn.trainerflow.TransX_trainer',
    'han_nc_trainer': 'openhgnn.trainerflow.han_trainer',
    'han_lp_trainer': 'openhgnn.trainerflow.han_trainer',
    'RoHe_trainer': 'openhgnn.trainerflow.RoHe_trainer',
    'mg2vec_trainer': 'openhgnn.trainerflow.mg2vec_trainer',
    'DHNE_trainer': 'openhgnn.trainerflow.DHNE_trainer',
    'DiffMG_trainer': 'openhgnn.trainerflow.DiffMG_trainer',
    'MeiREC_trainer': 'openhgnn.trainerflow.MeiRec_trainer',
    'abnorm_event_detection': 'openhgnn.trainerflow.AbnormEventDetection',
    'SHGP_trainer': 'openhgnn.trainerflow.SHGP_trainer',
    'KGAT_trainer': 'openhgnn.trainerflow.kgat_trainer',
    'DSSL_trainer': 'openhgnn.trainerflow.DSSL_trainer',
    'hgcltrainer': 'openhgnn.trainerflow.hgcl_trainer',
    'lightGCN_trainer': 'openhgnn.trainerflow.lightGCN_trainer',
    'KTN_trainer':'openhgnn.trainerflow.KTN_trainer',
    'SeHGNN_trainer': 'openhgnn.trainerflow.SeHGNN_trainer',
    'Grail_trainer': 'openhgnn.trainerflow.Grail_trainer',
    'ComPILE_trainer': 'openhgnn.trainerflow.ComPILE_trainer',
    'AdapropT_trainer': 'openhgnn.trainerflow.AdapropT_trainer',
    'AdapropI_trainer':'openhgnn.trainerflow.AdapropI_trainer',
    'LTE_trainer': 'openhgnn.trainerflow.LTE_trainer',
    'SACN_trainer': 'openhgnn.trainerflow.SACN_trainer',
    'ExpressGNN_trainer': 'openhgnn.trainerflow.ExpressGNN_trainer',
    'NBF_trainer':'openhgnn.trainerflow.NBF_trainer',
    'Ingram_Trainer' : 'openhgnn.trainerflow.Ingram_trainer',
    'DisenKGAT_trainer':'openhgnn.trainerflow.DisenKGAT_trainer',
    'RedGNN_trainer': 'openhgnn.trainerflow.RedGNN_trainer',
    'RedGNNT_trainer': 'openhgnn.trainerflow.RedGNNT_trainer',
    'HGPrompt':'openhgnn.trainerflow.HGPrompt',
    'sehtgnn_trainer': 'openhgnn.trainerflow.sehtgnn_trainer',
    'htgformer_trainer': 'openhgnn.trainerflow.htgformer_trainer',
    'hero_trainer': "openhgnn.trainerflow.HERO_trainer",
    'hero_homo_trainer': "openhgnn.trainerflow.HERO_homo_trainer",



    'HCMGNN_trainer':'openhgnn.trainerflow.HCMGNN_trainer',
    'rmr_trainer':'openhgnn.trainerflow.rmr_trainer',
    'HGEN_trainer': 'openhgnn.trainerflow.HGEN_trainer',
}


_FLOW_CLASS_MODULES = {
    "NodeClassification": "openhgnn.trainerflow.node_classification",
    "LinkPrediction": "openhgnn.trainerflow.link_prediction",
    "Recommendation": "openhgnn.trainerflow.recommendation",
    "HetGNNTrainer": "openhgnn.trainerflow.hetgnn_trainer",
    "HGTTrainer": "openhgnn.trainerflow.hgt_trainer",
    "KGCNTrainer": "openhgnn.trainerflow.kgcn_trainer",
    "HeGANTrainer": "openhgnn.trainerflow.HeGAN_trainer",
    "Metapath2VecTrainer": "openhgnn.trainerflow.mp2vec_trainer",
    "HERecTrainer": "openhgnn.trainerflow.herec_trainer",
    "HeCoTrainer": "openhgnn.trainerflow.HeCo_trainer",
    "DMGI_trainer": "openhgnn.trainerflow.DMGI_trainer",
    "SLiCETrainer": "openhgnn.trainerflow.slice_trainer",
    "hde_trainer": "openhgnn.trainerflow.hde_trainer",
    "SIAN_Trainer": "openhgnn.trainerflow.sian_trainer",
    "GATNE": "openhgnn.trainerflow.GATNE_trainer",
    "HANNodeClassification": "openhgnn.trainerflow.han_trainer",
    "HANLinkPrediction": "openhgnn.trainerflow.han_trainer",
    "Mg2vecTrainer": "openhgnn.trainerflow.mg2vec_trainer",
    "DHNE_trainer": "openhgnn.trainerflow.DHNE_trainer",
    "DiffMG_trainer": "openhgnn.trainerflow.DiffMG_trainer",
    "MeiRECTrainer": "openhgnn.trainerflow.MeiRec_trainer",
    "KTN_NodeClassification": "openhgnn.trainerflow.KTN_trainer",
    "SeHGNNtrainer": "openhgnn.trainerflow.SeHGNN_trainer",
    "RelGTTrainer": "openhgnn.trainerflow.relgt_trainer",
    "HGSketchTrainer": "openhgnn.trainerflow.HGSketch_trainer",
    "SEHTGNNTrainer": "openhgnn.trainerflow.sehtgnn_trainer",
    "HTGformerTrainer": "openhgnn.trainerflow.htgformer_trainer",
    "HGENTrainer": "openhgnn.trainerflow.HGEN_trainer",
    "RMR_trainer": "openhgnn.trainerflow.rmr_trainer",
}


def __getattr__(name):
    if name in _FLOW_CLASS_MODULES:
        module = importlib.import_module(_FLOW_CLASS_MODULES[name])
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "BaseFlow",
    "FLOW_REGISTRY",
    "SUPPORTED_FLOWS",
    "build_flow",
    "register_flow",
    "try_import_flow",
]
classes = list(SUPPORTED_FLOWS.keys())
