import importlib
from dgl.data import DGLDataset
from .base_dataset import BaseDataset
from .utils import load_acm, load_acm_raw, generate_random_hg
from .academic_graph import AcademicDataset,IMDB4MAGNN_Dataset
from .hgb_dataset import HGBDataset
from .ohgb_dataset import OHGBDataset
from .gtn_dataset import *
from .alircd_dataset import *
from .adapter import AsLinkPredictionDataset, AsNodeClassificationDataset
from .mg2vec_dataset import Mg2vecDataSet
from .meirec_dataset import MeiRECDataset, get_data_loader
from .AdapropT_dataset import AdapropTDataLoader
from .AdapropI_dataset import AdapropIDataLoader
from .LTE_dataset import *
from .SACN_dataset import *
from .NBF_dataset import NBF_Dataset 
from .Ingram_dataset import Ingram_KG_TrainData, Ingram_KG_TestData
from .MetaHIN_dataset import Meta_DataHelper

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
            raise ValueError(
                "Dataset ({}: {}) must extend cogdl.data.Dataset".format(
                    name, cls.__name__
                )
            )
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



common = ['Cora', 'Citeseer', 'Pubmed', 'Texas', 'Cornell']
hgbl_datasets = ['HGBl-amazon', 'HGBl-LastFM', 'HGBl-PubMed']
hgbn_datasets = ['HGBn-ACM', 'HGBn-DBLP', 'HGBn-Freebase', 'HGBn-IMDB']
kg_lp_datasets = ['wn18', 'FB15k', 'EXP_FB15k-237', 'EXP_FB15k-237_data_ratio_0', 'EXP_FB15k-237_data_ratio_0',
                  'EXP_FB15k-237_data_ratio_0.1', 'EXP_FB15k-237_data_ratio_0.2', 'EXP_FB15k-237_data_ratio_zero_shot',
                  'kinship', 'uw_cse']
kg_sub_datasets = [f'fb237_v{i}' for i in range(1, 5)]
kg_sub_datasets += [f'nell_v{i}' for i in range(1, 5)]
kg_sub_datasets += [f'WN18RR_v{i}' for i in range(1,5)]
kg_subT_datasets = ['family']
ohgbl_datasets = ['ohgbl-MTWM', 'ohgbl-yelp1', 'ohgbl-yelp2', 'ohgbl-Freebase']
ohgbn_datasets = ['ohgbn-Freebase', 'ohgbn-yelp2', 'ohgbn-acm', 'ohgbn-imdb']
hypergraph_datasets = ['GPS', 'drug', 'MovieLens', 'wordnet', 'aminer4AEHCL']



def build_dataset_GB(dataset,*args,**kwargs):
    #   dataset："imdb4GTN","HGBl-amazon"
    if dataset in ['imdb4GTN','HGBl-amazon']:
        return DATASET_REGISTRY['GraphBolt_Dataset'](dataset, logger=kwargs['logger'],args = kwargs['args'])
    


def build_dataset(dataset, task, *args, **kwargs):
    args =kwargs.get('args')
    model = args.model
    if isinstance(dataset, DGLDataset):
        return dataset

#######  add  dataset  here

    if dataset == "meirec":
        train_dataloader = get_data_loader("train", batch_size=args[0])
        test_dataloader = get_data_loader("test", batch_size=args[0])
        return train_dataloader, test_dataloader
    elif dataset == 'NL-100':
        train_dataloader = Ingram_KG_TrainData('',dataset)
        valid_dataloader = Ingram_KG_TestData('', dataset,'valid')
        test_dataloader = Ingram_KG_TestData('',dataset,'test')
        return train_dataloader,valid_dataloader,test_dataloader
    elif dataset == 'AdapropT':
        dataload=AdapropTDataLoader(args)
        return dataload
    elif dataset == 'AdapropI':
        dataload=AdapropIDataLoader(args)
        return dataload
    elif dataset == 'SACN' or dataset == 'LTE':
        return
    elif dataset == "dbook":
        dataload = Meta_DataHelper(args.input_dir, args)
        return dataload

#############

    if dataset in CLASS_DATASETS:
        return build_dataset_v2(dataset, task)
    if not try_import_task_dataset(task):
        exit(1)

    _dataset = None
    if dataset in ['aifb', 'mutag', 'bgs', 'am']:
        _dataset = 'rdf_' + task

###########    add dataset here
    elif dataset in ['acm4HGMAE','hgprompt_acm_dblp','acm4FedHGNN']:      
        return DATASET_REGISTRY['common_dataset'](dataset, logger=kwargs['logger'],args = kwargs['args'])

    elif dataset in ['acm4HGA','dblp4HGA']:
        _dataset = 'hga_'+ task
        return DATASET_REGISTRY[_dataset](dataset, logger=kwargs['logger'],args = kwargs['args'])
    elif dataset in ['dblp4RHINE']:
        _dataset = 'rhine_'+task
        return DATASET_REGISTRY[_dataset](dataset, logger=kwargs['logger'],args = kwargs['args'])
    elif dataset in ['dblp4MHGCN','imdb4MHGCN','alibaba4MHGCN']:
        _dataset = 'mhgcn_' + task   
        return DATASET_REGISTRY[_dataset](dataset, logger=kwargs['logger'],args = kwargs['args']) 
##########

    elif dataset in ['acm4NSHE', 'acm4GTN', 'academic4HetGNN', 'acm_han', 'acm_han_raw', 'acm4HeCo', 'dblp',
                     'dblp4MAGNN', 'imdb4MAGNN', 'imdb4GTN', 'acm4NARS', 'demo_graph', 'yelp4HeGAN', 'DoubanMovie',
                     'Book-Crossing', 'amazon4SLICE', 'MTWM', 'HNE-PubMed', 'HGBl-ACM', 'HGBl-DBLP', 'HGBl-IMDB',
                     'amazon', 'yelp4HGSL']:
        _dataset = 'hin_' + task
    elif dataset in ['imdb4MAGNN']:
        _dataset = 'hin_' + task
        return DATASET_REGISTRY[_dataset](dataset, logger=kwargs['logger'],
                                          args = kwargs['args'] )

    elif dataset in ohgbn_datasets + ohgbl_datasets:
        _dataset = "ohgb_" + task
    elif dataset in ["ogbn-mag"]:
        _dataset = "ogbn_" + task
    elif dataset in hgbn_datasets:
        _dataset = "HGBn_node_classification"
    elif dataset in hgbl_datasets:
        _dataset = "HGBl_link_prediction"
    elif dataset in kg_lp_datasets:

        if model == 'ExpressGNN':
            assert task == 'link_prediction'
            _dataset = 'express_gnn'
            return DATASET_REGISTRY[_dataset](dataset, logger=kwargs['logger'], args=kwargs['args'])
        else:
            assert task == 'link_prediction'
            _dataset = 'kg_link_prediction'
    elif dataset in kg_sub_datasets:
        assert task == 'link_prediction'
        _dataset = 'kg_sub_link_prediction'
    elif dataset in kg_subT_datasets:
        assert task == 'link_prediction'
        _dataset = 'kg_subT_link_prediction'
    elif dataset in ['LastFM4KGCN']:
        _dataset = 'kgcn_recommendation'
    elif dataset in ['yelp']:
        _dataset = 'sian_recommendation'
    elif dataset in ['gowalla', 'yelp2018', 'amazon-book']:
        _dataset = 'lightGCN_recommendation'
    elif dataset in ['yelp4rec']:
        _dataset = 'hin_' + task
    elif dataset in ['Epinions', 'CiaoDVD', 'Yelp']:
        _dataset = 'hgcl_recommendation'
    elif dataset in ['dblp4Mg2vec_4', 'dblp4Mg2vec_5']:
        _dataset = 'hin_' + task
    elif dataset == 'demo':
        _dataset = 'demo_' + task
    elif dataset == 'mag':
        _dataset = 'mag_dataset'
    elif dataset in hypergraph_datasets:
        _dataset = task
    elif dataset in ['LastFM_KGAT', 'yelp2018_KGAT', 'amazon-book_KGAT']:
        change_name = {'LastFM_KGAT': 'last-fm', 'yelp2018_KGAT': 'yelp2018', 'amazon-book_KGAT': 'amazon-book'}
        dataset = change_name[dataset]
        _dataset = 'kgat_recommendation'
    elif dataset in common:
        if model == 'ExpressGNN' and dataset == 'Cora':
            assert task == 'link_prediction'
            _dataset = 'express_gnn'
            return DATASET_REGISTRY[_dataset](dataset, logger=kwargs['logger'], args=kwargs['args'])
        else:
            _dataset = 'common_' + task
    elif dataset in ['NBF_WN18RR','NBF_FB15k-237']:
        _dataset = 'NBF_' + task  
    elif dataset in ['DisenKGAT_WN18RR','DisenKGAT_FB15k-237']:
        _dataset = 'DisenKGAT_' + task  #  == 'DisenKGAT_link_prediction'
        return DATASET_REGISTRY[_dataset](dataset, logger=kwargs['logger'],args = kwargs.get('args'))  

    if kwargs['args'].model=='Grail' or kwargs['args'].model=='ComPILE':
        _dataset = 'grail_'+ task
        return DATASET_REGISTRY[_dataset](dataset, logger=kwargs['logger'],args=kwargs['args'])
    return DATASET_REGISTRY[_dataset](dataset, logger=kwargs['logger'])



SUPPORTED_DATASETS = {
    "node_classification": "openhgnn.dataset.NodeClassificationDataset",
    "link_prediction": "openhgnn.dataset.LinkPredictionDataset",
    "recommendation": "openhgnn.dataset.RecommendationDataset",
    "edge_classification": "openhgnn.dataset.EdgeClassificationDataset",
    "hypergraph": "openhgnn.dataset.HypergraphDataset",
    "pretrain": "openhgnn.dataset.mag_dataset",
    "ktn": "openhgnn.dataset.oag_dataset",

}

from .NodeClassificationDataset import NodeClassificationDataset
from .LinkPredictionDataset import LinkPredictionDataset
from .RecommendationDataset import RecommendationDataset
from .EdgeClassificationDataset import EdgeClassificationDataset
from .HypergraphDataset import HGraphDataset
from .oag_dataset import OAGDataset



def build_dataset_v2(dataset, task):
    if dataset in CLASS_DATASETS:
        path = ".".join(CLASS_DATASETS[dataset].split(".")[:-1])
        module = importlib.import_module(path)
        class_name = CLASS_DATASETS[dataset].split(".")[-1]
        dataset_class = getattr(module, class_name)

        d = dataset_class( )
        if task == 'node_classification':
            target_ntype = getattr(d, 'category')
            if target_ntype is None:
                target_ntype = getattr(d, "target_ntype")
            res = AsNodeClassificationDataset(d, target_ntype=target_ntype)
        elif task == "link_prediction":
            target_link = getattr(d, "target_link")
            target_link_r = getattr(d, "target_link_r")
            res = AsLinkPredictionDataset(
                d, target_link=target_link, target_link_r=target_link_r
            )
        return res


CLASS_DATASETS = {
    "dblp4GTN": "openhgnn.dataset.DBLP4GTNDataset",
    # "acm4GTN": "openhgnn.dataset.ACM4GTNDataset",
    "imdb4GTN": "openhgnn.dataset.IMDB4GTNDataset",
    "alircd_small": "openhgnn.dataset.AliRCDSmallDataset",
    "alircd_session1": "openhgnn.dataset.AliRCDSession1Dataset",
    "ICDM": "openhgnn.dataset.AliICDMDataset",
    "ohgbn-alircd_session1": "openhgnn.dataset.AliRCDSession1Dataset",
    "alircd_session2": "openhgnn.dataset.AliRCDSession2Dataset",
    "ohgbn-alircd_session2": "openhgnn.dataset.AliRCDSession2Dataset",
    "pretrain": "openhgnn.dataset.mag_dataset",
    "ktn": "openhgnn.dataset.oag_dataset",

}

__all__ = [
    "BaseDataset",
    "NodeClassificationDataset",
    "LinkPredictionDataset",
    "RecommendationDataset",
    "AcademicDataset",
    "HGBDataset",
    "OHGBDataset",
    "GTNDataset",
    "AsLinkPredictionDataset",
    "AsNodeClassificationDataset",
    "EdgeClassificationDataset",
    "HGraphDataset",
    "AbnormEventDetectionDataset",
    "mag_dataset",
    "OAGDataset",
]

classes = __all__
