from .utils import set_random_seed, set_best_config
from .trainerflow import build_flow
from .auto import hpo_experiment


def OpenHGNN(args):
    if not getattr(args, 'seed', False):
        args.seed = 0
    set_random_seed(args.seed)

    if getattr(args, "use_best_config", False):
        args = set_best_config(args)
    if hasattr(args, 'trainerflow'):
        trainerflow = args.trainerflow
    else:
        trainerflow = get_trainerflow(args.model, args.task)
    print(args)
    if getattr(args, "use_hpo", False):
        # hyper-parameter search
        hpo_experiment(args, trainerflow)
    else:
        flow = build_flow(args, trainerflow)
        result = flow.train()
        return result


def get_trainerflow(model, task):
    if model == 'demo':
        if task in ['node_classification']:
            return 'entity_classification'
        elif task in ['link_prediction']:
            return 'link_prediction'
        elif task == 'demo':
            return 'demo'
    elif model in ['RGCN', 'CompGCN', 'RSHN']:
        if task in ['node_classification']:
            return 'entity_classification'
        if task in ['link_prediction']:
            return 'link_prediction'
    elif model in ['HetGNN']:
        return 'hetgnntrainer'
    elif model in ['HAN', 'MAGNN', 'GTN', 'NARS', 'MHNF', 'RHGNN', 'HPN']:
        if task in ['node_classification']:
            return 'node_classification'
        if task in ['link_prediction']:
            return 'link_prediction'
    elif model in ['MAGNN_AC']:
        return 'node_classification_ac'
    elif model in ['HGT', 'HGT_hetero']:
        return 'hgttrainer'
    elif model in ['NSHE']:
        return 'nshetrainer'
    elif model == 'DMGI':
        return 'DMGI_trainer'
    elif model in ['KGCN']:
        return 'kgcntrainer'


def trans_feature(hg, het_gnn):
    for i in hg.ntypes:
        ndata = hg.nodes[i].data
        for j in ndata:
            het_gnn.nodes[i].data[j] = ndata[j]
    return het_gnn


