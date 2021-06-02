from openhgnn.utils.trainer import run, run_GTN, run_RSHN, run_RGCN, run_CompGCN, run_HetGNN
from openhgnn.utils.evaluater import evaluate
from openhgnn.utils import set_random_seed
from openhgnn.utils.dgl_graph import load_HIN, load_KG, load_link_pred
import torch.nn.functional as F
import torch as th
from openhgnn.trainerflow import build_flow


def OpenHGNN(args):
    set_random_seed(args.seed)

    # TODO find the best parameter
    # if getattr(args, "use_best_config", False):
    #     args = set_best_config(args)
    if hasattr(args, 'trainerflow'):
        trainerflow = args.trainerflow
    else:
        trainerflow = get_trainerflow(args.model, args.task)
    print(args)
    flow = build_flow(args, trainerflow)
    result = flow.train()

    return result

def get_trainerflow(model, task):
    if model in ['RGCN', 'CompGCN', 'RSHN']:
        if task in ['node_classification']:
            return 'entity_classification'
        if task in ['link_prediction']:
            return 'distmult'
    elif model in ['HetGNN']:
        return 'hetgnntrainer'
    elif model in ['HAN', 'MAGNN', 'GTN']:
        if task in ['node_classification']:
            return 'node_classification'
    elif model in ['HGT', 'HGT_hetero']:
        return 'hgttrainer'
    elif model in ['NSHE']:
        return 'nshetrainer'


def trans_feature(hg, het_gnn):
    for i in hg.ntypes:
        ndata = hg.nodes[i].data
        for j in ndata:
            het_gnn.nodes[i].data[j] = ndata[j]
    return het_gnn


