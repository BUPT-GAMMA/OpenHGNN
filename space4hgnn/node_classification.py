import argparse
from openhgnn.utils import set_random_seed, set_best_config
from openhgnn.trainerflow import build_flow
from openhgnn.auto import hpo_experiment
from space4hgnn.utils import read_config

def OpenHGNN(args):
    if not getattr(args, 'seed', False):
        args.seed = 0
    set_random_seed(args.seed)

    if getattr(args, "use_best_config", False):
        args = set_best_config(args)

    print(args)
    if getattr(args, "use_hpo", False):
        # hyper-parameter search
        hpo_experiment(args, 'node_classification')
    else:
        flow = build_flow(args, 'node_classification')
        result = flow.train()
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default='GCN', type=str, help='name of models')
    parser.add_argument('--task', '-t', default='node_classification', type=str, help='name of task')
    # link_prediction / node_classification
    parser.add_argument('--dataset', '-d', default='HGBn-DBLP', type=str, help='name of datasets')
    parser.add_argument('--gpu', '-g', default='0', type=int, help='-1 means cpu')
    parser.add_argument('--use_best_config', action='store_true', help='use best config')
    parser.add_argument('--use_hpo', action='store_true', help='use hpo')
    args = parser.parse_args()

    args = read_config(args)
    #config = Config(file_path=config_file, model=args.model, dataset=args.dataset, task=args.task, gpu=args.gpu)
    OpenHGNN(args=args)
