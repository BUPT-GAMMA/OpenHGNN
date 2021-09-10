import argparse
from openhgnn.config import Config
from openhgnn.utils import set_random_seed, set_best_config
from openhgnn.trainerflow import build_flow
from openhgnn.auto import hpo_experiment
from space4hgnn.utils import read_config
import os

def OpenHGNN(args):
    for i in range(args.repeat):
        args.seed = i
        set_random_seed(args.seed)
        path = './prediction/{}_{}_{}'.format(args.model, args.aggregation, args.times)
        if not os.path.exists(path):
            os.makedirs(path)
        if 'HGB' in args.dataset:
            args.HGB_results_path = './prediction/{}_{}_{}/{}_{}.txt'.format(args.model, args.aggregation, args.times, args.dataset[5:], str(i+1))
        else:
            args.results_path = './prediction/{}_{}_{}/{}_{}.txt'.format(args.model, args.aggregation, args.times, args.dataset, str(i+1))
        print(args)
        flow = build_flow(args, args.task)
        flow.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default='homo_GNN', type=str, help='name of models')
    parser.add_argument('--task', '-t', default='link_prediction', type=str, help='name of task')
    # link_prediction / node_classification / recommendation
    parser.add_argument('--dataset', '-d', default='HGBl-PubMed', type=str, help='name of datasets')
    parser.add_argument('--gpu', '-g', default='0', type=int, help='-1 means cpu')
    parser.add_argument('--repeat', '-r', default='5', type=int, help='-1 means cpu')
    parser.add_argument('--aggregation', '-a', default='gcnconv', type=str, help='aggregation type')
    parser.add_argument('--times', '-s', default=2, type=int, help='which yaml file')
    args = parser.parse_args()

    args = read_config(args)
    #config = Config(file_path=config_file, model=args.model, dataset=args.dataset, task=args.task, gpu=args.gpu)
    OpenHGNN(args=args)
