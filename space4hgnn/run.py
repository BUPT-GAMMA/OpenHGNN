import argparse
from openhgnn.config import Config
from openhgnn.utils import set_random_seed, set_best_config
from openhgnn.trainerflow import build_flow
from openhgnn.auto import hpo_experiment
from space4hgnn.utils import read_config


def OpenHGNN(args):
    for i in range(args.repeat):
        args.seed = i
        set_random_seed(args.seed)
        args.HGB_results_path = './prediction/{}_{}.txt'.format(args.dataset[5:], str(i+1))
        print(args)
        flow = build_flow(args, args.task)
        flow.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default='homo_GNN', type=str, help='name of models')
    parser.add_argument('--task', '-t', default='link_prediction', type=str, help='name of task')
    # link_prediction / node_classification
    parser.add_argument('--dataset', '-d', default='HGBl-LastFM', type=str, help='name of datasets')
    parser.add_argument('--gpu', '-g', default='0', type=int, help='-1 means cpu')
    parser.add_argument('--repeat', '-r', default='5', type=int, help='-1 means cpu')
    args = parser.parse_args()

    args = read_config(args)
    #config = Config(file_path=config_file, model=args.model, dataset=args.dataset, task=args.task, gpu=args.gpu)
    OpenHGNN(args=args)