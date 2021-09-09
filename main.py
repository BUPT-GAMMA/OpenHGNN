# @Time   : 2021/1/28
# @Author : Tianyu Zhao
# @Email  : tyzhao@bupt.edu.cn


import argparse
from openhgnn.config import Config
from openhgnn.start import OpenHGNN


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default='RGCN', type=str, help='name of models')
    parser.add_argument('--task', '-t', default='node_classification', type=str, help='name of task')
    # link_prediction / node_classification
    parser.add_argument('--dataset', '-d', default='am', type=str, help='name of datasets')
    parser.add_argument('--gpu', '-g', default='0', type=int, help='-1 means cpu')
    parser.add_argument('--use_best_config', action='store_true', help='will load utils.best_config')
    args = parser.parse_args()

    config_file = ["./openhgnn/config.ini"]
    config = Config(file_path=config_file, model=args.model, dataset=args.dataset, task=args.task, gpu=args.gpu)
    config.use_best_config = args.use_best_config
    OpenHGNN(args=config)
