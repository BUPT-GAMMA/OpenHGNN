# @Time   : 2021/1/28
# @Author : Tianyu Zhao
# @Email  : tyzhao@bupt.edu.cn


import argparse
from openhgnn.config import Config
from openhgnn.start import OpenHGNN


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default='HPN', type=str, help='name of models')
    parser.add_argument('--task', '-t', default='link_prediction', type=str, help='name of task')
    # link_prediction / node_classification
    parser.add_argument('--dataset', '-d', default='HGBl-PubMed', type=str, help='name of datasets')
    parser.add_argument('--gpu', '-g', default='0', type=int, help='-1 means cpu')
    parser.add_argument('--use_best_config', action='store_true', help='will load utils.best_config')
    parser.add_argument('--use_hpo', action='store_true', help='hyper-parameter optimization')
    parser.add_argument('--load_from_pretrained', action='store_true', help='load model from the checkpoint')
    args = parser.parse_args()

    config_file = ["./openhgnn/config.ini"]
    config = Config(file_path=config_file, model=args.model, dataset=args.dataset, task=args.task, gpu=args.gpu)
    config.use_best_config = args.use_best_config
    config.use_hpo = args.use_hpo
    config.load_from_pretrained = args.load_from_pretrained
    OpenHGNN(args=config)