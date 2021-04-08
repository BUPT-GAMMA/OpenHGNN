# @Time   : 2021/1/28
# @Author : Tianyu Zhao
# @Email  : tyzhao@bupt.edu.cn


import argparse
from openhgnn.config import Config
from openhgnn.start import OpenHGNN


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default='HetGNN', type=str, help='name of models')
    parser.add_argument('--dataset', '-d', default='academic', type=str, help='name of datasets')
    parser.add_argument('--gpu', '-g', default='0', type=int, help='-1 means cpu')
    args = parser.parse_args()

    config_file = ["./openhgnn/config.ini"]
    config = Config(file_path=config_file, model=args.model, dataset=args.dataset, gpu=args.gpu)
    OpenHGNN(args=config)
