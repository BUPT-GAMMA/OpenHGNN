# @Time   : 2021/1/28
# @Author : Tianyu Zhao
# @Email  : tyzhao@bupt.edu.cn

import argparse


from openhgnn.experiment import Experiment
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default='GTN', type=str, help='name of models')
    parser.add_argument('--task', '-t', default='node_classification', type=str, help='name of task')
    # link_prediction / node_classification
    parser.add_argument('--dataset', '-d', default='acm4GTN', type=str, help='name of datasets')
    parser.add_argument('--gpu', '-g', default='0', type=int, help='-1 means cpu')
    parser.add_argument('--use_distributed', action='store_true', help='will use distributed training')
    parser.add_argument('--use_best_config', action='store_true', help='will load utils.best_config')
    parser.add_argument('--load_from_pretrained', action='store_true', help='load model from the checkpoint')
    parser.add_argument('--use_database', action='store_true',help = 'use database')
    parser.add_argument('--mini_batch_flag', action='store_true', help='will train in mini_batch mode')
    parser.add_argument('--graphbolt',action='store_true',help = 'use graphbolt to access dataset')
    args = parser.parse_args()

    experiment = Experiment(model=args.model, dataset=args.dataset, task=args.task, gpu=args.gpu,
                            use_best_config=args.use_best_config, load_from_pretrained=args.load_from_pretrained,
                            mini_batch_flag=args.mini_batch_flag, use_distributed = args.use_distributed,
                            graphbolt = args.graphbolt)

    experiment.run()