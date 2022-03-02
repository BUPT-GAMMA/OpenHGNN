import os
import time
import argparse
import numpy as np
from pandas import DataFrame
from space4hgnn.utils import read_config
from openhgnn.utils import set_random_seed, Logger
from openhgnn.trainerflow import build_flow


def Space4HGNN(args):
    metric_list = []
    epoches = []
    start = time.time()
    args.logger = Logger(args)
    for i in range(args.repeat):
        args.seed = i
        set_random_seed(args.seed)
        path = './space4hgnn/prediction/txt/{}/{}_{}/{}_{}_{}'.format(args.predictfile, args.key, args.value, args.model_family, args.gnn_type, args.times)
        if not os.path.exists(path):
            os.makedirs(path)
        args.HGB_results_path = '{}/{}_{}.txt'.format(path, args.dataset[5:], str(i+1))
        print(args)
        flow = build_flow(args, args.task)
        result = flow.train()
        metric = result['metric']
        epoch = result['epoch']
        metric_list.append(metric)
        epoches.append(epoch)
    out_dict = {}
    for metrics in metric_list:
        for mode, metric in metrics.items():
            for m, score in metric.items():
                if out_dict.get(f"{mode}_{m}", None) is None:
                    out_dict[f"{mode}_{m}"] = []
                out_dict[f"{mode}_{m}"].append(score)
            
    end = time.time()
    mean_dict = {k + 'mean': np.mean(v) for k, v in out_dict.items()}
    std_dict = {k + 'std': np.std(v) for k, v in out_dict.items()}
    para = sum(p.numel() for p in flow.model.parameters())
    result = {
        'key': [args.key],
        'value': [args.value],
        'dataset': [args.dataset],
        'model_family': [args.model_family],
        'gnn_type': [args.gnn_type],
        'times': [args.times],
        'hidden_dim': [args.hidden_dim],
        'layers_pre_mp': [args.layers_pre_mp],
        'layers_post_mp': [args.layers_post_mp],
        'layers_gnn': [args.layers_gnn],
        'stage_type': [args.stage_type],
        'activation': [args.activation],
        'has_bn': [args.has_bn],
        'has_l2norm': [args.has_l2norm],
        'mini_batch_flag': [args.mini_batch_flag],
        'macro_func': [args.macro_func],
        'dropout': [args.dropout],
        'lr': [args.lr],
        'num_heads': [args.num_heads],
        'weight_decay': [args.weight_decay],
        'patience': [args.patience],
        'max_epoch': [args.max_epoch],
        'feat': [args.feat],
        'optimizer': [args.optimizer],
        'loss_fn': [args.loss_fn],
        'parameter': [para],
        'epoch': [np.mean(epoches)],
        'time': [end - start],
    }
    result.update(mean_dict)
    result.update(std_dict)
    df = DataFrame(result)
    print(df)
    path = 'space4hgnn/prediction/excel/{}/{}_{}'.format(args.predictfile, args.key, args.value)
    if not os.path.exists(path):
        os.makedirs(path)
    df.to_csv('{}/{}_{}_{}_{}.csv'.format(path, args.model_family, args.gnn_type, args.times, args.dataset))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default='homo_GNN', type=str, help='name of models')
    parser.add_argument('--subgraph_extraction', '-u', default='metapath', type=str, help='subgraph_extraction of models')
    parser.add_argument('--task', '-t', default='link_prediction', type=str, help='name of task')
    parser.add_argument('--dataset', '-d', default='HGBl-PubMed', type=str, help='name of datasets')
    parser.add_argument('--gpu', '-g', default='0', type=int, help='-1 means cpu')
    parser.add_argument('--repeat', '-r', default='5', type=int, help='-1 means cpu')
    parser.add_argument('--gnn_type', '-a', default='gcnconv', type=str, help='aggregation type')
    parser.add_argument('--times', '-s', default=1, type=int, help='which yaml file')
    parser.add_argument('--key', '-k', default='has_bn', type=str, help='attribute')
    parser.add_argument('--value', '-v', default='True', type=str, help='value')
    parser.add_argument('--configfile', '-c', default='config', type=str, help='The file path to load the configuration.')
    parser.add_argument('--predictfile', '-p', default='predict', type=str, help='The file path to store predict files.')
    args = parser.parse_args()

    args = read_config(args)

    Space4HGNN(args=args)
