import yaml
from random import choice
import argparse
import os


hidden_dim = [8, 16, 32, 64, 128]
layers_pre_mp = [1, 2, 3]
layers_post_mp = [1, 2, 3]
layers_gnn = [1, 2, 3, 4, 5, 6]
stage_type = ['stack', 'skipsum', 'skipconcat']
activation = ['relu', 'lrelu', 'elu', 'sigmoid', 'tanh',]
has_bn = [True, False]
has_l2norm = [True, False]
mini_batch_flag = False
macro_func = ['attention', 'sum', 'mean', 'max']
dropout = [0.0, 0.3, 0.6]
lr = [0.1, 0.01, 0.001, 0.0001]
weight_decay = 0.0001
patience = 40
max_epoch = [100, 200, 400]
optimizer = ['Adam', 'SGD']
num_heads = [1, 2, 4, 8]
featn = [0, 1, 2]
featl = [0, 2]
loss_fn = ['distmult', 'dot-product']
gnn_type = ['gcnconv', 'gatconv', 'ginconv', 'sageconv']


def makeDict(gnn_type, type):
    dict = {
        'hidden_dim': choice(hidden_dim),
        'layers_pre_mp': choice(layers_pre_mp),
        'layers_post_mp': choice(layers_post_mp),
        'layers_gnn': choice(layers_gnn),
        'stage_type': choice(stage_type),
        'activation': choice(activation),
        'dropout': choice(dropout),
        'has_bn': choice(has_bn),
        'feat': choice(featn) if type == 'node' else choice(featl),
        'has_l2norm': choice(has_l2norm),
        'lr': choice(lr),
        'weight_decay': weight_decay,
        'patience': patience,
        'max_epoch': choice(max_epoch),
        'mini_batch_flag': mini_batch_flag,
        'macro_func': choice(macro_func),
        'optimizer': choice(optimizer),
        'num_heads': choice(num_heads),
        'loss_fn': choice(loss_fn) if type == 'link' else None,
        'gnn_type': gnn_type,
    }
    return dict

def generate(gnn_type, i, key, configfile):
    datasets_node = ['HGBn-ACM', 'HGBn-IMDB', 'HGBn-DBLP', 'HGBn-Freebase', 'HNE-PubMed']
    datasets_link = ['HGBl-amazon', 'HGBl-LastFM', 'HGBl-PubMed', 'HGBl-ACM', 'HGBl-DBLP', 'HGBl-IMDB']
    datasets_rec = ['yelp4HeGAN', 'DoubanMovie']
    dicts = {}
    dicts2 = {}
    dicts3 = {}

    for a in datasets_node:
        dict = makeDict(gnn_type, 'node')
        dicts[a] = dict
    for a in datasets_link:
        dict = makeDict(gnn_type, 'link')
        dicts2[a] = dict
    for a in datasets_rec:
        dict = makeDict(gnn_type, 'rec')
        dicts3[a] = dict

    aproject = {'node_classification': dicts,
                'link_prediction': dicts2,
                'recommendation': dicts3
    }

    fileNamePath = os.path.split(os.path.realpath(__file__))[0]
    path = fileNamePath + '/config/{}'.format(configfile)
    if not os.path.exists(path):
        os.makedirs(path)
    path = '{}/{}'.format(path, key)
    if not os.path.exists(path):
        os.makedirs(path)
    name = gnn_type + '_' + i + '.yaml'
    yamlPath = os.path.join(path, name)

    with open(yamlPath, 'w') as f:
        yaml.dump(aproject, f)
        print('Generate yaml file successfully!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gnn_type', '-a', default='gcnconv', type=str, help='gnn type')
    parser.add_argument('--times', '-s', default='1', type=str, help='times')
    parser.add_argument('--key', '-k', default='has_bn', type=str, help='attribute')
    parser.add_argument('--configfile', '-c', default='test', type=str, help='config file path')

    args = parser.parse_args()
    generate(args.gnn_type, args.times, args.key, args.configfile)
