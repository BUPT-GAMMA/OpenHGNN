import os
import yaml
import torch as th
import torch.nn as nn
import numpy as np

act_dict = {
    'relu': nn.ReLU(),
    'relu6': nn.ReLU6(),
    'sigmoid': nn.Sigmoid(),
    'lrelu': nn.LeakyReLU(negative_slope=0.5),
    'tanh': nn.Tanh(),
    'elu': nn.ELU(),
    'prelu': nn.PReLU(),
    'selu': nn.SELU(),
    'lrelu_01': nn.LeakyReLU(negative_slope=0.1),
    'lrelu_025': nn.LeakyReLU(negative_slope=0.25),
    'lrelu_05': nn.LeakyReLU(negative_slope=0.5),
}


def load_act(act):
    act = act_dict.get(act, None)
    if act is None:
        raise ValueError('No corresponding activation')
    return act


def read_config(args):
    # specify the model family
    if args.model == 'homo_GNN':
        args.model_family = 'homogenization'
    elif args.model == 'general_HGNN':
        assert args.subgraph_extraction in ['relation', 'metapath', 'mixed']
        args.model_family = args.subgraph_extraction
    else:
        raise ValueError('Wrong model name or subgraph_extraction')
    fileNamePath = os.path.split(os.path.realpath(__file__))[0]
    if args.key == 'gnn_type':
        yamlPath = os.path.join(fileNamePath, 'config/{}/{}.yaml'.format(args.configfile, args.times))
    else:
        yamlPath = os.path.join(fileNamePath, 'config/{}/{}/{}_{}.yaml'.format(args.configfile, args.key, args.gnn_type, args.times))
    print(yamlPath)
    with open(yamlPath, 'r', encoding='utf-8') as f:
        cont = f.read()
        config_dict = yaml.safe_load(cont)[args.task][args.dataset]

    if args.gpu == -1:
        device = th.device('cpu')
    elif args.gpu >= 0:
        if th.cuda.is_available():
            device = th.device('cuda', int(args.gpu))
        else:
            print("cuda is not available, please set 'gpu' -1")
    for key, value in config_dict.items():
        args.__setattr__(key, value)

    if args.key in ['has_bn', 'has_l2norm']:
        args.value = args.value == "True"
    elif args.key in ['stage_type', 'activation', 'macro_func', 'gnn_type', 'optimizer']:
        args.value = args.value
    else:
        args.value = float(args.value)
        if args.value % 1 == 0:
            args.value = int(args.value)

    args.__setattr__(args.key, args.value)
    args.__setattr__('device', device)
    args.__setattr__('metric', "f1")

    path = './space4hgnn/config/{}/{}/{}'.format(args.configfile, args.key, args.value)
    if not os.path.exists(path):
        os.makedirs(path)
    args.__setattr__('_checkpoint', path)
    args.__setattr__('HGB_results_path', None)
    args.activation = load_act(args.activation)
    return args

