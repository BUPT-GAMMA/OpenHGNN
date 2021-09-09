import os
import yaml
import torch as th
import torch.nn as nn
import numpy as np

act_dict = {
    'relu': nn.ReLU(),
    'selu': nn.SELU(),
    'prelu': nn.PReLU(),
    'elu': nn.ELU(),
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
    fileNamePath = os.path.split(os.path.realpath(__file__))[0]
    yamlPath = os.path.join(fileNamePath, 'config/{}_{}/{}_{}.yaml'.format(args.key, args.value, args.aggregation, args.times))
    print(yamlPath)
    with open(yamlPath, 'r', encoding='utf-8') as f:
        cont = f.read()
        config_dict = yaml.safe_load(cont)[args.task][args.dataset][args.model]

    if args.gpu == -1:
        device = th.device('cpu')
    elif args.gpu >= 0:
        if th.cuda.is_available():
            device = th.device('cuda', int(args.gpu))
        else:
            print("cuda is not available, please set 'gpu' -1")
    for key, value in config_dict.items():
        args.__setattr__(key, value)
    args.__setattr__('device', device)
    args.__setattr__('_checkpoint', './space4hgnn/')
    args.__setattr__('HGB_results_path', None)
    args.activation = load_act(args.activation)
    return args

