import torch as th
import yaml
import os


def read_config(args):
    fileNamePath = os.path.split(os.path.realpath(__file__))[0]
    yamlPath = os.path.join(fileNamePath, 'config.yaml')
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

    args.__setattr__('device', device)
    args.__setattr__('_checkpoint', './space4hgnn/homo_models/')
    for key, value in config_dict.items():
        args.__setattr__(key, value)
    return args
