import random
import os
import argparse
import torch
import numpy as np
from ..models import build_model
from . import BaseFlow, register_flow
from ..tasks import build_task
from ..utils.AdapropI_utils import *
@register_flow("AdapropI_trainer")
class AdapropITrainer(BaseFlow):
    def __init__(self, args):
        class Options(object):
            pass

        dataset = args.data_path
        dataset = dataset.split('/')
        if len(dataset[-1]) > 0:
            dataset = dataset[-1]
        else:
            dataset = dataset[-2]
        # args.dataset_name=dataset
        args.dataset_name=dataset
        self.model_name='AdapropI'
        self.args = Options
        self.args=args
        self.args.hidden_dim = 64
        self.args.init_dim = 10
        self.args.attn_dim = 5
        self.args.n_layer = 3
        self.args.n_batch = 50
        self.args.lr = 0.001
        self.args.decay_rate = 0.999
        self.args.perf_file = './results.txt'
        self.args.task_dir=args.data_path
        gpu = args.device
        torch.cuda.set_device(gpu)
        print('==> selected GPU id: ', gpu)
        args.n_batch=self.args.n_batch
        self.task = build_task(self.args)
        self.loader=self.task.dataloader
        loader=self.loader
        # loader = DataLoader(args.data_path, n_batch=self.args.n_batch)
        self.args.n_ent = loader.n_ent
        self.args.n_rel = loader.n_rel



        params = {}
        if 'fb237_v1' in args.data_path:
            params['lr'], params['decay_rate'], params["lamb"], params['hidden_dim'], params['init_dim'], params[
                'attn_dim'], params['n_layer'], params['n_batch'], params['dropout'], params['act'], params['topk'], \
            params['increase'] = 0.0005, 0.9968, 0.000081, 32, 32, 5, 3, 100, 0.4137, 'relu', 100, True
        if 'fb237_v2' in args.data_path:
            params['lr'], params['decay_rate'], params["lamb"], params['hidden_dim'], params['init_dim'], params[
                'attn_dim'], params['n_layer'], params['n_batch'], params['dropout'], params['act'], params['topk'], \
            params['increase'] = 0.0087, 0.9937, 0.000025, 16, 16, 5, 5, 20, 0.3265, 'relu', 200, True

        if 'fb237_v3' in args.data_path:
            params['lr'], params['decay_rate'], params["lamb"], params['hidden_dim'], params['init_dim'], params[
                'attn_dim'], params['n_layer'], params['n_batch'], params['dropout'], params['act'], params['topk'], \
            params['increase'] = 0.0079, 0.9934, 0.000187, 48, 48, 5, 7, 20, 0.4632, 'relu', 200, True

        if 'fb237_v4' in args.data_path:
            params['lr'], params['decay_rate'], params["lamb"], params['hidden_dim'], params['init_dim'], params[
                'attn_dim'], params['n_layer'], params['n_batch'], params['dropout'], params['act'], params['topk'], \
            params['increase'] = 0.0010, 0.9997, 0.000186, 16, 16, 5, 7, 50, 0.4793, 'relu', 500, True

        print(params)

        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        self.args.lr = params['lr']
        self.args.lamb = params["lamb"]
        self.args.decay_rate = params['decay_rate']
        self.args.hidden_dim = params['hidden_dim']
        self.args.init_dim = params['hidden_dim']
        self.args.attn_dim = params['attn_dim']
        self.args.dropout = params['dropout']
        self.args.act = params['act']
        self.args.n_layer = params['n_layer']
        self.args.n_batch = params['n_batch']
        self.args.topk = params['topk']
        self.args.increase = params['increase']

        config_str = '%.4f, %.4f, %.6f,  %d, %d, %d, %d, %d, %.4f, %s  %d, %s\n' % (
        self.args.lr, self.args.decay_rate, self.args.lamb, self.args.hidden_dim, self.args.init_dim, self.args.attn_dim, self.args.n_layer,
        self.args.n_batch, self.args.dropout, self.args.act, self.args.topk, str(self.args.increase))
        print(args.data_path)
        print(config_str)
        try:
            self.model = build_model(self.model_name).build_model_from_args(
                self.args, self.loader).model
            model = self.model
            best_mrr = 0
            best_tmrr = 0
            early_stop = 0
            for epoch in range(30):
                print("epoch:"+str(epoch))
                mrr, t_mrr, out_str = model.train_batch()
                print(mrr, t_mrr, out_str)
                if mrr > best_mrr:
                    best_mrr = mrr
                    best_tmrr = t_mrr
                    best_str = out_str
                    early_stop = 0
                else:
                    early_stop += 1

            with open(self.args.perf_file, 'a') as f:
                f.write(args.data_path + '\n')
                f.write(config_str)
                f.write(best_str + '\n')
                print('\n\n')

        except RuntimeError:
            best_tmrr = 0

        print(
            'self.time_1, self.time_2, time_3, v_mrr, v_mr, v_h1, v_h3, v_h10, v_h1050, t_mrr, t_mr, t_h1, t_h3, t_h10, t_h1050')
        print(best_str)
    def train(self):
        print(2)