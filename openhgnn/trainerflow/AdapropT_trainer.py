import random
import os
import argparse
import torch
import numpy as np
from ..models import build_model
from . import BaseFlow, register_flow
from ..tasks import build_task
from ..utils.Adaprop_utils import *


@register_flow("AdapropT_trainer")
class AdapropTTrainer(BaseFlow):
    def __init__(self, args):
        opts = args
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.set_num_threads(8)
        args.dataset="AdapropT"
        dataset = opts.data_path
        dataset = dataset.split('/')
        if len(dataset[-1]) > 0:
            self.dataset_name = dataset[-1]
        else:
            self.dataset_name = dataset[-2]

        torch.cuda.set_device(args.gpu)
        print('==> gpu:', args.gpu)
        self.task=build_task(args)
        self.loader=self.task.dataloader
        self.model_name='AdapropT'
        args.n_ent = self.loader.n_ent
        args.n_rel = self.loader.n_rel
        self.args=args
        self.model=build_model(self.model_name).build_model_from_args(
            self.args,self.loader).model
    #     loader = DataLoader(opts)

    def train(self):
        print("111")
        opts=self.args
        # check all output paths
        checkPath('./results/')
        checkPath(f'./results/{self.dataset_name}/')
        checkPath(f'{self.loader.task_dir}/saveModel/')
        model = self.model
        opts.perf_file = f'results/{self.dataset_name}/{model.modelName}_perf.txt'
        print(f'==> perf_file: {self.args.perf_file}')

        config_str = '%.4f, %.4f, %.6f,  %d, %d, %d, %d, %.4f,%s\n' % (
        opts.lr, opts.decay_rate, opts.lamb, opts.hidden_dim, opts.attn_dim, opts.n_layer, opts.n_batch, opts.dropout,
        opts.act)
        print(config_str)
        # with open(opts.perf_file, 'a+') as f:
        #     f.write(config_str)

        # if self.args.weight != None:
        #     model.loadModel(self.args.weight)
        #     model._update()
        #     model.model.updateTopkNums(opts.n_node_topk)

        if opts.train:
            # training mode
            best_v_mrr = 0
            for epoch in range(opts.epoch):
                model.train_batch()
                # eval on val/test set
                if (epoch + 1) % self.args.eval_interval == 0:
                    result_dict, out_str = model.evaluate(eval_val=True, eval_test=True)
                    v_mrr, t_mrr = result_dict['v_mrr'], result_dict['t_mrr']
                    print(out_str)
                    with open(opts.perf_file, 'a+') as f:
                        f.write(out_str)
                    if v_mrr > best_v_mrr:
                        best_v_mrr = v_mrr
                        best_str = out_str
                        print(str(epoch) + '\t' + best_str)
                        BestMetricStr = f'ValMRR_{str(v_mrr)[:5]}_TestMRR_{str(t_mrr)[:5]}'
                        model.saveModelToFiles(BestMetricStr, deleteLastFile=False)

            # show the final result
            print(best_str)

        if opts.eval:
            # evaluate on test set with loaded weight to save time
            result_dict, out_str = model.evaluate(eval_val=False, eval_test=True, verbose=True)
            print(result_dict, '\n', out_str)

