import torch
import torch.nn as nn
import numpy as np
import time
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from ..layers.AdapropI import GNNModel
from ..utils.AdapropI_utils import *
from tqdm import tqdm
from . import BaseModel, register_model

@register_model('AdapropI')
class AdapropI(BaseModel):
    @classmethod
    def build_model_from_args(cls, config,loader):
        return cls(config,loader)

    def __init__(self, config,loader):
        super().__init__()
        self.model = AdapropI_Base(config,loader)

    def forward(self, *args):
        return self.model(*args)

    def extra_loss(self):
        pass
class AdapropI_Base(object):
    def __init__(self, args, loader):
        self.model = GNNModel(args, loader)
        self.model.cuda()
        self.loader = loader
        self.n_ent = loader.n_ent
        self.n_ent_ind = loader.n_ent_ind
        self.n_batch = args.n_batch
        self.n_train = loader.n_train
        self.n_valid = loader.n_valid
        self.n_test = loader.n_test
        self.n_layer = args.n_layer
        self.optimizer = Adam(self.model.parameters(), lr=args.lr, weight_decay=args.lamb)
        self.scheduler = ExponentialLR(self.optimizer, args.decay_rate)
        self.smooth = 1e-5
        self.params = args

    def train_batch(self, ):
        epoch_loss = 0
        i = 0
        batch_size = self.n_batch
        n_batch = self.n_train // batch_size + (self.n_train % batch_size > 0)
        self.model.train()
        self.time_1 = 0
        self.time_2 = 0

        for i in range(n_batch):
            start = i * batch_size
            end = min(self.n_train, (i + 1) * batch_size)
            batch_idx = np.arange(start, end)
            triple = self.loader.get_batch(batch_idx)

            self.model.zero_grad()
            scores = self.model(triple[:, 0], triple[:, 1])
            pos_scores = scores[[torch.arange(len(scores)).cuda(), torch.LongTensor(triple[:, 2]).cuda()]]
            self.time_1 += self.model.time_1
            self.time_2 += self.model.time_2
            t_2 = time.time()

            max_n = torch.max(scores, 1, keepdim=True)[0]
            loss = torch.sum(- pos_scores + max_n + torch.log(torch.sum(torch.exp(scores - max_n), 1)))

            loss.backward()
            self.optimizer.step()
            self.time_2 += time.time() - t_2

            for p in self.model.parameters():
                X = p.data.clone()
                flag = X != X
                X[flag] = np.random.random()
                p.data.copy_(X)
            epoch_loss += loss.item()

        self.loader.shuffle_train()
        self.scheduler.step()
        valid_mrr, test_mrr, out_str = self.evaluate()
        return valid_mrr, test_mrr, out_str

    def evaluate(self, ):
        batch_size = self.n_batch
        n_data = self.n_valid
        n_batch = n_data // batch_size + (n_data % batch_size > 0)
        ranking = []
        masks = []
        self.model.eval()
        time_3 = time.time()
        for i in range(n_batch):
            start = i * batch_size
            end = min(n_data, (i + 1) * batch_size)
            batch_idx = np.arange(start, end)
            subs, rels, objs = self.loader.get_batch(batch_idx, data='valid')
            scores = self.model(subs, rels).data.cpu().numpy()
            filters = []
            for i in range(len(subs)):
                filt = self.loader.val_filters[(subs[i], rels[i])]
                filt_1hot = np.zeros((self.n_ent,))
                filt_1hot[np.array(filt)] = 1
                filters.append(filt_1hot)

                masks += [self.n_ent - len(filt)] * int(objs[i].sum())

            filters = np.array(filters)
            ranks = cal_ranks(scores, objs, filters)
            ranking += ranks

        ranking = np.array(ranking)
        v_mrr, v_mr, v_h1, v_h3, v_h10, v_h1050 = cal_performance(ranking, masks)

        n_data = self.n_test
        n_batch = n_data // batch_size + (n_data % batch_size > 0)
        ranking = []
        masks = []
        self.model.eval()
        for i in range(n_batch):
            start = i * batch_size
            end = min(n_data, (i + 1) * batch_size)
            batch_idx = np.arange(start, end)
            subs, rels, objs = self.loader.get_batch(batch_idx, data='test')
            scores = self.model(subs, rels, 'inductive').data.cpu().numpy()
            filters = []
            for i in range(len(subs)):
                filt = self.loader.tst_filters[(subs[i], rels[i])]
                filt_1hot = np.zeros((self.n_ent_ind,))
                filt_1hot[np.array(filt)] = 1
                filters.append(filt_1hot)
                masks += [self.n_ent_ind - len(filt)] * int(objs[i].sum())

            filters = np.array(filters)
            ranks = cal_ranks(scores, objs, filters)
            ranking += ranks
        ranking = np.array(ranking)
        t_mrr, t_mr, t_h1, t_h3, t_h10, t_h1050 = cal_performance(ranking, masks)
        time_3 = time.time() - time_3

        out_str = '%.4f %.4f %.4f\t%.4f %.1f %.4f %.4f %.4f %.4f\t\t%.4f %.1f %.4f %.4f %.4f %.4f\n' % (
        self.time_1, self.time_2, time_3, v_mrr, v_mr, v_h1, v_h3, v_h10, v_h1050, t_mrr, t_mr, t_h1, t_h3, t_h10,
        t_h1050)
        return v_h10, t_h10, out_str
