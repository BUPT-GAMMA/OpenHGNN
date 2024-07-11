import torch as th
from tqdm import tqdm
from . import BaseFlow, register_flow
from ..models import build_model
from ..utils import EarlyStopping
from scipy.sparse import csr_matrix
from ..sampler.TransX_sampler import TransX_Sampler
import numpy as np
from torch.optim.lr_scheduler import ExponentialLR
from scipy.stats import rankdata
from openhgnn.tasks import build_task

def cal_performance(ranks):
    mrr = (1. / ranks).sum() / len(ranks)
    h_1 = sum(ranks<=1) * 1.0 / len(ranks)
    h_10 = sum(ranks<=10) * 1.0 / len(ranks)
    return mrr, h_1, h_10

def cal_ranks(scores, labels, filters):
    scores = scores - np.min(scores, axis=1, keepdims=True) + 1e-8
    full_rank = rankdata(-scores, method='average', axis=1)
    filter_scores = scores * filters
    filter_rank = rankdata(-filter_scores, method='min', axis=1)
    ranks = (full_rank - filter_rank + 1) * labels      # get the ranks of multiple answering entities simultaneously
    ranks = ranks[np.nonzero(ranks)]
    return list(ranks)


@register_flow("RedGNN_trainer")
class RedGNNTrainer(BaseFlow):
    """RedGNN flows."""

    def __init__(self, args):
        super(RedGNNTrainer, self).__init__(args)

        self.args = args
        self.model_name = args.model
        self.device = args.device
        self.task = build_task(args)
        self.batch_size = args.batch_size
        self.max_epoch = args.max_epoch

        self.loader = self.task.dataset
        self.n_ent = self.loader.n_ent
        self.n_ent_ind = self.loader.n_ent_ind

        self.n_train = self.loader.n_train
        self.n_valid = self.loader.n_valid
        self.n_test = self.loader.n_test

        self.model = build_model(self.model).build_model_from_args(self.args, self.task.dataset)
        self.model = self.model.to(self.device)

        self.optimizer = self.candidate_optimizer[args.optimizer](self.model.parameters(),
                                                                  lr=args.lr, weight_decay=args.weight_decay)
        self.scheduler = ExponentialLR(self.optimizer, args.decay_rate)
        self.stopper = EarlyStopping(args.patience, self._checkpoint)

    def train(self):
        for epoch in range(self.max_epoch):
            mrr, out_str = self.train_batch()
            if epoch % self.evaluate_interval == 0:
                self.logger.info("[Evaluation metric] " + out_str)  # out test result
                early_stop = self.stopper.loss_step(-mrr, self.model)  # less is better
                if early_stop:
                    self.logger.train_info(f'Early Stop!\tEpoch:{epoch:03d}.')
                    break


    def train_batch(self):
        epoch_loss = 0

        batch_size = self.batch_size
        n_batch = self.n_train // batch_size + (self.n_train % batch_size > 0)

        self.model.train()
        for i in range(n_batch):
            start = i * batch_size
            end = min(self.n_train, (i + 1) * batch_size)
            batch_idx = np.arange(start, end)
            triple = self.loader.get_batch(batch_idx)

            self.model.zero_grad()
            scores = self.model(triple[:, 0], triple[:, 1])
            pos_scores = scores[[th.arange(len(scores)).to(self.device), th.LongTensor(triple[:, 2]).to(self.device)]]
            max_n = th.max(scores, 1, keepdim=True)[0]
            loss = th.sum(- pos_scores + max_n + th.log(th.sum(th.exp(scores - max_n), 1)))
            loss.backward()
            self.optimizer.step()

            # avoid NaN
            for p in self.model.parameters():
                X = p.data.clone()
                flag = X != X
                X[flag] = np.random.random()
                p.data.copy_(X)
            epoch_loss += loss.item()
        self.scheduler.step()

        valid_mrr, out_str = self.evaluate()
        return valid_mrr, out_str


    def evaluate(self, ):
        batch_size = self.batch_size

        n_data = self.n_valid
        n_batch = n_data // batch_size + (n_data % batch_size > 0)
        ranking = []
        self.model.eval()
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

            filters = np.array(filters)
            ranks = cal_ranks(scores, objs, filters)
            ranking += ranks
        ranking = np.array(ranking)
        v_mrr, v_h1, v_h10 = cal_performance(ranking)

        n_data = self.n_test
        n_batch = n_data // batch_size + (n_data % batch_size > 0)
        ranking = []
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

            filters = np.array(filters)
            ranks = cal_ranks(scores, objs, filters)
            ranking += ranks
        ranking = np.array(ranking)
        t_mrr, t_h1, t_h10 = cal_performance(ranking)

        out_str = '[VALID] MRR:%.4f H@1:%.4f H@10:%.4f\t [TEST] MRR:%.4f H@1:%.4f H@10:%.4f' % (
        v_mrr, v_h1, v_h10, t_mrr, t_h1, t_h10)
        return v_mrr, out_str
