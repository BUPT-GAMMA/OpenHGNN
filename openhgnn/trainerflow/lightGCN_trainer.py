import random
import dgl
from tqdm import tqdm
import numpy as np
import torch as th
from time import time
from dgl.nn.functional import edge_softmax
from openhgnn.models import build_model
from dgl.dataloading import DataLoader, NeighborSampler, as_edge_prediction_sampler
import dgl.backend as F
from . import BaseFlow, register_flow
from ..tasks import build_task
from sklearn.metrics import f1_score, roc_auc_score
import torch.nn as nn
from torch import Tensor
from torch.nn.modules.loss import _Loss


@register_flow("lightGCN_trainer")
class lightGCNTrainer(BaseFlow):
    """Demo flows."""

    def __init__(self, args):
        super(lightGCNTrainer, self).__init__(args)

        self.l2_weight = args.weight_decay
        self.task = build_task(args)
        self.train_dataloader = None

        self.g, _, _ = self.task.get_split()

        self.user_num = self.task.dataset.n_user
        self.item_num = self.task.dataset.m_item
        self.g_dict = {"g": self.hg, "user_num": self.user_num, "item_num": self.item_num}
        self.f = nn.Sigmoid()

        self.model = build_model(self.model).build_model_from_args(self.args, self.g_dict).to(self.device)
        # self.optimizer = th.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        self.optimizer = th.optim.Adam(self.model.parameters(), lr=self.args.lr)

    def train(self):

        # for epoch in tqdm.tqdm(range(self.args.max_epoch)):
        for epoch in range(self.args.max_epoch):

            if epoch % 10 == 0:
                self.model.eval()
                u_batch_size = self.args.test_u_batch_size   # the batch size of users for testing
                dataset = self.task.dataset
                testDict = dataset.testDict    # all testdata

                max_K = self.args.topks       # update
                results = {'precision': np.zeros(1),
                           'recall': np.zeros(1),
                           'ndcg': np.zeros(1)}

                with th.no_grad():
                    users = list(testDict.keys())  # get test userID
                    try:
                        assert u_batch_size <= len(users) / 10
                    except AssertionError:

                        print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
                    users_list = []
                    rating_list = []
                    groundTrue_list = []
                    # auc_record = []
                    # ratings = []
                    total_batch = len(users) // u_batch_size + 1

                    for batch_users in self.minibatch(users, batch_size=u_batch_size):
                        allPos = dataset.getUserPosItems(batch_users)
                        groundTrue = [testDict[u] for u in batch_users]
                        batch_users_gpu = th.Tensor(batch_users).long()
                        batch_users_gpu = batch_users_gpu.to(self.device)

                        x = self.model.embedding.weight

                        all_users, all_items = self.model.computer()
                        users_emb = all_users[batch_users_gpu.long()]
                        items_emb = all_items
                        rating = self.f(th.matmul(users_emb, items_emb.t()))
                        # rating = th.matmul(users_emb, items_emb.t())

                        rating = rating.cpu()
                        exclude_index = []
                        exclude_items = []
                        for range_i, items in enumerate(allPos):
                            exclude_index.extend([range_i] * len(items))
                            exclude_items.extend(items)
                        rating[exclude_index, exclude_items] = -(1 << 10)

                        _, rating_K = th.topk(rating, k=max_K)  # mak_K = 20

                        rating = rating.cpu().numpy()

                        del rating
                        users_list.append(batch_users)
                        rating_list.append(rating_K.cpu())
                        groundTrue_list.append(groundTrue)
                    assert total_batch == len(users_list)
                    X = zip(rating_list, groundTrue_list)

                    pre_results = []
                    for x in X:
                        pre_results.append(self.test_one_batch(x))
                    scale = float(u_batch_size / len(users))
                    for result in pre_results:
                        results['recall'] += result['recall']
                        results['precision'] += result['precision']
                        results['ndcg'] += result['ndcg']
                    results['recall'] /= float(len(users))
                    results['precision'] /= float(len(users))
                    results['ndcg'] /= float(len(users))

                    print('[TEST]')
                    print(results)

            # for it, (input_nodes, positive_graph, negative_graph, blocks) in tqdm.tqdm(enumerate(self.train_dataloader)):
            self.model.train()

            S = self.UniformSample_original_python()

            users = th.Tensor(S[:, 0]).long()
            posItems = th.Tensor(S[:, 1]).long()
            negItems = th.Tensor(S[:, 2]).long()

            users = users.to(self.device)
            posItems = posItems.to(self.device)
            negItems = negItems.to(self.device)
            users, posItems, negItems = self.shuffle(users, posItems, negItems)
            total_batch = len(users) // self.args.batch_size + 1
            aver_loss = 0.

            for (batch_i,
                 (batch_users,
                  batch_pos,
                  batch_neg)) in enumerate(self.minibatch(users, posItems, negItems,batch_size=self.args.batch_size)):

                loss, reg_loss = self.model.bpr_loss(batch_users, batch_pos, batch_neg)
                reg_loss = reg_loss * self.l2_weight
                loss = loss + reg_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                cri = loss.cpu().item()
                aver_loss += cri

                # tqdm.set_postfix(f"Batch {batch_i}/{total_batch}")

            aver_loss = aver_loss / total_batch

            # print(epoch)
            # print(aver_loss)
            print(f'EPOCH[{epoch + 1}/{self.args.max_epoch}] loss:{aver_loss}')

    class BPRLoss(_Loss):
        r"""The Bayesian Personalized Ranking (BPR) loss.

        The BPR loss is a pairwise loss that encourages the prediction of an
        observed entry to be higher than its unobserved counterparts
        (see `here <https://arxiv.org/abs/2002.02126>`__).

        .. math::
            L_{\text{BPR}} = - \sum_{u=1}^{M} \sum_{i \in \mathcal{N}_u}
            \sum_{j \not\in \mathcal{N}_u} \ln \sigma(\hat{y}_{ui} - \hat{y}_{uj})
            + \lambda \vert\vert \textbf{x}^{(0)} \vert\vert^2

        where :math:`lambda` controls the :math:`L_2` regularization strength.
        We compute the mean BPR loss for simplicity.

        Args:
            lambda_reg (float, optional): The :math:`L_2` regularization strength
                (default: 0).
            **kwargs (optional): Additional arguments of the underlying
                :class:`torch.nn.modules.loss._Loss` class.
        """
        __constants__ = ['lambda_reg']
        lambda_reg: float

        def __init__(self, lambda_reg: float = 0, **kwargs):
            super().__init__(None, None, "sum", **kwargs)
            self.lambda_reg = lambda_reg

        def forward(self, positives: Tensor, negatives: Tensor,
                    parameters: Tensor = None) -> Tensor:
            log_prob = nn.functional.logsigmoid(positives - negatives).mean()
            # log_prob = - th.mean(th.nn.functional.softplus(negatives - positives))
            regularization = 0

            if self.lambda_reg != 0:
                regularization = self.lambda_reg * (1/2) * parameters.norm(2).pow(2) / float(parameters.shape[0])
            # print(-log_prob)
            #
            # print(regularization)
            return -log_prob + regularization

    def test_one_batch(self, X):
        sorted_items = X[0].numpy()
        groundTrue = X[1]
        r = self.getLabel(groundTrue, sorted_items)
        pre, recall, ndcg = [], [], []
        # for k in self.args.topks:
        k = self.args.topks
        ret = self.recall(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(self.ndcg(groundTrue, r, k))
        return {'recall': np.array(recall),
                'precision': np.array(pre),
                'ndcg': np.array(ndcg)}

    def minibatch(self, *tensors, **kwargs):

        batch_size = kwargs.get('batch_size', self.args.batch_size)

        if len(tensors) == 1:
            tensor = tensors[0]
            for i in range(0, len(tensor), batch_size):
                yield tensor[i:i + batch_size]
        else:
            for i in range(0, len(tensors[0]), batch_size):
                yield tuple(x[i:i + batch_size] for x in tensors)

    def getLabel(self, test_data, pred_data):
        r = []
        for i in range(len(test_data)):
            groundTrue = test_data[i]
            predictTopK = pred_data[i]
            pred = list(map(lambda x: x in groundTrue, predictTopK))
            pred = np.array(pred).astype("float")
            r.append(pred)

        return np.array(r).astype('float')

    def recall(self, test_data, r, k):
        """
            test_data should be a list? cause users may have different amount of pos items. shape (test_batch, k)
            pred_data : shape (test_batch, k) NOTE: pred_data should be pre-sorted
            k : top-k
        """
        right_pred = r[:, :k].sum(1)
        precis_n = k
        recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
        recall = np.sum(right_pred / recall_n)
        precis = np.sum(right_pred) / precis_n
        return {'recall': recall, 'precision': precis}

    def ndcg(self, test_data, r, k):
        """
        Normalized Discounted Cumulative Gain
        rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
        """
        assert len(r) == len(test_data)
        pred_data = r[:, :k]

        test_matrix = np.zeros((len(pred_data), k))
        for i, items in enumerate(test_data):
            length = k if k <= len(items) else len(items)
            test_matrix[i, :length] = 1
        max_r = test_matrix
        idcg = np.sum(max_r * 1. / np.log2(np.arange(2, k + 2)), axis=1)
        dcg = pred_data * (1. / np.log2(np.arange(2, k + 2)))
        dcg = np.sum(dcg, axis=1)
        idcg[idcg == 0.] = 1.
        ndcg = dcg / idcg
        ndcg[np.isnan(ndcg)] = 0.
        return np.sum(ndcg)

    def UniformSample_original_python(self):
        """
        the original impliment of BPR Sampling in LightGCN
        :return:
            np.array
        """
        total_start = time()
        # dataset: BasicDataset
        user_num = self.task.dataset.traindataSize
        users = np.random.randint(0, self.task.dataset.n_user, user_num)
        allPos = self.task.dataset.allPos
        S = []
        sample_time1 = 0.
        sample_time2 = 0.
        for i, user in enumerate(users):
            start = time()
            posForUser = allPos[user]
            if len(posForUser) == 0:
                continue
            sample_time2 += time() - start
            posindex = np.random.randint(0, len(posForUser))
            positem = posForUser[posindex]
            while True:
                negitem = np.random.randint(0, self.task.dataset.m_item)
                if negitem in posForUser:
                    continue
                else:
                    break
            S.append([user, positem, negitem])
            end = time()
            sample_time1 += end - start
        total = time() - total_start
        return np.array(S)

    def shuffle(self, *arrays, **kwargs):

        require_indices = kwargs.get('indices', False)

        if len(set(len(x) for x in arrays)) != 1:
            raise ValueError('All inputs to shuffle must have '
                             'the same length.')

        shuffle_indices = np.arange(len(arrays[0]))
        np.random.shuffle(shuffle_indices)

        if len(arrays) == 1:
            result = arrays[0][shuffle_indices]
        else:
            result = tuple(x[shuffle_indices] for x in arrays)

        if require_indices:
            return result, shuffle_indices
        else:
            return result