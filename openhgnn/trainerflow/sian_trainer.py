import random
from time import time
import numpy as np
import heapq
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from . import BaseFlow,register_flow
from ..models import build_model
from ..tasks import build_task
import torch
import torch.nn as nn
import numpy as np
import time
import math
import torch.utils.data
from collections import defaultdict
from sklearn.metrics import  accuracy_score
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, precision_recall_curve
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

@register_flow("SIAN_trainer")
class SIAN_Trainer(BaseFlow):
    def __init__(self, args):
        super(SIAN_Trainer, self).__init__(args)
        self.args = args
        self.device = args.device
        self.task = build_task(args)
        self.logger=self.task.logger

        self.dataset = self.task.dataset
        self.user_profile = self.dataset.user_profile.to(self.device)
        self.item_profile = self.dataset.item_profile.to(self.device)

        self.train_data, self.test_data, self.val_data = self.dataset.data

        self.train_data_helper = DataUtil(self.train_data)
        self.test_data_helper = DataUtil(self.test_data)
        self.val_data_helper = DataUtil(self.val_data)

        self.train_loader = torch.utils.data.DataLoader(self.train_data_helper, batch_size=args.batch_size,
                                                   shuffle=True, collate_fn=self.collate_fn)
        self.valid_loader = torch.utils.data.DataLoader(self.val_data_helper, batch_size=args.test_batch_size,
                                                   shuffle=True, collate_fn=self.collate_fn)
        self.test_loader = torch.utils.data.DataLoader(self.test_data_helper, batch_size=args.test_batch_size,
                                                  shuffle=True, collate_fn=self.collate_fn)

        #build model
        self.model_name = self.args.model
        self.model = build_model(self.model_name).build_model_from_args(self.args, self.task.dataset)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=args.weight_decay)

    def collate_fn(self, batch):
        """
        sample = {
                'user': np.array([user]),
                'item': np.array([item]),
                'label': np.array([label]),
                'act': np.array(act, dtype=object),
            }
        :param batch:
        :return:
        """
        users = np.array([x['user'] for x in batch])
        items = np.array([x['item'] for x in batch])
        labels = np.array([x['label'] for x in batch])
        acts = np.array([item['act'] for item in batch], dtype=object)

        return {
            'user': torch.tensor(users, dtype=torch.long),
            'item': torch.tensor(items, dtype=torch.long),
            'label': torch.tensor(labels, dtype=torch.float),
            'act': acts,
        }

    def _mini_train_step(self):
        self.model.train()
        train_loss = 0.0
        valid_loss = 0.0
        test_loss = 0.0
        max_batches = 50
        criterion = nn.BCELoss()

        data_list = list(self.train_loader)

        selected_batches = random.sample(data_list, min(max_batches, len(data_list)))

        for data in tqdm(selected_batches):
            batch_user, batch_item, batch_label, batch_act = data['user'].long().to(self.device), \
                data['item'].long().to(self.device), \
                data['label'].float().to(self.device), \
                data['act']
            self.optimizer.zero_grad()
            output = self.model(batch_user, batch_item, batch_act)
            loss = criterion(output, batch_label)
            loss.backward(retain_graph=False)
            self.optimizer.step()
            train_loss += loss.data

        return train_loss, valid_loss, test_loss

    def train(self):
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)

        for epoch in range(self.args.epochs):
            t0 = time.time()
            train_loss, _, _ = self._mini_train_step()
            t1 = time.time()

            print('Epoch:%d, time: %.4fs, train loss:%.4f' %
                       (epoch, (t1 - t0), train_loss))

            # if epoch % 10 == 0:
            auc, f1, acc, pre, rec, test_loss = self.evaluate()
            t2 = time.time()
            print("evaluate time:%.4fs,"
                   "auc: %.4f, f1: %.4f, acc: %.4f, pre: %.4f, rec: %.4f" % (
                   (t2 - t1),  auc, f1, acc, pre, rec))

    def evaluate(self):
        """
        Evaluation for classification
        """
        loss = 0.0
        criterion = nn.BCELoss()
        self.model.eval()
        y_true, y_pred, y_score = [], [], []

        with torch.no_grad(): 
            for i, data in enumerate(self.test_loader, 0):
                batch_user, batch_item, batch_label, batch_act = data['user'].long().to(self.device), \
                    data['item'].long().to(self.device), \
                    data['label'].float().to(self.device), \
                    data['act']
                output = self.model(batch_user, batch_item, batch_act)
                loss += criterion(output, batch_label).item()

                y_true.extend(batch_label.cpu().tolist())
                y_score.extend(output.cpu().tolist())

                tmp_pred = (output > 0.5).int()
                y_pred.extend(tmp_pred.cpu().tolist())

        self.model.train()

        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
        auc = roc_auc_score(y_true, y_score)
        acc = accuracy_score(y_true, y_pred)

        return auc, f1, acc, prec, rec, loss

    def get_precision(self, rank_item, gt_item):
        p = 0
        for item in rank_item:
            if item in gt_item:
                p += 1
        return p * 1.0 / len(rank_item)

    def get_recall(self, rank_item, gt_item):
        r = 0
        for item in rank_item:
            if item in gt_item:
                r += 1
        return r * 1.0 / len(gt_item)

    def get_dcg(self, rank_item, gt_item):
        dcg = 0.0
        for i in range(len(rank_item)):
            item = rank_item[i]
            if item in gt_item:
                dcg += 1.0 / math.log(i + 2)
        return dcg

    def get_idcg(self, rank_item, gt_item):
        idcg = 0.0
        i = 0
        for item in rank_item:
            if item in gt_item:
                idcg += 1.0 / math.log(i + 2)
                i += 1
        return idcg

    def get_hit(self, rank_item, gt_item):
        for item in rank_item:
            if item in gt_item:
                return 1
        return 0

    def get_ndcg(self, rank_item, gt_item):
        dcg = self.get_dcg(rank_item, gt_item)
        idcg = self.get_idcg(rank_item, gt_item)
        if idcg == 0:
            return 0
        return dcg / idcg

    def eval_ranking(self, users, items, pos_prediction, neg_prediction):
        pos_u = list(np.array(users))
        pos_i = list(map(lambda x: np.array(x[0]).tolist(), items))
        neg_i = list(map(lambda x: np.array(x[1:]).tolist(), items))

        precision = []
        recall = []
        hit = []
        ndcg = []
        ground_truth_u_items = defaultdict(list)
        for idx, u in enumerate(pos_u):
            ground_truth_u_items[u].append(pos_i[idx])

        for idx, u in enumerate(pos_u):
            item_score = {}
            real_score = pos_prediction[idx]
            pred_scores = neg_prediction[idx]

            item_score[pos_i[idx]] = real_score
            for jdx, n_i in enumerate(neg_i[idx]):
                item_score[n_i] = pred_scores[jdx]

            rank_item = heapq.nlargest(self.args.top_k, item_score, key=item_score.get)

            gt_item = ground_truth_u_items[u]
            precision.append(self.get_precision(rank_item, gt_item))
            recall.append(self.get_recall(rank_item, gt_item))
            hit.append(self.get_hit(rank_item, gt_item))
            ndcg.append(self.get_ndcg(rank_item, gt_item))
        return np.array(precision).mean(), np.array(recall).mean(), np.array(hit).mean(), np.array(ndcg).mean()



class DataUtil(Dataset):
    def __init__(self, data):
        self.idx2user = np.array(data[0])
        self.idx2item = np.array(data[1])
        self.idx2label = np.array(data[2])
        self.idx2act = np.array(data[3])

    def __len__(self):
        return len(self.idx2label)

    def __getitem__(self, idx):
        user = self.idx2user[idx]
        item = self.idx2item[idx]
        label = self.idx2label[idx]
        act = self.idx2act[idx]

        sample = {
            'user': user,
            'item': item,
            'label': label,
            'act': act,
        }
        return sample
