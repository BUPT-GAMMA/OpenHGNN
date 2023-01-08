from abc import abstractmethod
import dgl
import torch as th
from torch import nn
from tqdm import tqdm
import torch
import torch.nn.functional as F
from openhgnn.trainerflow import BaseFlow, register_flow
from openhgnn.models import build_model
from openhgnn.utils import extract_embed, EarlyStopping, get_nodes_dict, add_reverse_edges
from torch.utils.data import DataLoader
from openhgnn.tasks import build_task
import os
from sklearn import metrics
import numpy as np


@register_flow("DHNE_trainer")
class DHNE_trainer(BaseFlow):
    """
    DHNE trainer flow.
    Supported Model: DHNE
    Supported Datase: drug, GPS, MovieLens, wordnet
    The trainerflow supports hypergraph embedding task.
    For more details, please refer to the original paper: https://arxiv.org/abs/1711.10146
    """
    def __init__(self, args):
        super(DHNE_trainer, self).__init__(args)
        self.hg = None

        self.args = args
        self.logger = self.args.logger
        self.model_name = args.model_name
        self.model = args.model
        self.task = build_task(args)
        self.train_dataset = self.task.train_data
        self.val_dataset = self.task.val_data
        self.device = self.args.device
        self.args.nums_type = self.train_dataset.nums_type
        self.model = build_model(self.model).build_model_from_args(
            self.args).to(self.device)
        self.optimizer = torch.optim.RMSprop(
            self.model.parameters(), lr=args.lr)

        self.train_dataloader = DataLoader(self.train_dataset,
                                           batch_size=self.args.batch_size,
                                           shuffle=True,
                                           num_workers=4,
                                           collate_fn=self.collate_fn)
        self.val_dataloader = DataLoader(self.val_dataset,
                                         batch_size=self.args.batch_size,
                                         shuffle=True,
                                         num_workers=4,
                                         collate_fn=self.collate_fn)

    def collate_fn(self, batches):
        encode1, encode2, encode3 = [], [], []
        labels = []
        for encodes, label in batches:
            encode1.append(torch.from_numpy(encodes[0]))
            encode2.append(torch.from_numpy(encodes[1]))
            encode3.append(torch.from_numpy(encodes[2]))
            labels.append(torch.from_numpy(label))

        return [torch.concat(encode1).type(torch.float32), torch.concat(encode2).type(torch.float32), torch.concat(encode3).type(torch.float32)], torch.concat(labels)

    def preprocess(self):

        pass

    def train(self):
        total_metric = {}
        max_auc = 0
        max_auc_epoch = 0
        max_auc_epoch_loss = 0
        for epoch in tqdm(range(self.max_epoch)):
            loss = self._mini_train_step()
            if epoch % self.evaluate_interval == 0:
                val_metric = self._test_step()
                self.logger.train_info(
                    f"Epoch: {epoch:03d}, train loss: {loss:.4f}. " +
                    "encode_0:{encode_0},encode_1:{encode_1},encode_2:{encode_2},label:{label},auc_micro:{auc_micro}".format(**val_metric))
                total_metric[epoch] = val_metric
                if val_metric['auc_micro'] > max_auc:
                    max_auc = val_metric['auc_micro']
                    max_auc_epoch = epoch
                    max_auc_epoch_loss = loss
        print("[Best Info] the best training results:")
        print(f"Epoch: {max_auc_epoch:03d}, train loss: {max_auc_epoch_loss:.4f}. " +
              "encode_0:{encode_0},encode_1:{encode_1},encode_2:{encode_2},label:{label},auc_micro:{auc_micro}".format(
                  **total_metric[max_auc_epoch]))
        save_path = os.path.dirname(os.path.abspath('__file__'))+'/openhgnn/output/'+self.model_name
        th.save(self.model.state_dict(), save_path+'/DHNE.pth')

    def _full_train_setp(self):
        pass

    def _mini_train_step(self,):
        self.model.train()
        all_loss = 0
        loader_tqdm = tqdm(self.train_dataloader, ncols=120)
        for blocks, target in loader_tqdm:
            blocks = [b.to(self.device) for b in blocks]
            target = target.to(self.device)
            logits = self.model(blocks)
            loss = self.loss_calculation(blocks+[target], logits)
            all_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return all_loss

    def sparse_autoencoder_error(self, y_true, y_pred):
        return torch.mean(torch.square(torch.sign(y_true)*(y_true-y_pred)), axis=-1)

    def loss_calculation(self, targets, preds):
        loss = 0
        for i in range(3):
            target = targets[i]
            pred = preds[i]
            loss += self.sparse_autoencoder_error(target, pred)
        label_target = targets[-1]
        label_target = label_target.reshape(-1, 1)
        label_pred = preds[-1]
        loss = loss.mean()
        loss += F.binary_cross_entropy_with_logits(label_pred, label_target)
        return loss

    def _test_step(self):
        self.model.eval()
        metric = {"encode_0": 0, "encode_1": 0, "encode_2": 0, "label": 0}
        with th.no_grad():
            loader_tqdm = tqdm(self.val_dataloader, ncols=120)
            num = 0
            y_true = []
            y_score = []
            for blocks, target in loader_tqdm:
                blocks = [b.to(self.device) for b in blocks]
                targets = target.to(self.device)
                logits = self.model(blocks)
                for i in range(3):
                    pred = logits[i]
                    target = blocks[i]
                    metric["encode_%d" % i] += F.mse_loss(pred, target)
                label_logit = torch.sigmoid(logits[-1])
                label_pred = (label_logit > 0.5).int()
                label_pred = label_pred.reshape(-1)
                metric["label"] += (label_pred == targets).float().mean()
                y_true.append(targets.cpu().numpy())
                y_score.append(label_logit.cpu().numpy())
                num += 1
        metric = {k: v.cpu().numpy() / num for k, v in metric.items()}
        y_true = np.hstack(y_true)
        y_score = np.vstack(y_score).reshape(-1)
        auc_micro = metrics.roc_auc_score(y_true, y_score, average='micro')
        metric['auc_micro'] = auc_micro
        return metric

    def save_checkpoint(self):
        if self._checkpoint and hasattr(self.model, "_parameters()"):
            torch.save(self.model.state_dict(), self._checkpoint)
