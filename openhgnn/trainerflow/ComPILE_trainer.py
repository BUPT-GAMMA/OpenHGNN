import statistics
import timeit
import os
import logging
import pdb
import numpy as np
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from openhgnn.models import build_model
from . import BaseFlow, register_flow
from ..tasks import build_task

from sklearn import metrics
from ..utils.Grail_utils import collate_dgl2,move_batch_to_device_dgl
@register_flow('ComPILE_trainer')
class ComPILETrainer(BaseFlow):
    def __init__(self,args):
        super(ComPILETrainer, self).__init__(args)
        #self.train_hg = self.task.get_train()
        self.trainset = self.task.dataset.train
        self.valid = self.task.dataset.valid
        self.args.num_rels = self.trainset.num_rels
        self.args.aug_num_rels = self.trainset.aug_num_rels
        self.args.inp_dim = self.trainset.n_feat_dim

        self.args.collate_fn = collate_dgl2
        self.args.move_batch_to_device = move_batch_to_device_dgl

        # Log the max label value to save it in the model. This will be used to cap the labels generated on test set.
        self.args.max_label_value = self.trainset.max_n_label

        self.model = build_model(self.model).build_model_from_args(self.args, self.task.dataset.relation2id).to(
            self.device)
        self.updates_counter = 0
        model_params = list(self.model.parameters())
        #logging.info('Total number of parameters: %d' % sum(map(lambda x: x.numel(), model_params)))

        if self.args.optimizer == "SGD":
            self.optimizer = optim.SGD(model_params, lr=self.args.lr, momentum=self.args.momentum,
                                       weight_decay=self.args.l2)
        if self.args.optimizer == "Adam":
            self.optimizer = optim.Adam(model_params, lr=self.args.lr, weight_decay=self.args.l2)

        self.criterion = nn.MarginRankingLoss(self.args.margin, reduction='sum')

        self.reset_training_state()
        #graph_classifier = initialize_model(params, dgl_model, params.load_model)

        self.logger.info(f"Device: {args.device}")
        self.logger.info(
            f"Input dim : {args.inp_dim}, # Relations : {args.num_rels}, # Augmented relations : {args.aug_num_rels}")

        self.args.save_path = os.path.dirname(os.path.abspath('__file__')) + '/openhgnn/output/' + self.model_name
        self.valid_evaluator = Evaluator(self.args, self.model, self.valid)
        #self.save_path = os.path.dirname(os.path.abspath('__file__')) + '/openhgnn/output/' + self.model_name

        self.logger.info('Starting training with full batch...')




    def reset_training_state(self):
        self.best_metric = 0
        self.last_metric = 0
        self.not_improved_count = 0

    def train_epoch(self):
        total_loss = 0
        all_preds = []
        all_labels = []
        all_scores = []

        dataloader = DataLoader(self.trainset, batch_size=self.args.batch_size, shuffle=True,
                                num_workers=self.args.num_workers, collate_fn=self.args.collate_fn)
        #      dataloader = DataLoader(self.train_data, batch_size=self.params.batch_size, shuffle=True, num_workers=self.params.num_workers)
        self.model.train()
        model_params = list(self.model.parameters())
        torch.multiprocessing.set_sharing_strategy('file_system')
        for b_idx, batch in enumerate(dataloader):
            (graphs_pos, r_labels_pos), g_labels_pos, (graph_neg, r_labels_neg), g_labels_neg = batch
            #(graphs_pos, r_labels_pos), g_labels_pos, (graph_neg, r_labels_neg), g_labels_neg = self.args.move_batch_to_device(batch, self.args.device)

            g_labels_pos = torch.LongTensor(g_labels_pos).to(device=self.args.device)
            r_labels_pos = torch.LongTensor(r_labels_pos).to(device=self.args.device)

            g_labels_neg = torch.LongTensor(g_labels_neg).to(device=self.args.device)
            r_labels_neg = torch.LongTensor(r_labels_neg).to(device=self.args.device)

            self.model.train()
            # data_pos, targets_pos, data_neg, targets_neg = self.params.move_batch_to_device(batch, self.params.device)
            self.optimizer.zero_grad()
            # print('batch size ', len(targets_pos), '     ', len(targets_neg))
            # print('r label pos ', len(data_pos[1]), '   r label neg  ', len(data_neg[1]))
            score_pos = self.model(graphs_pos)
            score_neg = self.model(graph_neg)
            loss = self.criterion(score_pos.squeeze(), score_neg.view(len(score_pos), -1).mean(dim=1),
                                  torch.Tensor([1]).to(device=self.args.device))
            # print(score_pos, score_neg, loss)
            loss.backward()
            self.optimizer.step()
            self.updates_counter += 1

            with torch.no_grad():
                # print(score_pos.shape, score_neg.shape)
                #  print(score_pos)
                all_scores += score_pos.squeeze(1).detach().cpu().tolist() + score_neg.squeeze(
                    1).detach().cpu().tolist()
                all_labels += g_labels_pos.tolist() + g_labels_neg.tolist()
                total_loss += loss

            if self.valid_evaluator and self.args.eval_every_iter and self.updates_counter % self.args.eval_every_iter == 0:
                tic = time.time()
                result = self.valid_evaluator.eval()
                logging.info('\nPerformance:' + str(result) + 'in ' + str(time.time() - tic))

                if result['auc'] >= self.best_metric:
                    self.save_classifier()
                    self.best_metric = result['auc']
                    self.not_improved_count = 0

                else:
                    self.not_improved_count += 1
                    if self.not_improved_count > self.args.early_stop:
                        logging.info(
                            f"Validation performance didn\'t improve for {self.args.early_stop} epochs. Training stops.")
                        break
                self.last_metric = result['auc']

        auc = metrics.roc_auc_score(all_labels, all_scores)
        auc_pr = metrics.average_precision_score(all_labels, all_scores)

        weight_norm = sum(map(lambda x: torch.norm(x), model_params))

        return total_loss, auc, auc_pr, weight_norm

    def train(self):
        self.reset_training_state()

        for epoch in range(1, self.args.num_epochs + 1):
            time_start = time.time()
            loss, auc, auc_pr, weight_norm = self.train_epoch()
            time_elapsed = time.time() - time_start
            self.logger.info(
                f'Epoch {epoch} with loss: {loss}, training auc: {auc}, training auc_pr: {auc_pr}, best validation AUC: {self.best_metric}, weight_norm: {weight_norm} in {time_elapsed}')

            if epoch % self.args.save_every == 0:
                #save_path = os.path.dirname(os.path.abspath('__file__')) + '/openhgnn/output/' + self.model_name
                torch.save(self.model, self.args.save_path + '/ComPILE_chk.pth')

    def save_classifier(self):
        #save_path = os.path.dirname(os.path.abspath('__file__')) + '/openhgnn/output/' + self.model_name
        torch.save(self.model, self.args.save_path + '/best.pth')
        self.logger.info('Better models found w.r.t accuracy. Saved it!')


class Evaluator():
    def __init__(self, params, graph_classifier, data):
        self.params = params
        self.graph_classifier = graph_classifier
        self.data = data

    def eval(self, save=False):
        pos_scores = []
        pos_labels = []
        neg_scores = []
        neg_labels = []
        dataloader = DataLoader(self.data, batch_size=self.params.batch_size, shuffle=False,
                                num_workers=self.params.num_workers, collate_fn=self.params.collate_fn)

        self.graph_classifier.eval()
        with torch.no_grad():
            for b_idx, batch in enumerate(dataloader):
                (graphs_pos, r_labels_pos), g_labels_pos, (graph_neg, r_labels_neg), g_labels_neg = batch
                # data_pos, targets_pos, data_neg, targets_neg = self.params.move_batch_to_device(batch, self.params.device)
                # print([self.data.id2relation[r.item()] for r in data_pos[1]])
                # pdb.set_trace()

                g_labels_pos = torch.LongTensor(g_labels_pos).to(device=self.params.device)
                r_labels_pos = torch.LongTensor(r_labels_pos).to(device=self.params.device)

                g_labels_neg = torch.LongTensor(g_labels_neg).to(device=self.params.device)
                r_labels_neg = torch.LongTensor(r_labels_neg).to(device=self.params.device)

                score_pos = self.graph_classifier(graphs_pos)
                score_neg = self.graph_classifier(graph_neg)

                # preds += torch.argmax(logits.detach().cpu(), dim=1).tolist()
                pos_scores += score_pos.squeeze(1).detach().cpu().tolist()
                neg_scores += score_neg.squeeze(1).detach().cpu().tolist()
                pos_labels += g_labels_pos.tolist()
                neg_labels += g_labels_neg.tolist()

        # acc = metrics.accuracy_score(labels, preds)
        auc = metrics.roc_auc_score(pos_labels + neg_labels, pos_scores + neg_scores)
        auc_pr = metrics.average_precision_score(pos_labels + neg_labels, pos_scores + neg_scores)

        if save:
            pos_test_triplets_path = os.path.join(self.params.save_path,
                                                  'data/{}/{}.txt'.format(self.params.dataset, self.data.file_name))
            with open(pos_test_triplets_path) as f:
                pos_triplets = [line.split() for line in f.read().split('\n')[:-1]]
            pos_file_path = os.path.join(self.params.save_path,
                                         'data/{}/grail_{}_predictions.txt'.format(self.params.dataset,
                                                                                   self.data.file_name))
            with open(pos_file_path, "w") as f:
                for ([s, r, o], score) in zip(pos_triplets, pos_scores):
                    f.write('\t'.join([s, r, o, str(score)]) + '\n')

            neg_test_triplets_path = os.path.join(self.params.save_path,
                                                  'data/{}/neg_{}_0.txt'.format(self.params.dataset,
                                                                                self.data.file_name))
            with open(neg_test_triplets_path) as f:
                neg_triplets = [line.split() for line in f.read().split('\n')[:-1]]
            neg_file_path = os.path.join(self.params.save_path,
                                         'data/{}/grail_neg_{}_{}_predictions.txt'.format(self.params.dataset,
                                                                                          self.data.file_name,
                                                                                          self.params.constrained_neg_prob))
            with open(neg_file_path, "w") as f:
                for ([s, r, o], score) in zip(neg_triplets, neg_scores):
                    f.write('\t'.join([s, r, o, str(score)]) + '\n')

        return {'auc': auc, 'auc_pr': auc_pr}

