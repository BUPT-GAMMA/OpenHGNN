import torch
import torch.nn as nn
from tqdm import tqdm
from . import BaseFlow, register_flow
from ..models import build_model


@register_flow("abnorm_event_detection")
class AbnormEventDetection(BaseFlow):
    """AbnormEventDetection flow

    Supported Model: AEHCL
    Supported Dataset: aminer4AEHCL

    """

    def __init__(self, args=None):
        super(AbnormEventDetection, self).__init__(args)
        self.event_label = self.task.dataset.event_label
        args.center_type, args.context_type = self.task.dataset.center_type, self.task.dataset.context_type
        args.in_dim = self.task.dataset.in_dim
        all_len = self.task.dataset.type_max_num[args.center_type]
        for tp in args.context_type:
            all_len += self.task.dataset.type_max_num[tp]
        args.all_len = all_len
        self.model = build_model(self.model).build_model_from_args(self.args).to(self.device)
        self.ap = self.task.get_evaluator('AP')
        self.auc = self.task.get_evaluator('AUC')
        self.batch_size = args.batch_size
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.eval_epoch = args.eval_epoch
        self.task.dataset.set_neg_num(args.neg_num)

    def train(self):
        epoch_iter = tqdm(range(self.max_epoch))
        scores = dict()
        for epoch in epoch_iter:
            self.task.dataset.get_complete_events_features()
            loss = self._full_train_step()
            print("loss: ", loss)

        epoch_test_iter = tqdm(range(self.eval_epoch))
        for epoch in epoch_test_iter:
            score = self._full_test_step()
            scores[epoch] = score.tolist()
        event_label = self.event_label.tolist()
        the_score_all = []
        for i in range(len(event_label)):
            the_score = 0.0
            for j in range(self.eval_epoch):
                the_score += scores[j][i]
            the_score_all.append(the_score)
        ap = self.ap(event_label, the_score_all)
        auc = self.auc(event_label, the_score_all)
        print("AP:", ap)
        print("AUC:", auc)

    def _full_train_step(self):
        self.model.train()
        event_list, neg_event_list, pos_event_list, neg_context_list, neg_entity_list, event_mask, type_num = self.task.dataset.get_batch(
            self.batch_size, device=self.device)
        num_of_batch = len(event_list)
        end_loss = 0.0
        train_epoch_iter = tqdm(range(num_of_batch))
        for i in train_epoch_iter:
            self.optimizer.zero_grad()
            loss = self.model(event_list[i], pos_event_list[i], neg_event_list[i], neg_context_list[i], neg_entity_list[i], event_mask[i], type_num[i])
            loss.backward()
            self.optimizer.step()
            end_loss += loss
        return end_loss / num_of_batch

    def _full_test_step(self):
        self.model.eval()
        event_list, neg_event_list, pos_event_list, neg_context_list, neg_entity_list, event_mask, type_num = self.task.dataset.get_batch(
            self.batch_size, shuffle=False, device=self.device)
        num_of_batch = len(event_list)
        scores = None
        with torch.no_grad():
            test_epoch_iter = tqdm(range(num_of_batch))
            for i in test_epoch_iter:
                self.optimizer.zero_grad()
                score = self.model(event_list[i], pos_event_list[i], neg_event_list[i], neg_context_list[i],
                                   neg_entity_list[i], event_mask[i], type_num[i])
                if scores is None:
                    scores = score
                else:
                    scores = torch.cat((scores, score), 0)

        return scores
