from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from ..models import build_model
from . import BaseFlow, register_flow
from ..tasks import build_task


@register_flow("MeiREC_trainer")
class MeiRECTrainer(BaseFlow):

    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.model_name = args.model
        self.device = args.device
        self.task = build_task(args)

        self.model = build_model(self.model_name).build_model_from_args(
            self.args).model
        print("build_model_finish")
        self.model = self.model.to(self.device)
        self.loss_fn = nn.BCELoss(reduction='mean')
        # self.evaluator = self.task.evaluate
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, 0.99)
        self.patience = args.patience
        self.max_epoch = args.max_epoch

    def train(self):
        train_loader = self.task.train_loader
        test_loader = self.task.test_loader

        for epoch in range(1, self.max_epoch + 1):
            pbar = tqdm(train_loader)
            if epoch % 3 == 0:
                for i, data in enumerate(pbar):
                    # train step, data: batch * features
                    if i != len(train_loader) - 1:
                        output, train_loss, train_auc, train_acc = self._train_step(data)
                        # record bar
                        pbar.set_description(f"epoch[{epoch}/{self.max_epoch}],\
                                data[{i}/{len(train_loader)}]")
                        pbar.set_postfix(
                            OrderedDict({
                                "train_loss": train_loss.item(),
                                "train_auc": train_auc,
                                "train_acc": train_acc
                            }))

                    else:
                        pr = []
                        la = []
                        _, train_loss, train_auc, train_acc = self._train_step(data)
                        for j, data_test in enumerate(test_loader):

                            output, test_loss, test_auc, test_acc = self.eval_model(data_test)
                            pr.append(output)
                            la.append(data_test["labels"])

                        prediction_test = torch.cat(pr, 0).cuda()
                        labels_test = torch.cat(la, 0).cuda()
                        loss_cross = self.loss_fn(prediction_test, labels_test.float())
                        loss_total = loss_cross + self.model.regular_loss
                        # auc and acc value
                        predicts_cpu = prediction_test.cpu().detach().numpy()
                        labels_cpu = labels_test.cpu().detach().numpy()
                        correct_num = sum((predicts_cpu > 0.5) == labels_cpu)
                        acc = correct_num / len(predicts_cpu)

                        # auc = roc_auc_score(labels_cpu, predicts_cpu)
                        auc = self.auc(labels_cpu, predicts_cpu)

                        pbar.set_description(f"epoch[{epoch}/{self.max_epoch}],\
                                                data[{i}/{len(train_loader)}]")
                        pbar.set_postfix(
                            OrderedDict({
                                "train_loss": train_loss.item(),
                                "train_auc": train_auc,
                                "train_acc": train_acc,
                                "test_loss": loss_total.item(),
                                "test_auc": auc,
                                "test_acc": acc
                            }))
            else:
                for i, data in enumerate(pbar):
                    # train step, data: batch * features
                    output, train_loss, train_auc, train_acc = self._train_step(data)
                    # record bar
                    pbar.set_description(f"epoch[{epoch}/{self.max_epoch}],\
                            data[{i}/{len(train_loader)}]")
                    pbar.set_postfix(
                        OrderedDict({
                            "train_loss": train_loss.item(),
                            "train_auc": train_auc,
                            "train_acc": train_acc
                        }))

            self.scheduler.step()

            # if epoch % 10 == 0:
            #     self.save_model(str(epoch))

            # self.save_model("latest.pth.tar")

    def _train_step(self, x):
        self.model.train()         # Sets the module in training mode
        output, loss, auc, acc = self.forward(x)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
        self.optimizer.step()
        return output, loss, auc, acc

    def eval_model(self, x):
        self.model.eval()         # Sets the module in evaluation mode.

        with torch.no_grad():
            output, loss, auc, acc = self.forward(x)

        return output, loss, auc, acc

    def _confusion_matrix_at_thresholds(self, labels, preds, thresholds):

        values = {}
        tp = []
        fp = []
        tn = []
        fn = []

        for thr in thresholds:
            tp.append(sum((preds > thr) & (labels == 1)))
            fp.append(sum((preds > thr) & (labels == 0)))
            tn.append(sum((preds < thr) & (labels == 0)))
            fn.append(sum((preds < thr) & (labels == 1)))

        values['tp'] = torch.from_numpy(np.array(tp))
        values['fp'] = torch.from_numpy(np.array(fp))
        values['tn'] = torch.from_numpy(np.array(tn))
        values['fn'] = torch.from_numpy(np.array(fn))

        return values

    def auc(self, labels, preds, num_thresholds=200):   # tensorflow1.0_auc
        kepsilon = 1e-7                # to account for floating point imprecisions
        thresholds = [(i + 1) * 1.0 / (num_thresholds - 1)
                      for i in range(num_thresholds - 2)]
        thresholds = [0.0 - kepsilon] + thresholds + [1.0 + kepsilon]

        values = self._confusion_matrix_at_thresholds(
            labels, preds, thresholds)

        # Add epsilons to avoid dividing by 0.
        epsilon = 1.0e-6

        def compute_auc(tp, fn, tn, fp):

            rec = torch.div(tp + epsilon, tp + fn + epsilon)

            fp_rate = torch.div(fp, fp + tn + epsilon)
            x = fp_rate
            y = rec

            return torch.sum(
                torch.multiply(x[:num_thresholds - 1] - x[1:],
                          (y[:num_thresholds - 1] + y[1:]) / 2.))

        # sum up the areas of all the trapeziums
        auc_value = compute_auc(
            values['tp'], values['fn'], values['tn'], values['fp'])

        return auc_value.item()

    def forward(self, x):
        inputs = x['data'].T
        labels = x['labels']

        inputs = inputs.cuda()
        labels = labels.cuda()

        predicts = self.model(inputs)
        # loss
        loss_cross = self.loss_fn(predicts, labels.float())
        loss_total = loss_cross + self.model.regular_loss
        # auc and acc value
        predicts_cpu = predicts.cpu().detach().numpy()
        labels_cpu = labels.cpu().detach().numpy()
        correct_num = sum((predicts_cpu > 0.5) == labels_cpu)
        acc = correct_num / len(predicts_cpu)

        auc = self.auc(labels_cpu, predicts_cpu)

        return predicts, loss_total.cpu(), auc, acc

