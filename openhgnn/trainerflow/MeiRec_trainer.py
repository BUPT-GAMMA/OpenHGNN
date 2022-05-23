from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from ..models import build_model
from . import BaseFlow, register_flow
from ..tasks import build_task


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


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
        self.evaluator = self.task.evaluate
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)
        # self.optimizer = (torch.optim.Adam(self.model.parameters(),
        #                                    lr=args.lr,
        #                                    weight_decay=args.weight_decay))
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, 0.99)
        self.patience = args.patience
        self.max_epoch = args.max_epoch
        self.gpu_support = True if torch.cuda.is_available() else False

    def train(self):
        train_loader = self.task.train_loader
        test_loader = self.task.test_loader
        test_loader = cycle(test_loader)

        for epoch in range(1, self.max_epoch + 1):
            pbar = tqdm(train_loader)
            for i, data in enumerate(pbar):
                # train step, data: batch * features
                output, train_loss, train_acc = self._train_step(data)
                # record bar
                pbar.set_description(f"epoch[{epoch}/{self.max_epoch}],\
                        data[{i}/{len(train_loader)}]")
                pbar.set_postfix(
                    OrderedDict({
                        "train_loss": train_loss.item(),
                        "train_acc": train_acc
                    }))

                if i % 2 == 0:
                    val_data = next(test_loader)
                    self.model.set_mode(mode="test")
                    output, test_loss, test_acc = self.eval_model(val_data)
                    self.model.set_mode(mode="train")
                    pbar.set_postfix(
                        OrderedDict({
                            "train_loss": train_loss.item(),
                            "test_loss": test_loss.item(),
                            "train_acc": train_acc,
                            "test_acc": test_acc,
                        }))

            self.scheduler.step()

            # if epoch % 10 == 0:
            #     self.save_model(str(epoch))

            # self.save_model("latest.pth.tar")

    def _train_step(self, x):
        self.model.train()
        output, loss, acc = self.forward(x)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return output, loss, acc

    def eval_model(self, x):
        self.model.eval()

        with torch.no_grad():
            output, loss, acc = self.forward(x)

        return output, loss, acc

    def forward(self, x):
        inputs = x['data'].T
        labels = x['labels']
        if self.gpu_support:
            inputs = inputs.cuda()
            labels = labels.cuda()

        predicts = self.model(inputs)
        # loss
        loss_cross = self.loss_fn(predicts, labels.float())
        loss_total = loss_cross + self.model.regular_loss
        # auc value
        predicts_cpu = predicts.cpu().detach().numpy()
        labels_cpu = labels.cpu().detach().numpy()
        correct_num = sum((predicts_cpu > 0.5) == labels_cpu)
        acc = correct_num / len(predicts_cpu)
        return predicts, loss_total.cpu(), acc
