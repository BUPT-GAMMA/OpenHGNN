import argparse
import copy
import dgl
import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F
from ..models import build_model
from ..sampler import HGT_preprocess4mag, HGTsampler
from . import BaseFlow, register_flow
from ..tasks import build_task
from ..utils import extract_embed, EarlyStopping


@register_flow("hgttrainer")
class HGTTrainer(BaseFlow):

    """HGTtrainer flows.
    Supported Model: HGT
    Supported Dataset：ogbn-mag

    """

    def __init__(self, args):
        super(HGTTrainer, self).__init__(args)

        self.args = args
        self.model_name = args.model
        self.device = args.device
        self.task = build_task(args)

        self.hg = self.task.get_graph().to(self.device)
        self.num_classes = self.task.dataset.num_classes

        # Build the model. If the output dim is not equal the number of classes, a MLP will follow the gnn model.
        if args.out_dim != self.num_classes:
            print('Modify the out_dim with num_classes')
            args.out_dim = self.num_classes
        self.model = build_model(self.model_name).build_model_from_args(self.args, self.hg)
        if not hasattr(args, 'out_dim') or args.out_dim == self.num_classes:
            pass
        else:
            #self.model = MLP_follow_model(self.model, args.out_dim, self.num_classes)
            pass
        self.model = self.model.to(self.device)

        self.evaluator = self.task.get_evaluator('acc')
        self.loss_fn = self.task.get_loss_fn()
        self.optimizer = (
            torch.optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay))

        self.patience = args.patience
        self.max_epoch = args.max_epoch

        self.category = self.task.dataset.category
        self.train_idx, self.val_idx, self.test_idx = self.task.get_split()
        self.labels = self.task.get_labels().to(self.device)


    def preprocess(self):
        if self.args.mini_batch_flag:
            N_SAMPLE_NODES_PER_TYPE = 1280  # number of nodes to sample per node type per sampler step
            N_SAMPLE_STEPS = 6  # number of sampler steps
           # self.hg = HGT_preprocess4mag(self.hg, self.train_idx)
            sampler = HGTsampler(self.hg.to('cpu'), self.category, N_SAMPLE_NODES_PER_TYPE, N_SAMPLE_STEPS)
            self.dataloader = torch.utils.data.DataLoader(
                self.train_idx,
                batch_size=self.args.batch_size,
                collate_fn=sampler.sampler_subgraph,
                #num_workers=self.args.num_workers,
                shuffle=True,
                drop_last=False,
            )
            # self.dataloader_it = iter(dataloader)
            # next(self.dataloader_it)
        return

    def train(self):
        self.preprocess()
        stopper = EarlyStopping(self.args.patience)
        epoch_iter = tqdm(range(self.max_epoch))
        print(0.2)
        for epoch in epoch_iter:
            self.evaluate()
            # if self.args.mini_batch_flag:
            #     train_loss = self._mini_train_step()
            # else:
            #     train_loss = self._full_train_setp()
            # print(train_loss)
            # torch.save(self.model.state_dict(), './openhgnn/output/HGT/epoch' + str(epoch) + 'HGT.pt')
            #if (epoch + 1) % self.evaluate_interval == 0:
            #f1, losses = self._test_step()

            # train_f1 = f1["train"]
            # val_f1 = f1["val"]
            # val_loss = losses["val"]
            # epoch_iter.set_description(
            #     f"Epoch: {epoch:03d}, Train_macro_f1: {train_f1[0]:.4f},Val_macro_f1: {val_f1[0]:.4f}, train_loss:{train_loss: .4f}"
            # )
            # early_stop = stopper.step(val_loss, val_f1[0], self.model)
            # if early_stop:
            #     print('Early Stop!\tEpoch:' + str(epoch))
            #     break
        print(f"Valid accurracy = {stopper.best_score: .4f}")
        self.model = stopper.best_model
        test_f1, _ = self._test_step(split="test")
        val_f1, _ = self._test_step(split="val")
        print(f"Test accuracy = {test_f1[0]:.4f}")
        return dict(Acc=test_f1, ValAcc=val_f1)

    def _full_train_setp(self):
        self.model.train()
        h = self.hg.ndata['h']
        logits = self.model(self.hg, h)[self.category]
        loss = self.loss_fn(logits[self.train_idx], self.labels[self.train_idx])
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _mini_train_step(self,):
        self.model.train()
        loss_all = 0
        #torch.save(self.model.state_dict(), './openhgnn/output/HGT/'+ str(0) + 'HGT.pt')
        for i, (sg, seed_nodes) in tqdm(enumerate(self.dataloader)):
            sg = sg.to(self.device)
            h = sg.ndata.pop('h')
            logits = self.model(sg, h)[self.category]

            labels = self.labels[sg.ndata[dgl.NID][self.category]].squeeze()
            loss = self.loss_fn(logits, labels)
            loss_all += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return loss_all
    def evaluate(self, split=None):
        ck_pt = torch.load('./openhgnn/output/HGT/epoch7HGT.pt')
        log = open('./openhgnn/output/HGT/train.log', 'w')
        self.model.load_state_dict(ck_pt)
        self.model.eval()
        with torch.no_grad():
            sum_preds = torch.zeros(self.hg.num_nodes(self.category), self.num_classes).to(self.device)
            counts = torch.zeros(self.hg.num_nodes(self.category)).to(self.device)
            for sg, num_seed_nodes in tqdm(self.dataloader):
                sg = sg.to(self.device)
                h = sg.ndata.pop('h')
                logits = self.model(sg, h)[self.category]
                #labels = self.labels[sg.ndata[dgl.NID][self.category]].squeeze()
                nid = sg.ndata[dgl.NID][self.category]

                ones = torch.ones(nid.shape[0]).to(self.device)
                sum_preds.scatter_add_(0, nid[:, None].expand_as(logits), logits)
                counts.scatter_add_(0, nid, ones)

            avg_preds = sum_preds/ counts[:, None]
            avg_preds[torch.isnan(avg_preds)] = 0
            final_preds = sum_preds.argmax(1)

            del sum_preds
            del counts

            if split == "train":
                mask = self.train_idx
            elif split == "val":
                mask = self.val_idx
            elif split == "test":
                mask = self.test_idx
            else:
                mask = None

            if mask is not None:
                loss = self.loss_fn(avg_preds[mask], self.labels[mask])
                metric = self.evaluator(self.labels[mask].to('cpu'), final_preds[mask].to('cpu'))
                return metric, loss
            else:
                masks = {'train': self.train_idx, 'val': self.val_idx, 'test': self.test_idx}
                metrics = {key: self.evaluator(self.labels[mask].to('cpu'), final_preds[mask].to('cpu'))
                           for key, mask in masks.items()}
                losses = {key: self.loss_fn(avg_preds[mask], self.labels[mask].squeeze()) for key, mask in masks.items()}
                print('Train:', metrics['train'], 'Validation:', metrics['val'], 'Test:', metrics['test'])
                print('Train:', metrics['train'], 'Validation:', metrics['val'], 'Test:', metrics['test'], file=log, flush=True)
                return metrics, losses

    def _test_step(self, split=None, logits=None):
        self.model.eval()
        with torch.no_grad():
            h = self.hg.ndata['h']
            logits = logits if logits else self.model(self.hg, h)[self.category]
            if split == "train":
                mask = self.train_idx
            elif split == "val":
                mask = self.val_idx
            elif split == "test":
                mask = self.test_idx
            else:
                mask = None

            if mask is not None:
                loss = self.loss_fn(logits[mask], self.labels[mask]).item()
                metric = self.evaluator(self.labels[mask].to('cpu'), logits[mask].argmax(dim=1).to('cpu'))
                return metric, loss
            else:
                masks = {'train': self.train_idx, 'val': self.val_idx, 'test': self.test_idx}
                metrics = {key: self.evaluator(self.labels[mask].to('cpu'), logits[mask].argmax(dim=1).to('cpu')) for key, mask in masks.items()}
                losses = {key: self.loss_fn(logits[mask], self.labels[mask]).item() for key, mask in masks.items()}
                return metrics, losses


