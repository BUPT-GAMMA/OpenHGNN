import argparse
import copy
import dgl
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from ..models import build_model
from ..models.HeCo import LogReg
from . import BaseFlow, register_flow
from ..tasks import build_task
from ..utils import extract_embed, EarlyStopping
from sklearn.metrics import f1_score, roc_auc_score


@register_flow("HeCo_trainer")
class HeCoTrainer(BaseFlow):

    def __init__(self, args):
        super(HeCoTrainer, self).__init__(args)

        self.args = args
        self.model_name = args.model
        self.device = args.device
        self.task = build_task(args)

        self.hg = self.task.get_graph().to(self.device)
        self.num_classes = int(self.task.dataset.num_classes)
        self.args.category = self.task.dataset.category
        
        self.category = self.args.category
        self.pos = self.task.dataset.pos.to(self.device)
        self.model = build_model(self.model_name).build_model_from_args(self.args, self.hg)
        print("build_model_finish")
        self.model = self.model.to(self.device)

        self.evaluator = self.task.get_evaluator('f1')
        self.optimizer = (
            torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay))
        self.patience = args.patience
        self.max_epoch = args.max_epoch

        self.train_idx, self.val_idx, self.test_idx = self.task.get_split()
        self.labels = self.task.get_labels().to(self.device)

    def preprocess(self):
        super(HeCoTrainer, self).preprocess()

    def train(self):
        self.preprocess()
        stopper = EarlyStopping(self.args.patience)
        # epoch_iter = tqdm(range(self.max_epoch))
        for epoch in range(self.max_epoch):
            '''use earlyStopping'''
            loss = self._full_train_step()
            early_stop = stopper.loss_step(loss, self.model)
            print((f"Epoch: {epoch:03d}, Loss: {loss:.4f}"))
            
            if early_stop:
                print('Early Stop!\tEpoch:' + str(epoch))
                break
        
        # Evaluation
        model = stopper.load_model(self.model)
        model.eval()
        h_dict = self.model.input_feature()
        embeds = model.get_embeds(self.hg, h_dict=h_dict)
        self.evaluate(embeds,)

    def _full_train_step(self):
        self.model.train()
        self.optimizer.zero_grad()
        h_dict = self.model.input_feature()
        loss = self.model(self.hg, h_dict, self.pos)
        loss.backward()
        self.optimizer.step()
        loss = loss.cpu()
        loss = loss.detach().numpy()
        return loss

    def evaluate(self, embeds):
        hid_units = embeds.shape[1]

        xent = nn.CrossEntropyLoss()

        train_embs = embeds[self.train_idx]
        val_embs = embeds[self.val_idx]
        test_embs = embeds[self.test_idx]

        train_lbls = self.labels[self.train_idx].reshape(-1)
        val_lbls = self.labels[self.val_idx].reshape(-1)
        test_lbls = self.labels[self.test_idx].reshape(-1)
        
        accs = []
        micro_f1s = []
        macro_f1s = []
        macro_f1s_val = []
        auc_score_list = []
        for _ in range(50):
            log = LogReg(hid_units, self.num_classes)
            opt = torch.optim.Adam(log.parameters(), lr=self.args.eva_lr, weight_decay=self.args.eva_wd)
            log.to(self.device)

            val_accs = []
            test_accs = []
            val_micro_f1s = []
            test_micro_f1s = []
            val_macro_f1s = []
            test_macro_f1s = []
            
            logits_list = []
            for iter_ in range(200):
                # train
                log.train()
                opt.zero_grad()

                logits = log(train_embs)
                loss = xent(logits, train_lbls)

                loss.backward()
                opt.step()

                # val
                logits = log(val_embs)
                preds = torch.argmax(logits, dim=1)
    
                val_acc = torch.sum(preds == val_lbls).float() / val_lbls.shape[0]
                val_f1_macro = f1_score(val_lbls.cpu(), preds.cpu(), average='macro')
                val_f1_micro = f1_score(val_lbls.cpu(), preds.cpu(), average='micro')
    
                val_accs.append(val_acc.item())
                val_macro_f1s.append(val_f1_macro)
                val_micro_f1s.append(val_f1_micro)
    
                # test
                logits = log(test_embs)
                preds = torch.argmax(logits, dim=1)
    
                test_acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
                test_f1_macro = f1_score(test_lbls.cpu(), preds.cpu(), average='macro')
                test_f1_micro = f1_score(test_lbls.cpu(), preds.cpu(), average='micro')
    
                test_accs.append(test_acc.item())
                test_macro_f1s.append(test_f1_macro)
                test_micro_f1s.append(test_f1_micro)
                logits_list.append(logits)
    
            max_iter = val_accs.index(max(val_accs))
            accs.append(test_accs[max_iter])
            max_iter = val_macro_f1s.index(max(val_macro_f1s))
            macro_f1s.append(test_macro_f1s[max_iter])
            macro_f1s_val.append(val_macro_f1s[max_iter])
    
            max_iter = val_micro_f1s.index(max(val_micro_f1s))
            micro_f1s.append(test_micro_f1s[max_iter])
    
            # auc
            best_logits = logits_list[max_iter]
            best_proba = F.softmax(best_logits, dim=1)
            auc_score_list.append(roc_auc_score(y_true=test_lbls.detach().cpu().numpy(),
                                                y_score=best_proba.detach().cpu().numpy(),
                                                multi_class='ovr'
                                                ))
    
        
        print("\t[Classification] Macro-F1_mean: {:.4f} var: {:.4f}  Micro-F1_mean: {:.4f} var: {:.4f} auc {:.4f}"
              .format(np.mean(macro_f1s),
                      np.std(macro_f1s),
                      np.mean(micro_f1s),
                      np.std(micro_f1s),
                      np.mean(auc_score_list),
                      np.std(auc_score_list)
                      )
              )
