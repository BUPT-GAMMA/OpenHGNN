import copy

import dgl
import numpy as np
import torch as th
from sklearn.metrics import f1_score
from torch import nn
from tqdm import tqdm
import torch
import torch.nn.functional as F
from . import BaseFlow, register_flow
from ..models import build_model
from ..models.LogReg import LogReg

@register_flow("SHGP_trainer")
class SHGPTrainer(BaseFlow):
    """
    Pre-Training of GNNs on Heterogeneous Graphs
    """

    def __init__(self, args):
        super(SHGPTrainer, self).__init__(args)
        self.args = args
        self.target_type = args.target_type
        self.logger = args.logger
        self.train_percent = args.train_percent
        self.dataset = self.task.dataset
        self.label = self.dataset.label
        self.ft_dict = self.dataset.ft_dict
        self.adj_dict = self.dataset.adj_dict
        self.target_type = args.target_type
        self.num_cluster = int(
            self.ft_dict[self.target_type].shape[0] * args.compress_ratio)
        self.num_class = np.unique(self.label[self.target_type][0]).shape[0]
        self.layer_shape = []
        self.input_layer_shape = dict([(k, self.ft_dict[k].shape[1]) for k in self.ft_dict.keys()])
        self.hidden_layer_shape = [dict.fromkeys(self.ft_dict.keys(), l_hid) for l_hid in args.hidden_dim]
        self.output_layer_shape = dict.fromkeys(self.ft_dict.keys(), self.num_cluster)

        self.layer_shape.append(self.input_layer_shape)
        self.layer_shape.extend(self.hidden_layer_shape)
        self.layer_shape.append(self.output_layer_shape)

        self.net_schema = dict([(k, list(self.adj_dict[k].keys())) for k in self.adj_dict.keys()])
        self.args.net_schema = self.net_schema
        self.args.layer_shape = self.layer_shape
        self.args.label = self.label
        self.model = build_model(self.model).build_model_from_args(self.args, self.hg).to(self.device)
        # self.model = ATT_HGCN(
        #     net_schema=self.net_schema,
        #     layer_shape=self.layer_shape,
        #     label_keys=list(self.label.keys()),
        #     type_fusion=args.type_fusion,
        #     type_att_size=args.type_att_size,
        # )

    def preprocess(self):
        # super(SHGPTrainer, self).preprocess()
        if self.args.cuda and torch.cuda.is_available():
            self.model.cuda()
            for k in self.ft_dict:
                self.ft_dict[k] = self.ft_dict[k].cuda()
            for k in self.adj_dict:
                for kk in self.adj_dict[k]:
                    self.adj_dict[k][kk] = self.adj_dict[k][kk].cuda()
            for k in self.label:
                for i in range(len(self.label[k])):
                    self.label[k][i] = self.label[k][i].cuda()
        return

    def train(self):
        self.preprocess()
        if not self.args.load_from_pretrained:
            best = 1e9
            loss_list = []
            init_pseudo_label = 0
            pseudo_pseudo_label = 0
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.l2_coef)
            for epoch in range(self.args.epochs):
                self.model.train()
                optimizer.zero_grad()
                logits, embd, attention_dict = self.model(self.ft_dict, self.adj_dict)
                target_embd = embd[self.target_type]

                if epoch == 0:
                    init_pseudo_label = init_lpa(self.adj_dict, self.ft_dict, self.target_type, self.num_cluster)
                    pseudo_label_dict = init_pseudo_label
                elif epoch < self.args.warm_epochs:
                    pseudo_label_dict = init_pseudo_label
                else:
                    pseudo_label_dict = att_lpa(self.adj_dict, init_pseudo_label, attention_dict, self.target_type, self.num_cluster)
                    init_pseudo_label = pseudo_label_dict
                label_predict = torch.argmax(pseudo_label_dict[self.target_type], dim=1)
                logits = F.log_softmax(logits[self.target_type], dim=1)
                loss_train = F.nll_loss(logits, label_predict.long().detach())
                loss_train.backward()
                optimizer.step()
                loss_list.append(loss_train)
                if loss_train < best:
                    best = loss_train

                print(
                    'epoch: {:3d}'.format(epoch),
                    'train loss: {:.4f}'.format(loss_train.item()),
                )
            torch.save(self.model.state_dict(), "openhgnn/dataset/data/test/test")
        else:
            self.model.load_state_dict(torch.load("openhgnn/dataset/data/test/test"))

        # evaluate
        logits, embd, _ = self.model(self.ft_dict, self.adj_dict)
        target_embd = embd[self.target_type]
        label_target = self.label[self.target_type]
        true_label = label_target[0]
        idx_train = label_target[1]
        idx_val = label_target[2]
        idx_test = label_target[3]
        self.evaluate(target_embd, idx_train, idx_val, idx_test, true_label, self.num_class, isTest=True)


    def evaluate(self, embeds, idx_train, idx_val, idx_test, labels, num_class, isTest=True):
        print("----------------------------start evaluating----------------------------------")
        hid_units = embeds.shape[1]
        nb_classes = num_class
        xent = nn.CrossEntropyLoss()
        train_embs = embeds[idx_train]
        val_embs = embeds[idx_val]
        test_embs = embeds[idx_test]
        train_lbls = labels[idx_train]
        val_lbls = labels[idx_val]
        test_lbls = labels[idx_test]

        run_num = 10
        epoch_num = 50
        accs = []
        micro_f1s = []
        macro_f1s = []
        macro_f1s_val = []
        micro_f1s_val = []
        for _ in range(run_num):
            log = LogReg(hid_units, nb_classes)
            # log.load_state_dict(load_LogReg(torch.load(args.pretrain_model_dir)), strict=False)
            opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.0)
            if torch.cuda.is_available():
                log.cuda()

            val_accs = [];
            test_accs = []
            val_micro_f1s = [];
            test_micro_f1s = []
            val_macro_f1s = [];
            test_macro_f1s = []

            for iter_ in range(epoch_num):
                # train
                log.train()
                opt.zero_grad()

                logits = log(train_embs)
                loss = xent(logits, train_lbls)

                loss.backward(retain_graph=True)
                opt.step()

                # val
                logits = log(val_embs)
                preds = torch.argmax(logits, dim=1)

                val_acc = torch.sum(preds == val_lbls).float() / val_lbls.shape[0]
                if torch.cuda.is_available():
                    val_f1_macro = f1_score(val_lbls.cpu(), preds.cpu(), average='macro')
                    val_f1_micro = f1_score(val_lbls.cpu(), preds.cpu(), average='micro')
                else:
                    val_f1_macro = f1_score(val_lbls.cpu(), preds, average='macro')
                    val_f1_micro = f1_score(val_lbls.cpu(), preds, average='micro')

                val_accs.append(val_acc.item())
                val_macro_f1s.append(val_f1_macro)
                val_micro_f1s.append(val_f1_micro)

                # test
                logits = log(test_embs)
                preds = torch.argmax(logits, dim=1)

                test_acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
                if torch.cuda.is_available():
                    test_f1_macro = f1_score(test_lbls.cpu(), preds.cpu(), average='macro')
                    test_f1_micro = f1_score(test_lbls.cpu(), preds.cpu(), average='micro')
                else:
                    test_f1_macro = f1_score(test_lbls.cpu(), preds, average='macro')
                    test_f1_micro = f1_score(test_lbls.cpu(), preds, average='micro')

                test_accs.append(test_acc.item())
                test_macro_f1s.append(test_f1_macro)
                test_micro_f1s.append(test_f1_micro)

            max_iter = val_accs.index(max(val_accs))
            accs.append(test_accs[max_iter])

            max_iter = val_macro_f1s.index(max(val_macro_f1s))
            macro_f1s.append(test_macro_f1s[max_iter])
            macro_f1s_val.append(val_macro_f1s[max_iter])  ###

            max_iter = val_micro_f1s.index(max(val_micro_f1s))
            micro_f1s.append(test_micro_f1s[max_iter])
            micro_f1s_val.append(val_micro_f1s[max_iter])

        if isTest:
            print("\t[Classification-test] Macro-F1: {:.4f} ({:.4f}) | Micro-F1: {:.4f} ({:.4f})".format(
                np.mean(macro_f1s), np.std(macro_f1s), np.mean(micro_f1s), np.std(micro_f1s)))


def gen_rand_label(ft_dict,num_cluster):
    rand_label_dict = dict.fromkeys(ft_dict.keys())
    for k in ft_dict:
        rand_label = torch.randint(num_cluster, (ft_dict[k].shape[0],))
        rand_label = F.one_hot(rand_label, num_cluster).type(dtype=torch.float32)
        rand_label_dict[k] = rand_label

    return rand_label_dict



def cat_homo_adj(adj_dict):
    for k in adj_dict:
        print(k)

    return None



def lpa(init_label_dict,adj_dict,target_type,num_cluster,max_iter=1000):
    pseudo_label_dict = copy.deepcopy(init_label_dict)
    for k in pseudo_label_dict:
        if torch.cuda.is_available():
            pseudo_label_dict[k] = pseudo_label_dict[k].cuda()
    net_schema = dict([(k, list(adj_dict[k].keys())) for k in adj_dict.keys()])
    target_label_list=[]
    soft_label=0
    for i in range(max_iter):
        for k in net_schema:
            k_nbs_label_list = []
            for kk in net_schema[k]:
                try:
                    soft_label = torch.spmm(adj_dict[k][kk], pseudo_label_dict[kk])
                except KeyError as ke:
                    soft_label = torch.spmm(adj_dict[k][kk], pseudo_label_dict[k])
                finally:
                    k_nbs_label_list.append(soft_label)
            new_k_label = torch.cat([nb_label.unsqueeze(1) for nb_label in k_nbs_label_list], 1)

            new_k_label = new_k_label.sum(1)
            new_k_label = torch.argmax(new_k_label, dim=1)
            new_k_label = F.one_hot(new_k_label, num_cluster).type(dtype=torch.float32)
            pseudo_label_dict[k] = new_k_label
            if k==target_type:
                target_label_list.append(new_k_label)
        if len(target_label_list)>1:
            if target_label_list[-2].equal(target_label_list[-1]):
                break

    return pseudo_label_dict


def init_lpa(adj_dict, ft_dict, target_type, num_cluster):
    run_num = 1
    for i in range(run_num):
        init_label_dict = gen_rand_label(ft_dict, num_cluster)
        pseudo_label_dict = lpa(init_label_dict, adj_dict, target_type, num_cluster)

    return pseudo_label_dict


def att_lpa(adj_dict, init_pseudo_label, attention_dict, target_type, num_cluster, max_iter=1000):
    pseudo_label_dict = copy.deepcopy(init_pseudo_label)
    current_label_dict = copy.deepcopy(init_pseudo_label)

    for k in pseudo_label_dict:
        if torch.cuda.is_available():
            pseudo_label_dict[k] = pseudo_label_dict[k].cuda()
    for k in current_label_dict:
        if torch.cuda.is_available():
            current_label_dict[k] = current_label_dict[k].cuda()
    net_schema = dict([(k, list(adj_dict[k].keys())) for k in adj_dict.keys()])
    target_label_list = []
    soft_label = 0
    for _ in range(max_iter):
        for m in range(len(attention_dict)):
            for k in net_schema:
                k_nbs_label_list = []
                k_nbs_label_list.append(pseudo_label_dict[k])
                for kk in net_schema[k]:
                    try:
                        soft_label = torch.spmm(adj_dict[k][kk], current_label_dict[kk])
                    except KeyError as ke:
                        soft_label = torch.spmm(adj_dict[k][kk], current_label_dict[k])
                    finally:
                        k_nbs_label_list.append(soft_label)
                pseudo_label_dict[k] = torch.cat([nb_label.unsqueeze(1) for nb_label in k_nbs_label_list], 1).mul(
                    attention_dict[m][k].unsqueeze(-1)).sum(1)

            for k in net_schema:
                current_label_dict[k] = pseudo_label_dict[k]

        for k in net_schema:
            new_k_label = torch.argmax(pseudo_label_dict[k], dim=1)
            if k == target_type:
                target_label_list.append(new_k_label)
            pseudo_label_dict[k] = F.one_hot(new_k_label, num_cluster).type(dtype=torch.float32)
            current_label_dict[k] = pseudo_label_dict[k]
        if len(target_label_list) > 1:
            if target_label_list[-2].equal(target_label_list[-1]):
                break

    return pseudo_label_dict






