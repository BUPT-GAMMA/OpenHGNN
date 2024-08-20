import os
import gc
import time
import uuid
import argparse
import datetime
import numpy as np

import torch
import torch.nn.functional as F

import os
import sys
import gc
import random
import dgl.function as fn

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import scipy.sparse as sp
from sklearn.metrics import f1_score
from tqdm import tqdm
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
from openhgnn.models import build_model
from . import BaseFlow, register_flow
from ..tasks import build_task
import functools
from contextlib import closing
import multiprocessing as mp
from multiprocessing import Pool
from tqdm import tqdm
from ..tasks import NodeClassification

@register_flow("SeHGNN_trainer")
class SeHGNNtrainer(BaseFlow):
    def __init__(self,args):
        super(SeHGNNtrainer, self).__init__(args)
        args.stages = [int(item.strip()) for item in args.stages.split(',')]
        self.args = args
        self.flow = NodeClassification(args)
    def train(self):
        args = self.args
        if args.seed > 0:
            self.set_random_seed(args.seed)
        num_nodes = self.flow.dataset.SeHGNN_g.num_nodes("P")
        n_classes = int(self.flow.labels.max()) + 1
        evaluator = self.flow.get_evaluator("acc")
        # =======
        # rearange node idx (for feats & labels)
        # =======
        train_node_nums = len(self.flow.train_idx)
        valid_node_nums = len(self.flow.val_idx)
        test_node_nums = len(self.flow.test_idx)
        trainval_point = train_node_nums
        valtest_point = trainval_point + valid_node_nums
        total_num_nodes = len(self.flow.train_idx) + len(self.flow.val_idx) + len(self.flow.test_idx)

        init2sort = torch.cat([self.flow.train_idx, self.flow.val_idx, self.flow.test_idx])
        sort2init = torch.argsort(init2sort)
        assert torch.all(self.flow.labels[init2sort][sort2init] == self.flow.labels)
        labels = self.flow.labels[init2sort]
        # =======
        # features propagate alongside the metapath
        # =======
        tgt_type = 'P'
        max_hops = args.num_hops + 1
        # compute k-hop feature
        self.flow.dataset.SeHGNN_g = self.hg_propagate(self.flow.dataset.SeHGNN_g, tgt_type, args.num_hops, max_hops, echo=False)
        feats = {}
        keys = list(self.flow.dataset.SeHGNN_g.nodes[tgt_type].data.keys())
        print(f'Involved feat keys {keys}')
        for k in keys:
            feats[k] = self.flow.dataset.SeHGNN_g.nodes[tgt_type].data.pop(k)
        self.flow.dataset.SeHGNN_g = self.clear_hg(self.flow.dataset.SeHGNN_g, echo=False)
        feats = {k: v[init2sort] for k, v in feats.items()}
        gc.collect()
        all_loader = torch.utils.data.DataLoader(
            torch.arange(num_nodes), batch_size=args.batch_size, shuffle=False, drop_last=False)

        checkpt_folder = f'./openhgnn/output/SeHGNN/{args.dataset}/'
        if not os.path.exists(checkpt_folder):
            os.makedirs(checkpt_folder)

        if args.amp:
            scalar = torch.cuda.amp.GradScaler()
        else:
            scalar = None
        device = "cuda:{}".format(args.gpu) if not args.cpu else 'cpu'
        labels_cuda = labels.long().to(device)
        checkpt_file = checkpt_folder + uuid.uuid4().hex
        print(checkpt_file)

        for stage in range(args.start_stage, len(args.stages)):
            epochs = args.stages[stage]

            if len(args.reload):
                pt_path = f'./openhgnn/output/SeHGNN/ogbn-mag/{args.reload}_{stage-1}.pt'
                assert os.path.exists(pt_path)
                print(f'Reload raw_preds from {pt_path}', flush=True)
                raw_preds = torch.load(pt_path, map_location='cpu')

            # =======
            # Expand training set & train loader
            # =======
            if stage > 0:
                preds = raw_preds.argmax(dim=-1)
                predict_prob = raw_preds.softmax(dim=1)

                train_acc = evaluator(preds[:trainval_point], labels[:trainval_point])
                val_acc = evaluator(preds[trainval_point:valtest_point], labels[trainval_point:valtest_point])
                test_acc = evaluator(preds[valtest_point:total_num_nodes], labels[valtest_point:total_num_nodes])

                print(f'Stage {stage-1} history model:\n\t' \
                    + f'Train acc {train_acc*100:.4f} Val acc {val_acc*100:.4f} Test acc {test_acc*100:.4f}')

                confident_mask = predict_prob.max(1)[0] > args.threshold
                val_enhance_offset  = torch.where(confident_mask[trainval_point:valtest_point])[0]
                test_enhance_offset = torch.where(confident_mask[valtest_point:total_num_nodes])[0]
                val_enhance_nid     = val_enhance_offset + trainval_point
                test_enhance_nid    = test_enhance_offset + valtest_point
                enhance_nid = torch.cat((val_enhance_nid, test_enhance_nid))

                print(f'Stage: {stage}, threshold {args.threshold}, confident nodes: {len(enhance_nid)} / {total_num_nodes - trainval_point}')
                val_confident_level = (predict_prob[val_enhance_nid].argmax(1) == labels[val_enhance_nid]).sum() / len(val_enhance_nid)
                print(f'\t\t val confident nodes: {len(val_enhance_nid)} / {valid_node_nums},  val confident level: {val_confident_level}')
                test_confident_level = (predict_prob[test_enhance_nid].argmax(1) == labels[test_enhance_nid]).sum() / len(test_enhance_nid)
                print(f'\t\ttest confident nodes: {len(test_enhance_nid)} / {test_node_nums}, test confident_level: {test_confident_level}')

                del train_loader
                train_batch_size = int(args.batch_size * len(self.flow.train_idx) / (len(enhance_nid) + len(self.flow.train_idx)))
                train_loader = torch.utils.data.DataLoader(
                    torch.arange(train_node_nums), batch_size=train_batch_size, shuffle=True, drop_last=False)
                enhance_batch_size = int(args.batch_size * len(enhance_nid) / (len(enhance_nid) + len(self.flow.train_idx)))
                enhance_loader = torch.utils.data.DataLoader(
                    enhance_nid, batch_size=enhance_batch_size, shuffle=True, drop_last=False)
            else:
                train_loader = torch.utils.data.DataLoader(
                    torch.arange(train_node_nums), batch_size=args.batch_size, shuffle=True, drop_last=False)

            # =======
            # labels propagate alongside the metapath
            # =======
            label_feats = {}
            if args.label_feats:
                if stage > 0:
                    label_onehot = predict_prob[sort2init].clone()
                else:
                    label_onehot = torch.zeros((num_nodes, n_classes))
                label_onehot[self.flow.train_idx] = F.one_hot(self.flow.labels[self.flow.train_idx], n_classes).float()
                self.flow.dataset.SeHGNN_g.nodes['P'].data['P'] = label_onehot
                print(f'Current num label hops = {args.num_label_hops}')
                max_hops = args.num_label_hops + 1
                self.flow.dataset.SeHGNN_g = self.hg_propagate(self.flow.dataset.SeHGNN_g, tgt_type, args.num_label_hops, max_hops, echo=False)

                keys = list(self.flow.dataset.SeHGNN_g.nodes[tgt_type].data.keys())
                print(f'Involved label keys {keys}')
                for k in keys:
                    if k == tgt_type: continue
                    label_feats[k] = self.flow.dataset.SeHGNN_g.nodes[tgt_type].data.pop(k)
                self.flow.dataset.SeHGNN_g = self.clear_hg(self.flow.dataset.SeHGNN_g, echo=False)

                for k in ['PPP', 'PAP', 'PFP', 'PPPP', 'PAPP', 'PPAP', 'PFPP', 'PPFP']:
                    if k in label_feats:
                        diag = torch.load(f'{args.dataset}_{k}_diag.pt')
                        label_feats[k] = label_feats[k] - diag.unsqueeze(-1) * label_onehot
                        assert torch.all(label_feats[k] > -1e-6)
                        print(k, torch.sum(label_feats[k] < 0), label_feats[k].min())
                label_emb = (label_feats['PPP'] + label_feats['PAP'] + label_feats['PP'] + label_feats['PFP']) / 4
            else:
                label_emb = torch.zeros((num_nodes, n_classes))

            label_feats = {k: v[init2sort] for k, v in label_feats.items()}
            label_emb = label_emb[init2sort]

            if stage == 0:
                label_feats = {}

            # =======
            # Eval loader
            # =======
            if stage > 0:
                del eval_loader
            eval_loader = []
            for batch_idx in range((num_nodes-trainval_point-1) // args.batch_size + 1):
                batch_start = batch_idx * args.batch_size + trainval_point
                batch_end = min(num_nodes, (batch_idx+1) * args.batch_size + trainval_point)

                batch_feats = {k: v[batch_start:batch_end] for k,v in feats.items()}
                batch_label_feats = {k: v[batch_start:batch_end] for k,v in label_feats.items()}
                batch_labels_emb = label_emb[batch_start:batch_end]
                eval_loader.append((batch_feats, batch_label_feats, batch_labels_emb))

            data_size = {k: v.size(-1) for k, v in feats.items()}

            # =======
            # Construct network
            # =======
            args.data_size = data_size
            args.nclass = n_classes
            args.nfeat = args.embed_size
            args.num_feats = len(feats)
            args.num_label_feats = len(label_feats)
            args.tgt_key = tgt_type

            model = build_model(self.args.model).build_model_from_args(self.args).to(self.args.device)
            if stage == args.start_stage:
                print(model)
                print("# Params:", self.get_n_params(model))

            loss_fcn = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                        weight_decay=args.weight_decay)

            best_epoch = 0
            best_val_acc = 0
            best_test_acc = 0
            count = 0

            for epoch in range(epochs):
                gc.collect()
                torch.cuda.empty_cache()
                start = time.time()
                if stage == 0:
                    loss, acc = self.run(model, train_loader, loss_fcn, optimizer, evaluator, device, feats, label_feats, labels_cuda, label_emb, scalar=scalar)
                else:
                    loss, acc = self.train_multi_stage(model, train_loader, enhance_loader, loss_fcn, optimizer, evaluator, device, feats, label_feats, labels_cuda, label_emb, predict_prob, args.gama, scalar=scalar)
                end = time.time()

                log = "Epoch {}, Time(s): {:.4f}, estimated train loss {:.4f}, acc {:.4f}\n".format(epoch, end-start, loss, acc*100)
                torch.cuda.empty_cache()

                if epoch % args.eval_every == 0:
                    with torch.no_grad():
                        model.eval()
                        raw_preds = []

                        start = time.time()
                        for batch_feats, batch_label_feats, batch_labels_emb in eval_loader:
                            batch_feats = {k: v.to(device) for k,v in batch_feats.items()}
                            batch_label_feats = {k: v.to(device) for k,v in batch_label_feats.items()}
                            batch_labels_emb = batch_labels_emb.to(device)
                            fk = {'0': batch_feats, '1': batch_label_feats, '2': batch_labels_emb}
                            raw_preds.append(model(fk).cpu())
                        raw_preds = torch.cat(raw_preds, dim=0)

                        loss_val = loss_fcn(raw_preds[:valid_node_nums], labels[trainval_point:valtest_point]).item()
                        loss_test = loss_fcn(raw_preds[valid_node_nums:valid_node_nums+test_node_nums], labels[valtest_point:total_num_nodes]).item()

                        preds = raw_preds.argmax(dim=-1)
                        val_acc = evaluator(preds[:valid_node_nums], labels[trainval_point:valtest_point])
                        test_acc = evaluator(preds[valid_node_nums:valid_node_nums+test_node_nums], labels[valtest_point:total_num_nodes])

                        end = time.time()
                        log += f'Time: {end-start}, Val loss: {loss_val}, Test loss: {loss_test}\n'
                        log += 'Val acc: {:.4f}, Test acc: {:.4f}\n'.format(val_acc*100, test_acc*100)

                    if val_acc > best_val_acc:
                        best_epoch = epoch
                        best_val_acc = val_acc
                        best_test_acc = test_acc

                        torch.save(model.state_dict(), f'{checkpt_file}_{stage}.pkl')
                        count = 0
                    else:
                        count = count + args.eval_every
                        if count >= args.patience:
                            break
                    log += "Best Epoch {},Val {:.4f}, Test {:.4f}".format(best_epoch, best_val_acc*100, best_test_acc*100)
                print(log, flush=True)

            print("Best Epoch {}, Val {:.4f}, Test {:.4f}".format(best_epoch, best_val_acc*100, best_test_acc*100))

            model.load_state_dict(torch.load(checkpt_file+f'_{stage}.pkl'))
            raw_preds = self.gen_output_torch(model, feats, label_feats, label_emb, all_loader, device)
            torch.save(raw_preds, checkpt_file+f'_{stage}.pt')

    def set_random_seed(self, seed=0):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    def get_n_params(self, model):
        pp = 0
        for p in list(model.parameters()):
            nn = 1
            for s in list(p.size()):
                nn = nn * s
            pp += nn
        return pp

    def hg_propagate(self, new_g, tgt_type, num_hops, max_hops, echo=False):
        for hop in range(1, max_hops):
            for etype in new_g.etypes:
                stype, _, dtype = new_g.to_canonical_etype(etype)
                for k in list(new_g.nodes[stype].data.keys()):
                    if len(k) == hop:
                        current_dst_name = f'{dtype}{k}'
                        if (hop == num_hops and dtype != tgt_type) or (hop > num_hops):
                            continue
                        if echo: print(k, etype, current_dst_name)
                        new_g[etype].update_all(
                            fn.copy_u(k, 'm'),
                            fn.mean('m', current_dst_name), etype=etype)

            # remove no-use items
            for ntype in new_g.ntypes:
                if ntype == tgt_type: continue
                removes = []
                for k in new_g.nodes[ntype].data.keys():
                    if len(k) <= hop:
                        removes.append(k)
                for k in removes:
                    new_g.nodes[ntype].data.pop(k)
                if echo and len(removes): print('remove', removes)
            gc.collect()

            if echo: print(f'-- hop={hop} ---')
            for ntype in new_g.ntypes:
                for k, v in new_g.nodes[ntype].data.items():
                    if echo: print(f'{ntype} {k} {v.shape}')
            if echo: print(f'------\n')

        return new_g

    def clear_hg(self, new_g, echo=False):
        if echo: print('Remove keys left after propagation')
        for ntype in new_g.ntypes:
            keys = list(new_g.nodes[ntype].data.keys())
            if len(keys):
                if echo: print(ntype, keys)
                for k in keys:
                    new_g.nodes[ntype].data.pop(k)
        return new_g

    def run(self, model, train_loader, loss_fcn, optimizer, evaluator, device,
              feats, label_feats, labels_cuda, label_emb, mask=None, scalar=None):
        model.train()
        total_loss = 0
        iter_num = 0
        y_true, y_pred = [], []

        for batch in train_loader:
            batch_feats = {k: x[batch].to(device) for k, x in feats.items()}
            batch_labels_feats = {k: x[batch].to(device) for k, x in label_feats.items()}
            # if mask is not None:
            #     batch_mask = {k: x[batch].to(device) for k, x in mask.items()}
            # else:
            #     batch_mask = None
            batch_label_emb = label_emb[batch].to(device)
            batch_y = labels_cuda[batch]

            optimizer.zero_grad()
            if scalar is not None:
                with torch.cuda.amp.autocast():
                    fk = {'0': batch_feats, '1' :batch_labels_feats, '2': batch_label_emb}
                    output_att = model(fk)
                    if isinstance(loss_fcn, nn.BCELoss):
                        output_att = torch.sigmoid(output_att)
                    loss_train = loss_fcn(output_att, batch_y)
                scalar.scale(loss_train).backward()
                scalar.step(optimizer)
                scalar.update()
            else:
                fk = {'0': batch_feats, '1': batch_labels_feats,'2': batch_label_emb}
                output_att = model(fk)
                if isinstance(loss_fcn, nn.BCELoss):
                    output_att = torch.sigmoid(output_att)
                L1 = loss_fcn(output_att, batch_y)
                loss_train = L1
                loss_train.backward()
                optimizer.step()

            y_true.append(batch_y.cpu().to(torch.long))
            if isinstance(loss_fcn, nn.BCELoss):
                y_pred.append((output_att.data.cpu() > 0).int())
            else:
                y_pred.append(output_att.argmax(dim=-1, keepdim=True).cpu())
            total_loss += loss_train.item()
            iter_num += 1
        loss = total_loss / iter_num
        acc = evaluator(torch.cat(y_true, dim=0), torch.cat(y_pred, dim=0))
        return loss, acc

    def train_multi_stage(self, model, train_loader, enhance_loader, loss_fcn, optimizer, evaluator, device,
                          feats, label_feats, labels, label_emb, predict_prob, gama, scalar=None):
        model.train()
        loss_fcn = nn.CrossEntropyLoss()
        y_true, y_pred = [], []
        total_loss = 0
        loss_l1, loss_l2 = 0., 0.
        iter_num = 0
        for idx_1, idx_2 in zip(train_loader, enhance_loader):
            idx = torch.cat((idx_1, idx_2), dim=0)
            L1_ratio = len(idx_1) * 1.0 / (len(idx_1) + len(idx_2))
            L2_ratio = len(idx_2) * 1.0 / (len(idx_1) + len(idx_2))

            batch_feats = {k: x[idx].to(device) for k, x in feats.items()}
            batch_labels_feats = {k: x[idx].to(device) for k, x in label_feats.items()}
            batch_label_emb = label_emb[idx].to(device)
            y = labels[idx_1].to(torch.long).to(device)
            extra_weight, extra_y = predict_prob[idx_2].max(dim=1)
            extra_weight = extra_weight.to(device)
            extra_y = extra_y.to(device)

            optimizer.zero_grad()
            if scalar is not None:
                with torch.cuda.amp.autocast():
                    fk = {'0': batch_feats, '1': batch_labels_feats, '2': batch_label_emb}
                    output_att = model(fk)
                    L1 = loss_fcn(output_att[:len(idx_1)], y)
                    L2 = F.cross_entropy(output_att[len(idx_1):], extra_y, reduction='none')
                    L2 = (L2 * extra_weight).sum() / len(idx_2)
                    loss_train = L1_ratio * L1 + gama * L2_ratio * L2
                scalar.scale(loss_train).backward()
                scalar.step(optimizer)
                scalar.update()
            else:
                while True:
                    print("Yy")
                fk = {'0': batch_feats, '1': label_emb[idx].to(device)}
                output_att = model(fk)
                L1 = loss_fcn(output_att[:len(idx_1)], y)
                L2 = F.cross_entropy(output_att[len(idx_1):], extra_y, reduction='none')
                L2 = (L2 * extra_weight).sum() / len(idx_2)
                loss_train = L1_ratio * L1 + gama * L2_ratio * L2
                loss_train.backward()
                optimizer.step()

            y_true.append(labels[idx_1].to(torch.long))
            y_pred.append(output_att[:len(idx_1)].argmax(dim=-1, keepdim=True).cpu())
            total_loss += loss_train.item()
            loss_l1 += L1.item()
            loss_l2 += L2.item()
            iter_num += 1

        print(loss_l1 / iter_num, loss_l2 / iter_num)
        loss = total_loss / iter_num
        approx_acc = evaluator(torch.cat(y_true, dim=0), torch.cat(y_pred, dim=0))
        return loss, approx_acc

    @torch.no_grad()
    def gen_output_torch(self, model, feats, label_feats, label_emb, test_loader, device):
        model.eval()
        preds = []
        for batch in tqdm(test_loader):
            batch_feats = {k: x[batch].to(device) for k, x in feats.items()}
            batch_labels_feats = {k: x[batch].to(device) for k, x in label_feats.items()}
            batch_label_emb = label_emb[batch].to(device)
            preds.append(model(batch_feats, batch_labels_feats, batch_label_emb).cpu())
        preds = torch.cat(preds, dim=0)
        return preds

    def get_ogb_evaluator(self, dataset):
        evaluator = Evaluator(name=dataset)
        return lambda preds, labels: evaluator.eval({
            "y_true": labels.view(-1, 1),
            "y_pred": preds.view(-1, 1),
        })["acc"]


