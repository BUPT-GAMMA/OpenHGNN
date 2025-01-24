import torch
import dgl
import numpy as np
import os
from dgl.data.utils import download, _get_dgl_url
from scipy import io as sio
from tqdm import tqdm
import scipy.sparse as sp
from sklearn.metrics import f1_score
import random
import pickle as pkl
import copy
import warnings
import urllib.request

from . import BaseFlow, register_flow
from ..models import build_model
from ..layers.HeteroLinear import HeteroFeature
from ..utils import EarlyStopping
from ..utils.utils import get_nodes_dict
from ..sampler import HANSampler


@register_flow("RoHe_trainer")
class RoHeTrainer(BaseFlow):
    r"""
    HAN node classification flow.
    """

    def __init__(self, args):  # 传递对象（如列表、字典、类实例等）作为参数时，实际上传递的是对象的引用

        super(RoHeTrainer, self).__init__(args)
        self.args.category = self.task.dataset.category
        self.category = self.args.category

        self.num_classes = self.task.dataset.num_classes

        if not hasattr(self.task.dataset, 'out_dim') or self.args.out_dim != self.num_classes:
            self.logger.info('[NC Specific] Modify the out_dim with num_classes')
            self.args.out_dim = self.num_classes
        self.args.out_node_type = [self.category]

        self.hete_adjs = self.load_acm()
        self.args.settings = self.setup_settings()

        self.raw = True
        self.raw_model = build_model('HAN').build_model_from_args(self.args, self.hg).to(self.device)
        self.model = build_model(self.model).build_model_from_args(self.args, self.hg).to(self.device)

        self.optimizer = self.candidate_optimizer[args.optimizer](self.model.parameters(),
                                                                  lr=args.lr, weight_decay=args.weight_decay)

        self.train_idx, self.valid_idx, self.test_idx = self.task.get_split()

        if self.args.prediction_flag:
            self.pred_idx = self.task.dataset.pred_idx

        self.labels = self.task.get_labels().to(self.device)

        if self.args.mini_batch_flag:
            sampler = HANSampler(g=self.hg, seed_ntypes=[self.category], meta_paths_dict=self.args.meta_paths_dict,
                                 num_neighbors=20)
            if self.train_idx is not None:
                self.train_loader = dgl.dataloading.DataLoader(
                    self.hg, {self.category: self.train_idx.to(self.device)}, sampler,
                    batch_size=self.args.batch_size, device=self.device, shuffle=True)
            if self.valid_idx is not None:
                self.val_loader = dgl.dataloading.DataLoader(
                    self.hg, {self.category: self.valid_idx.to(self.device)}, sampler,
                    batch_size=self.args.batch_size, device=self.device, shuffle=True)
            if self.args.test_flag:
                self.test_loader = dgl.dataloading.DataLoader(
                    self.hg, {self.category: self.test_idx.to(self.device)}, sampler,
                    batch_size=self.args.batch_size, device=self.device, shuffle=True)
            if self.args.prediction_flag:
                self.pred_loader = dgl.dataloading.DataLoader(
                    self.hg, {self.category: self.pred_idx.to(self.device)}, sampler,
                    batch_size=self.args.batch_size, device=self.device, shuffle=True)

    def preprocess(self):
        super(RoHeTrainer, self).preprocess()

    def train(self):
        self.preprocess()  # 见*.md
        stopper = EarlyStopping(self.args.patience, self._checkpoint)
        epoch_iter = tqdm(range(self.max_epoch))
        for epoch in epoch_iter:
            if self.args.mini_batch_flag:
                train_loss = self._mini_train_step()
            else:
                train_loss = self._full_train_step()
            if epoch % self.evaluate_interval == 0:
                modes = ['train', 'valid']
                if self.args.test_flag:
                    modes = modes + ['test']
                if self.args.mini_batch_flag and hasattr(self, 'val_loader'):
                    metric_dict, losses = self._mini_test_step(modes=modes)
                    # train_score, train_loss = self._mini_test_step(modes='train')
                    # val_score, val_loss = self._mini_test_step(modes='valid')
                else:
                    metric_dict, losses = self._full_test_step(modes=modes)
                val_loss = losses['valid']
                Macro_f1 = metric_dict['valid']['Macro_f1']
                Micro_f1 = metric_dict['valid']['Micro_f1']
                score = (Macro_f1 + Micro_f1) / 2
                self.logger.train_info(f"Epoch: {epoch}, Train loss: {train_loss:.4f}, Valid loss: {val_loss:.4f}. "
                                       + self.logger.metric2str(metric_dict))
                early_stop = stopper.step(val_loss, score, self.model)
                if early_stop:
                    self.logger.train_info('Early Stop!\tEpoch:' + str(epoch))
                    break

        stopper.load_model(self.model)
        input_feature = HeteroFeature(self.hg.ndata['h'], get_nodes_dict(self.hg), self.args.hidden_dim, act=None)
        self.raw_model.add_module('input_feature', input_feature)
        try:
            save_path = './openhgnn/output/HAN/HAN_acm_han_raw_node_classification.pt'
            self.raw_model.load_state_dict(torch.load(save_path, map_location=self.device))
            self.raw_model.to(self.device)
        except FileNotFoundError:
            self.raw = False
            print(
                "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
                "\tFailed to import raw-HAN parameters. \n "
                "\tThe results output will no longer include the comparison of raw-HAN. \n"
                "\tPlease try running [model: HAN, task: node_classification, dataset: acm_han_raw] first \n"
                "\t\tto generate the checkpoint file HAN_acm_han_raw_node_classificationtest.pt.\n"
                "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            )

        if self.args.prediction_flag:
            if self.args.mini_batch_flag and hasattr(self, 'val_loader'):
                indices, y_predicts = self._mini_prediction_step()
            else:
                y_predicts = self._full_prediction_step()
                indices = torch.arange(self.hg.num_nodes(self.category))
            return indices, y_predicts

        if self.args.test_flag:
            if self.args.dataset[:4] == 'HGBn':
                # save results for HGBn
                if self.args.mini_batch_flag and hasattr(self, 'val_loader'):
                    metric_dict, val_loss = self._mini_test_step(modes=['valid'])
                else:
                    metric_dict, val_loss = self._full_test_step(modes=['valid'])
                self.logger.train_info('[Test Info]' + self.logger.metric2str(metric_dict))
                self.model.eval()
                with torch.no_grad():
                    h_dict = self.model.input_feature()
                    logits = self.model(self.hg, h_dict)[self.category]
                    self.task.dataset.save_results(logits=logits, file_path=self.args.HGB_results_path)
                return dict(metric=metric_dict, epoch=epoch)
            if self.args.mini_batch_flag and hasattr(self, 'val_loader'):
                metric_dict, _ = self._mini_test_step(modes=['valid', 'test'])
            else:
                metric_dict, _ = self._full_test_step(modes=['valid', 'test'])
            self.logger.train_info('[Test Info]' + self.logger.metric2str(metric_dict))
            self.generate_attacks_metric_dict(self.raw)
            return dict(metric=metric_dict, epoch=epoch)

    def _full_train_step(self):
        self.model.train()
        h_dict = self.model.input_feature()
        logits = self.model(self.hg, h_dict)[self.category]
        loss = self.loss_fn(logits[self.train_idx], self.labels[self.train_idx])
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _mini_train_step(self, ):
        self.model.train()
        loss_all = 0.0
        loader_tqdm = tqdm(self.train_loader, ncols=120)
        for i, (ntype_mp_name_input_nodes_dict, seeds, ntype_mp_name_block_dict) in enumerate(loader_tqdm):
            seeds = seeds[self.category]
            mp_name_input_nodes_dict = ntype_mp_name_input_nodes_dict[self.category]
            emb_dict = {}
            for meta_path_name, input_nodes in mp_name_input_nodes_dict.items():
                emb_dict[meta_path_name] = self.model.input_feature.forward_nodes({self.category: input_nodes})
            emb_dict = {self.category: emb_dict}
            logits = self.model(ntype_mp_name_block_dict, emb_dict)[self.category]
            lbl = self.labels[seeds].to(self.device)
            loss = self.loss_fn(logits, lbl)
            loss_all += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return loss_all / (i + 1)

    def _full_test_step(self, modes, logits=None):
        """
        Parameters
        ----------
        mode: list[str]
            `train`, 'test', 'valid' are optional in list.
        logits: dict[str, th.Tensor]
            given logits, default `None`.

        Returns
        -------
        metric_dict: dict[str, float]
            score of evaluation metric
        info: dict[str, str]
            evaluation information
        loss: dict[str, float]
            the loss item
        """
        self.model.eval()
        with torch.no_grad():
            h_dict = self.model.input_feature()
            h_dict = {k: e.to(self.device) for k, e in h_dict.items()}
            logits = logits if logits else self.model(self.hg, h_dict)[self.category]
            masks = {}
            for mode in modes:
                if mode == "train":
                    masks[mode] = self.train_idx
                elif mode == "valid":
                    masks[mode] = self.valid_idx
                elif mode == "test":
                    masks[mode] = self.test_idx

            metric_dict = {key: self.task.evaluate(logits, mode=key) for key in masks}
            loss_dict = {key: self.loss_fn(logits[mask], self.labels[mask]).item() for key, mask in masks.items()}
            return metric_dict, loss_dict

    def _mini_test_step(self, modes):
        self.model.eval()
        with torch.no_grad():
            metric_dict = {}
            loss_dict = {}
            loss_all = 0.0
            for mode in modes:
                if mode == 'train':
                    loader_tqdm = tqdm(self.train_loader, ncols=120)
                elif mode == 'valid':
                    loader_tqdm = tqdm(self.val_loader, ncols=120)
                elif mode == 'test':
                    loader_tqdm = tqdm(self.test_loader, ncols=120)
                y_trues = []
                y_predicts = []

                for i, (ntype_mp_name_input_nodes_dict, seeds, ntype_mp_name_block_dict) in enumerate(loader_tqdm):
                    seeds = seeds[self.category]
                    mp_name_input_nodes_dict = ntype_mp_name_input_nodes_dict[self.category]
                    emb_dict = {}
                    for meta_path_name, input_nodes in mp_name_input_nodes_dict.items():
                        emb_dict[meta_path_name] = self.model.input_feature.forward_nodes({self.category: input_nodes})
                    emb_dict = {self.category: emb_dict}
                    logits = self.model(ntype_mp_name_block_dict, emb_dict)[self.category]
                    lbl = self.labels[seeds].to(self.device)
                    loss = self.loss_fn(logits, lbl)
                    loss_all += loss.item()
                    y_trues.append(lbl.detach().cpu())
                    y_predicts.append(logits.detach().cpu())
                loss_all /= (i + 1)
                y_trues = torch.cat(y_trues, dim=0)
                y_predicts = torch.cat(y_predicts, dim=0)
                evaluator = self.task.get_evaluator(name='f1')
                metric_dict[mode] = evaluator(y_trues, y_predicts.argmax(dim=1).to('cpu'))
                loss_dict[mode] = loss
        return metric_dict, loss_dict

    def _full_prediction_step(self):
        self.model.eval()
        with torch.no_grad():
            h_dict = self.model.input_feature()
            h_dict = {k: e.to(self.device) for k, e in h_dict.items()}
            logits = self.model(self.hg, h_dict)[self.category]
            return logits

    def _mini_prediction_step(self):
        self.model.eval()
        with torch.no_grad():
            loader_tqdm = tqdm(self.pred_loader, ncols=120)
            indices = []
            y_predicts = []
            for i, (input_nodes_dict, seeds, block_dict) in enumerate(loader_tqdm):
                seeds = seeds[self.category]
                emb_dict = {}
                for meta_path_name, input_nodes in input_nodes_dict.items():
                    emb_dict[meta_path_name] = self.model.input_feature.forward_nodes(input_nodes)
                logits = self.model(block_dict, emb_dict)[self.category]
                indices.append(seeds.detach().cpu())
                y_predicts.append(logits.detach().cpu())
            indices = torch.cat(indices, dim=0)
            y_predicts = torch.cat(y_predicts, dim=0)
        return indices, y_predicts

    def get_transition(self, given_hete_adjs, metapath_info):
        # transition
        hete_adj_dict_tmp = {}
        for key in given_hete_adjs.keys():
            deg = given_hete_adjs[key].sum(1)
            hete_adj_dict_tmp[key] = given_hete_adjs[key] / (np.where(deg > 0, deg, 1))  # make sure deg>0
        homo_adj_list = []
        for i in range(len(metapath_info)):
            adj = hete_adj_dict_tmp[metapath_info[i][0]]
            for etype in metapath_info[i][1:]:
                adj = adj.dot(hete_adj_dict_tmp[etype])
            homo_adj_list.append(sp.csc_matrix(adj))
        return homo_adj_list

    def setup_settings(self):
        settings_pap = {'T': 2}  # acm
        settings_psp = {'T': 5}  # acm
        settings = [settings_pap, settings_psp]
        meta_paths_dict = {'acm_han_raw': [['pa', 'ap'], ['pf', 'fp']]}
        trans_adj_list = self.get_transition(self.hete_adjs, meta_paths_dict['acm_han_raw'])
        for i in range(len(trans_adj_list)):
            settings[i]['device'] = self.device
            settings[i]['TransM'] = trans_adj_list[i]
        return settings

    def get_hg(self, given_adj_dict):
        hg_new = dgl.heterograph({
            ('paper', 'pa', 'author'): given_adj_dict['pa'].nonzero(),
            ('author', 'ap', 'paper'): given_adj_dict['ap'].nonzero(),
            ('paper', 'pf', 'field'): given_adj_dict['pf'].nonzero(),
            ('field', 'fp', 'paper'): given_adj_dict['fp'].nonzero(),
        })
        return hg_new

    def score(self, logits, labels):
        _, indices = torch.max(logits, dim=1)
        prediction = indices.long().cpu().numpy()
        labels = labels.cpu().numpy()
        accuracy = (prediction == labels).sum() / len(prediction)
        micro_f1 = f1_score(labels, prediction, average='micro')
        macro_f1 = f1_score(labels, prediction, average='macro')
        return accuracy, micro_f1, macro_f1

    def load_acm(self, remove_self_loop=False):
        assert not remove_self_loop
        url = 'dataset/ACM.mat'
        data_path = './openhgnn/dataset/ACM.mat'
        if not os.path.exists(data_path):
            download(_get_dgl_url(url), path=data_path)

        data = sio.loadmat(data_path)
        p_vs_l = data['PvsL']  # paper-field
        p_vs_a = data['PvsA']  # paper-author
        p_vs_t = data['PvsT']  # paper-term, bag of words
        p_vs_c = data['PvsC']  # paper-conference, labels come from that

        conf_ids = [0, 1, 9, 10, 13]

        p_vs_c_filter = p_vs_c[:, conf_ids]
        p_selected = (p_vs_c_filter.sum(1) != 0).A1.nonzero()[0]
        p_vs_l = p_vs_l[p_selected]
        p_vs_a = p_vs_a[p_selected]
        hete_adjs = {'pa': p_vs_a, 'ap': p_vs_a.T,
                     'pf': p_vs_l, 'fp': p_vs_l.T}
        return hete_adjs

    def generate_attacks_metric_dict(self, raw=False):
        warnings.filterwarnings('ignore')
        tar_nodes_num = 100
        tar_idx = sorted(random.sample(range(self.hg.num_nodes(self.category)), tar_nodes_num))
        if raw:
            print(
                "Note that:\n"
                "\t Raw-HAN is only for better comparison with RoHe-HAN. \n"
                "\t(Raw-HAN is not necessarily the best performance, don't care about the value,"
                "and pay more attention to the trend of changes before and after)"
            )
        with torch.no_grad():
            if raw:
                h_dict = self.raw_model.input_feature()
                logits = self.raw_model(self.hg, h_dict)[self.category]
                _, micro_f1, macro_f1 = self.score(logits[tar_idx], self.labels[tar_idx])
                print(f"Raw-HAN in raw data:  Micro-F1: {micro_f1:.3f} Macro-F1: {macro_f1:.3f}")
            h_dict = self.model.input_feature()
            logits = self.model(self.hg, h_dict)[self.category]
            _, micro_f1, macro_f1 = self.score(logits[tar_idx], self.labels[tar_idx])
            print(f"RoHe-HAN in raw data:  Micro-F1: {micro_f1:.3f} Macro-F1: {macro_f1:.3f}")

        n_perturbation = [1, 3, 5]
        adv_filename = 'adv_acm_pap_pa_' + str(n_perturbation[0]) + '.pkl'
        file_local_path = './openhgnn/output/RoHe/' + adv_filename
        # download adversarial attacks
        if not os.path.exists(file_local_path):
            url = "https://s3.cn-north-1.amazonaws.com.cn/dgl-data/dataset/" + adv_filename
            urllib.request.urlretrieve(url, file_local_path)
        # 1.load adversarial attacks for each target node
        with open(file_local_path, 'rb') as f:
            modified_opt = pkl.load(f)
        # 2.attack
        logits_adv = []
        labels_adv = []
        if raw:
            raw_logits_adv = []
            raw_labels_adv = []
        modified_opt_iter = tqdm(modified_opt)
        for items in modified_opt_iter:
            # 2.1 init
            target_node = items[0]
            del_list = items[2]
            add_list = items[3]
            if target_node not in tar_idx:
                continue
            # 2.2 attack adjs
            mod_hete_adj_dict = copy.deepcopy(self.hete_adjs)
            for edge in del_list:
                mod_hete_adj_dict['pa'][edge[0], edge[1]] = 0
                mod_hete_adj_dict['ap'][edge[1], edge[0]] = 0
            for edge in add_list:
                mod_hete_adj_dict['pa'][edge[0], edge[1]] = 1
                mod_hete_adj_dict['ap'][edge[1], edge[0]] = 1
            meta_paths_dict = {'acm_han_raw': [['pa', 'ap'], ['pf', 'fp']]}

            trans_adj_list = self.get_transition(mod_hete_adj_dict, meta_paths_dict['acm_han_raw'])
            for i in range(len(self.model.mod_dict['paper'].layers)):
                self.model.mod_dict['paper'].layers[i].model.mods['PAP'].settings['TransM'] = trans_adj_list[0]
                self.model.mod_dict['paper'].layers[i].model.mods['PFP'].settings['TransM'] = trans_adj_list[1]

            self.hg = self.get_hg(mod_hete_adj_dict).to(self.device)
            # 2.3 run model
            with torch.no_grad():
                h_dict = self.model.input_feature()
                logits = self.model(self.hg, h_dict)[self.category]
                if raw:
                    raw_logits = self.raw_model(self.hg, h_dict)[self.category]
            # 2.4 evaluate
            logits_adv.append(logits[np.array([[target_node]])])
            labels_adv.append(self.labels[np.array([[target_node]])])
            if raw:
                raw_logits_adv.append(raw_logits[np.array([[target_node]])])
                raw_labels_adv.append(self.labels[np.array([[target_node]])])

        if raw:
            raw_logits_adv = torch.cat(raw_logits_adv, 0)
            raw_labels_adv = torch.cat(raw_labels_adv)
            _, raw_tar_micro_f1_atk, raw_tar_macro_f1_atk = self.score(raw_logits_adv, raw_labels_adv)
            print(
                f"Raw-HAN in attacked data:  Micro-F1: {raw_tar_micro_f1_atk:.3f} Macro-F1: {raw_tar_macro_f1_atk:.3f}")

        logits_adv = torch.cat(logits_adv, 0)
        labels_adv = torch.cat(labels_adv)
        _, tar_micro_f1_atk, tar_macro_f1_atk = self.score(logits_adv, labels_adv)
        print(f"RoHe-HAN in attacked data:  Micro-F1: {tar_micro_f1_atk:.3f} Macro-F1: {tar_macro_f1_atk:.3f}")
