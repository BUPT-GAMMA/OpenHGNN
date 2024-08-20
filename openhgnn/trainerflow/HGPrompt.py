from openhgnn.trainerflow import register_flow
import dgl
import torch.nn.functional as F
from tqdm import tqdm
import torch
from abc import ABC
import sys
import copy
import torch as th
import scipy.sparse as sp
import time
import argparse
import random
import numpy as np
import os
from dgl.nn.pytorch import GraphConv
import itertools
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score
from collections import Counter, defaultdict
from sklearn.metrics import f1_score


from ..dataset import build_dataset
from ..models.HGPrompt import *


global_dir_path = None
global_device = torch.device('cuda:0')

@register_flow('HGPrompt_trainer')
class HGPrompt_trainer(ABC):
    def __init__(self, args):
        super(HGPrompt_trainer, self).__init__()  # 父类ABC 初始化无事发生
        self.args = args
        #   数据集  dataset: hgprompt_acm_dblp   ,  _dataset是ACM，DBLP等
        self.dataset = build_dataset(args.dataset, 'node_classification',  # 数据集名称  和  任务名称  是必要参数，其他都是 额外 关键字参数
                                     args = args , logger = args.logger)  
        global global_dir_path
        global_dir_path = os.path.join(self.dataset.base_dir,'hgprompt_acm_dblp')
        pass

    def train(self):
        pretrain_DBLP(self.args)
        run_model_DBLP_downstream(self.args)
        pass



##################  pytorchtools.py文件

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience, verbose=False, delta=0, save_path='checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_path = save_path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score - self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.save_path)
        self.val_loss_min = val_loss

###################

################### data_loader.py文件


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class data_loader:
    def __init__(self, path):
        self.path = path
        self.nodes = self.load_nodes()
        self.links = self.load_links()
        self.labels_train = self.load_labels('label.dat')
        self.labels_test = self.load_labels('label.dat.test')

    def get_sub_graph(self, node_types_tokeep):
        """
        node_types_tokeep is a list or set of node types that you want to keep in the sub-graph
        We only support whole type sub-graph for now.
        This is an in-place update function!
        return: old node type id to new node type id dict, old edge type id to new edge type id dict
        """
        keep = set(node_types_tokeep)
        new_node_type = 0
        new_node_id = 0
        new_nodes = {'total': 0, 'count': Counter(), 'attr': {}, 'shift': {}}
        new_links = {'total': 0, 'count': Counter(), 'meta': {}, 'data': defaultdict(list)}
        new_labels_train = {'num_classes': 0, 'total': 0, 'count': Counter(), 'data': None, 'mask': None}
        new_labels_test = {'num_classes': 0, 'total': 0, 'count': Counter(), 'data': None, 'mask': None}
        old_nt2new_nt = {}
        old_idx = []
        for node_type in self.nodes['count']:
            if node_type in keep:
                nt = node_type
                nnt = new_node_type
                old_nt2new_nt[nt] = nnt
                cnt = self.nodes['count'][nt]
                new_nodes['total'] += cnt
                new_nodes['count'][nnt] = cnt
                new_nodes['attr'][nnt] = self.nodes['attr'][nt]
                new_nodes['shift'][nnt] = new_node_id
                beg = self.nodes['shift'][nt]
                old_idx.extend(range(beg, beg + cnt))

                cnt_label_train = self.labels_train['count'][nt]
                new_labels_train['count'][nnt] = cnt_label_train
                new_labels_train['total'] += cnt_label_train
                cnt_label_test = self.labels_test['count'][nt]
                new_labels_test['count'][nnt] = cnt_label_test
                new_labels_test['total'] += cnt_label_test

                new_node_type += 1
                new_node_id += cnt

        new_labels_train['num_classes'] = self.labels_train['num_classes']
        new_labels_test['num_classes'] = self.labels_test['num_classes']
        for k in ['data', 'mask']:
            new_labels_train[k] = self.labels_train[k][old_idx]
            new_labels_test[k] = self.labels_test[k][old_idx]

        old_et2new_et = {}
        new_edge_type = 0
        for edge_type in self.links['count']:
            h, t = self.links['meta'][edge_type]
            if h in keep and t in keep:
                et = edge_type
                net = new_edge_type
                old_et2new_et[et] = net
                new_links['total'] += self.links['count'][et]
                new_links['count'][net] = self.links['count'][et]
                new_links['meta'][net] = tuple(map(lambda x: old_nt2new_nt[x], self.links['meta'][et]))
                new_links['data'][net] = self.links['data'][et][old_idx][:, old_idx]
                new_edge_type += 1

        self.nodes = new_nodes
        self.links = new_links
        self.labels_train = new_labels_train
        self.labels_test = new_labels_test
        return old_nt2new_nt, old_et2new_et

    def get_meta_path(self, meta=[]):
        """
        Get meta path matrix
            meta is a list of edge types (also can be denoted by a pair of node types)
            return a sparse matrix with shape [node_num, node_num]
        """
        ini = sp.eye(self.nodes['total'])
        meta = [self.get_edge_type(x) for x in meta]
        for x in meta:
            ini = ini.dot(self.links['data'][x]) if x >= 0 else ini.dot(self.links['data'][-x - 1].T)
        return ini

    def dfs(self, now, meta, meta_dict):
        if len(meta) == 0:
            meta_dict[now[0]].append(now)
            return
        th_mat = self.links['data'][meta[0]] if meta[0] >= 0 else self.links['data'][-meta[0] - 1].T
        th_node = now[-1]
        for col in th_mat[th_node].nonzero()[1]:
            self.dfs(now + [col], meta[1:], meta_dict)

    def get_full_meta_path(self, meta=[], symmetric=False):
        """
        Get full meta path for each node
            meta is a list of edge types (also can be denoted by a pair of node types)
            return a dict of list[list] (key is node_id)
        """
        meta = [self.get_edge_type(x) for x in meta]
        if len(meta) == 1:
            meta_dict = {}
            start_node_type = self.links['meta'][meta[0]][0] if meta[0] >= 0 else self.links['meta'][-meta[0] - 1][1]
            for i in range(self.nodes['shift'][start_node_type],
                           self.nodes['shift'][start_node_type] + self.nodes['count'][start_node_type]):
                meta_dict[i] = []
                self.dfs([i], meta, meta_dict)
        else:
            meta_dict1 = {}
            meta_dict2 = {}
            mid = len(meta) // 2
            meta1 = meta[:mid]
            meta2 = meta[mid:]
            start_node_type = self.links['meta'][meta1[0]][0] if meta1[0] >= 0 else self.links['meta'][-meta1[0] - 1][1]
            for i in range(self.nodes['shift'][start_node_type],
                           self.nodes['shift'][start_node_type] + self.nodes['count'][start_node_type]):
                meta_dict1[i] = []
                self.dfs([i], meta1, meta_dict1)
            start_node_type = self.links['meta'][meta2[0]][0] if meta2[0] >= 0 else self.links['meta'][-meta2[0] - 1][1]
            for i in range(self.nodes['shift'][start_node_type],
                           self.nodes['shift'][start_node_type] + self.nodes['count'][start_node_type]):
                meta_dict2[i] = []
            if symmetric:
                for k in meta_dict1:
                    paths = meta_dict1[k]
                    for x in paths:
                        meta_dict2[x[-1]].append(list(reversed(x)))
            else:
                for i in range(self.nodes['shift'][start_node_type],
                               self.nodes['shift'][start_node_type] + self.nodes['count'][start_node_type]):
                    self.dfs([i], meta2, meta_dict2)
            meta_dict = {}
            start_node_type = self.links['meta'][meta1[0]][0] if meta1[0] >= 0 else self.links['meta'][-meta1[0] - 1][1]
            for i in range(self.nodes['shift'][start_node_type],
                           self.nodes['shift'][start_node_type] + self.nodes['count'][start_node_type]):
                meta_dict[i] = []
                for beg in meta_dict1[i]:
                    for end in meta_dict2[beg[-1]]:
                        meta_dict[i].append(beg + end[1:])
        return meta_dict

    def gen_file_for_evaluate(self, test_idx, label, file_name, mode='bi'):
        if test_idx.shape[0] != label.shape[0]:
            return
        if mode == 'multi':
            multi_label = []
            for i in range(label.shape[0]):
                label_list = [str(j) for j in range(label[i].shape[0]) if label[i][j] == 1]
                multi_label.append(','.join(label_list))
            label = multi_label
        elif mode == 'bi':
            label = np.array(label)
        else:
            return
        with open(file_name, "w") as f:
            for nid, l in zip(test_idx, label):
                f.write(f"{nid}\t\t{self.get_node_type(nid)}\t{l}\n")

    def evaluate(self, pred):
        print(
            f"{bcolors.WARNING}Warning: If you want to obtain test score, please submit online on biendata.{bcolors.ENDC}")
        y_true = self.labels_test['data'][self.labels_test['mask']]
        micro = f1_score(y_true, pred, average='micro')
        macro = f1_score(y_true, pred, average='macro')
        result = {
            'micro-f1': micro,
            'macro-f1': macro
        }
        return result

    def load_labels(self, name):
        """
        return labels dict
            num_classes: total number of labels
            total: total number of labeled data
            count: number of labeled data for each node type
            data: a numpy matrix with shape (self.nodes['total'], self.labels['num_classes'])
            mask: to indicate if that node is labeled, if False, that line of data is masked
        """
        labels = {'num_classes': 0, 'total': 0, 'count': Counter(), 'data': None, 'mask': None}
        nc = 0
        mask = np.zeros(self.nodes['total'], dtype=bool)
        data = [None for i in range(self.nodes['total'])]
        with open(os.path.join(self.path, name), 'r', encoding='utf-8') as f:
            for line in f:
                th = line.split('\t')
                node_id, node_name, node_type, node_label = int(th[0]), th[1], int(th[2]), list(
                    map(int, th[3].split(',')))
                for label in node_label:
                    nc = max(nc, label + 1)
                mask[node_id] = True
                data[node_id] = node_label
                labels['count'][node_type] += 1
                labels['total'] += 1
        labels['num_classes'] = nc
        new_data = np.zeros((self.nodes['total'], labels['num_classes']), dtype=int)
        for i, x in enumerate(data):
            if x is not None:
                for j in x:
                    new_data[i, j] = 1
        labels['data'] = new_data
        labels['mask'] = mask
        return labels

    def get_node_type(self, node_id):
        for i in range(len(self.nodes['shift'])):
            if node_id < self.nodes['shift'][i] + self.nodes['count'][i]:
                return i

    def get_edge_type(self, info):
        if type(info) is int or len(info) == 1:
            return info
        for i in range(len(self.links['meta'])):
            if self.links['meta'][i] == info:
                return i
        info = (info[1], info[0])
        for i in range(len(self.links['meta'])):
            if self.links['meta'][i] == info:
                return -i - 1
        raise Exception('No available edge type')

    def get_edge_info(self, edge_id):
        return self.links['meta'][edge_id]

    def list_to_sp_mat(self, li):
        data = [x[2] for x in li]
        i = [x[0] for x in li]
        j = [x[1] for x in li]
        return sp.coo_matrix((data, (i, j)), shape=(self.nodes['total'], self.nodes['total'])).tocsr()

    def load_links(self):
        """
        return links dict
            total: total number of links
            count: a dict of int, number of links for each type
            meta: a dict of tuple, explaining the link type is from what type of node to what type of node
            data: a dict of sparse matrices, each link type with one matrix. Shapes are all (nodes['total'], nodes['total'])
        """
        links = {'total': 0, 'count': Counter(), 'meta': {}, 'data': defaultdict(list)}
        with open(os.path.join(self.path, 'link.dat'), 'r', encoding='utf-8') as f:
            for line in f:
                th = line.split('\t')
                h_id, t_id, r_id, link_weight = int(th[0]), int(th[1]), int(th[2]), float(th[3])
                if r_id not in links['meta']:
                    h_type = self.get_node_type(h_id)
                    t_type = self.get_node_type(t_id)
                    links['meta'][r_id] = (h_type, t_type)
                links['data'][r_id].append((h_id, t_id, link_weight))
                links['count'][r_id] += 1
                links['total'] += 1
        new_data = {}
        for r_id in links['data']:
            new_data[r_id] = self.list_to_sp_mat(links['data'][r_id])
        links['data'] = new_data
        return links

    def load_nodes(self):
        """
        return nodes dict
            total: total number of nodes
            count: a dict of int, number of nodes for each type
            attr: a dict of np.array (or None), attribute matrices for each type of nodes
            shift: node_id shift for each type. You can get the id range of a type by
                        [ shift[node_type], shift[node_type]+count[node_type] )
        """
        nodes = {'total': 0, 'count': Counter(), 'attr': {}, 'shift': {}}
        import os

        dat_file = os.path.join(self.path ,'node.dat')
        with open(dat_file, 'r', encoding='utf-8') as f:
            for line in f:
                th = line.split('\t')
                if len(th) == 4:
                    # Then this line of node has attribute
                    node_id, node_name, node_type, node_attr = th
                    node_id = int(node_id)
                    node_type = int(node_type)
                    node_attr = list(map(float, node_attr.split(',')))
                    nodes['count'][node_type] += 1
                    nodes['attr'][node_id] = node_attr
                    nodes['total'] += 1
                elif len(th) == 3:
                    # Then this line of node doesn't have attribute
                    node_id, node_name, node_type = th
                    node_id = int(node_id)
                    node_type = int(node_type)
                    nodes['count'][node_type] += 1
                    nodes['total'] += 1
                else:
                    raise Exception("Too few information to parse!")
        shift = 0
        attr = {}
        for i in range(len(nodes['count'])):
            nodes['shift'][i] = shift
            if shift in nodes['attr']:
                mat = []
                for j in range(shift, shift + nodes['count'][i]):
                    mat.append(nodes['attr'][j])
                attr[i] = np.array(mat)
            else:
                attr[i] = None
            shift += nodes['count'][i]
        nodes['attr'] = attr
        return nodes

class data_loader_lp:
    def __init__(self, path):
        self.path = path
        self.nodes = self.load_nodes()
        self.links = self.load_links()
        self.links_lp_task_s,self.links_lp_task_d=self.load_links_task()
        self.labels_train = self.load_labels('label.dat')
        self.labels_test = self.load_labels('label.dat.test')

    def get_sub_graph(self, node_types_tokeep):
        """
        node_types_tokeep is a list or set of node types that you want to keep in the sub-graph
        We only support whole type sub-graph for now.
        This is an in-place update function!
        return: old node type id to new node type id dict, old edge type id to new edge type id dict
        """
        keep = set(node_types_tokeep)
        new_node_type = 0
        new_node_id = 0
        new_nodes = {'total':0, 'count':Counter(), 'attr':{}, 'shift':{}}
        new_links = {'total':0, 'count':Counter(), 'meta':{}, 'data':defaultdict(list)}
        new_labels_train = {'num_classes':0, 'total':0, 'count':Counter(), 'data':None, 'mask':None}
        new_labels_test = {'num_classes':0, 'total':0, 'count':Counter(), 'data':None, 'mask':None}
        old_nt2new_nt = {}
        old_idx = []
        for node_type in self.nodes['count']:
            if node_type in keep:
                nt = node_type
                nnt = new_node_type
                old_nt2new_nt[nt] = nnt
                cnt = self.nodes['count'][nt]
                new_nodes['total'] += cnt
                new_nodes['count'][nnt] = cnt
                new_nodes['attr'][nnt] = self.nodes['attr'][nt]
                new_nodes['shift'][nnt] = new_node_id
                beg = self.nodes['shift'][nt]
                old_idx.extend(range(beg, beg+cnt))
                
                cnt_label_train = self.labels_train['count'][nt]
                new_labels_train['count'][nnt] = cnt_label_train
                new_labels_train['total'] += cnt_label_train
                cnt_label_test = self.labels_test['count'][nt]
                new_labels_test['count'][nnt] = cnt_label_test
                new_labels_test['total'] += cnt_label_test
                
                new_node_type += 1
                new_node_id += cnt

        new_labels_train['num_classes'] = self.labels_train['num_classes']
        new_labels_test['num_classes'] = self.labels_test['num_classes']
        for k in ['data', 'mask']:
            new_labels_train[k] = self.labels_train[k][old_idx]
            new_labels_test[k] = self.labels_test[k][old_idx]

        old_et2new_et = {}
        new_edge_type = 0
        for edge_type in self.links['count']:
            h, t = self.links['meta'][edge_type]
            if h in keep and t in keep:
                et = edge_type
                net = new_edge_type
                old_et2new_et[et] = net
                new_links['total'] += self.links['count'][et]
                new_links['count'][net] = self.links['count'][et]
                new_links['meta'][net] = tuple(map(lambda x:old_nt2new_nt[x], self.links['meta'][et]))
                new_links['data'][net] = self.links['data'][et][old_idx][:, old_idx]
                new_edge_type += 1

        self.nodes = new_nodes
        self.links = new_links
        self.labels_train = new_labels_train
        self.labels_test = new_labels_test
        return old_nt2new_nt, old_et2new_et

    def get_meta_path(self, meta=[]):
        """
        Get meta path matrix
            meta is a list of edge types (also can be denoted by a pair of node types)
            return a sparse matrix with shape [node_num, node_num]
        """
        ini = sp.eye(self.nodes['total'])
        meta = [self.get_edge_type(x) for x in meta]
        for x in meta:
            ini = ini.dot(self.links['data'][x]) if x >= 0 else ini.dot(self.links['data'][-x - 1].T)
        return ini

    def dfs(self, now, meta, meta_dict):
        if len(meta) == 0:
            meta_dict[now[0]].append(now)
            return
        th_mat = self.links['data'][meta[0]] if meta[0] >= 0 else self.links['data'][-meta[0] - 1].T
        th_node = now[-1]
        for col in th_mat[th_node].nonzero()[1]:
            self.dfs(now+[col], meta[1:], meta_dict)

    def get_full_meta_path(self, meta=[], symmetric=False):
        """
        Get full meta path for each node
            meta is a list of edge types (also can be denoted by a pair of node types)
            return a dict of list[list] (key is node_id)
        """
        meta = [self.get_edge_type(x) for x in meta]
        if len(meta) == 1:
            meta_dict = {}
            start_node_type = self.links['meta'][meta[0]][0] if meta[0]>=0 else self.links['meta'][-meta[0]-1][1]
            for i in range(self.nodes['shift'][start_node_type], self.nodes['shift'][start_node_type]+self.nodes['count'][start_node_type]):
                meta_dict[i] = []
                self.dfs([i], meta, meta_dict)
        else:
            meta_dict1 = {}
            meta_dict2 = {}
            mid = len(meta) // 2
            meta1 = meta[:mid]
            meta2 = meta[mid:]
            start_node_type = self.links['meta'][meta1[0]][0] if meta1[0]>=0 else self.links['meta'][-meta1[0]-1][1]
            for i in range(self.nodes['shift'][start_node_type], self.nodes['shift'][start_node_type]+self.nodes['count'][start_node_type]):
                meta_dict1[i] = []
                self.dfs([i], meta1, meta_dict1)
            start_node_type = self.links['meta'][meta2[0]][0] if meta2[0]>=0 else self.links['meta'][-meta2[0]-1][1]
            for i in range(self.nodes['shift'][start_node_type], self.nodes['shift'][start_node_type]+self.nodes['count'][start_node_type]):
                meta_dict2[i] = []
            if symmetric:
                for k in meta_dict1:
                    paths = meta_dict1[k]
                    for x in paths:
                        meta_dict2[x[-1]].append(list(reversed(x)))
            else:
                for i in range(self.nodes['shift'][start_node_type], self.nodes['shift'][start_node_type]+self.nodes['count'][start_node_type]):
                    self.dfs([i], meta2, meta_dict2)
            meta_dict = {}
            start_node_type = self.links['meta'][meta1[0]][0] if meta1[0]>=0 else self.links['meta'][-meta1[0]-1][1]
            for i in range(self.nodes['shift'][start_node_type], self.nodes['shift'][start_node_type]+self.nodes['count'][start_node_type]):
                meta_dict[i] = []
                for beg in meta_dict1[i]:
                    for end in meta_dict2[beg[-1]]:
                        meta_dict[i].append(beg + end[1:])
        return meta_dict

    def gen_file_for_evaluate(self, test_idx, label, file_name, mode='bi'):
        if test_idx.shape[0] != label.shape[0]:
            return
        if mode == 'multi':
            multi_label=[]
            for i in range(label.shape[0]):
                label_list = [str(j) for j in range(label[i].shape[0]) if label[i][j]==1]
                multi_label.append(','.join(label_list))
            label=multi_label
        elif mode=='bi':
            label = np.array(label)
        else:
            return
        with open(file_name, "w") as f:
            for nid, l in zip(test_idx, label):
                f.write(f"{nid}\t\t{self.get_node_type(nid)}\t{l}\n")

    def evaluate(self, pred):
        print(f"{bcolors.WARNING}Warning: If you want to obtain test score, please submit online on biendata.{bcolors.ENDC}")
        y_true = self.labels_test['data'][self.labels_test['mask']]
        micro = f1_score(y_true, pred, average='micro')
        macro = f1_score(y_true, pred, average='macro')
        result = {
            'micro-f1': micro,
            'macro-f1': macro
        }
        return result

    def load_labels(self, name):
        """
        return labels dict
            num_classes: total number of labels
            total: total number of labeled data
            count: number of labeled data for each node type
            data: a numpy matrix with shape (self.nodes['total'], self.labels['num_classes'])
            mask: to indicate if that node is labeled, if False, that line of data is masked
        """
        labels = {'num_classes':0, 'total':0, 'count':Counter(), 'data':None, 'mask':None}
        nc = 0
        mask = np.zeros(self.nodes['total'], dtype=bool)
        data = [None for i in range(self.nodes['total'])]
        with open(os.path.join(self.path, name), 'r', encoding='utf-8') as f:
            for line in f:
                th = line.split('\t')
                node_id, node_name, node_type, node_label = int(th[0]), th[1], int(th[2]), list(map(int, th[3].split(',')))
                for label in node_label:
                    nc = max(nc, label+1)
                mask[node_id] = True
                data[node_id] = node_label
                labels['count'][node_type] += 1
                labels['total'] += 1
        labels['num_classes'] = nc
        new_data = np.zeros((self.nodes['total'], labels['num_classes']), dtype=int)
        for i,x in enumerate(data):
            if x is not None:
                for j in x:
                    new_data[i, j] = 1
        labels['data'] = new_data
        labels['mask'] = mask
        return labels

    def get_node_type(self, node_id):
        for i in range(len(self.nodes['shift'])):
            if node_id < self.nodes['shift'][i]+self.nodes['count'][i]:
                return i

    def get_edge_type(self, info):
        if type(info) is int or len(info) == 1:
            return info
        for i in range(len(self.links['meta'])):
            if self.links['meta'][i] == info:
                return i
        info = (info[1], info[0])
        for i in range(len(self.links['meta'])):
            if self.links['meta'][i] == info:
                return -i - 1
        raise Exception('No available edge type')

    def get_edge_info(self, edge_id):
        return self.links['meta'][edge_id]
    
    def list_to_sp_mat(self, li):
        data = [x[2] for x in li]
        i = [x[0] for x in li]
        j = [x[1] for x in li]
        return sp.coo_matrix((data, (i,j)), shape=(self.nodes['total'], self.nodes['total'])).tocsr()
    
    def load_links(self):
        """
        return links dict
            total: total number of links
            count: a dict of int, number of links for each type
            meta: a dict of tuple, explaining the link type is from what type of node to what type of node
            data: a dict of sparse matrices, each link type with one matrix. Shapes are all (nodes['total'], nodes['total'])
        """
        links = {'total':0, 'count':Counter(), 'meta':{}, 'data':defaultdict(list)}
        with open(os.path.join(self.path, 'link_lp.dat'), 'r', encoding='utf-8') as f:
            for line in f:
                th = line.split('\t')
                h_id, t_id, r_id, link_weight = int(th[0]), int(th[1]), int(th[2]), float(th[3])
                if r_id not in links['meta']:
                    h_type = self.get_node_type(h_id)
                    t_type = self.get_node_type(t_id)
                    links['meta'][r_id] = (h_type, t_type)
                links['data'][r_id].append((h_id, t_id, link_weight))
                links['count'][r_id] += 1
                links['total'] += 1
        new_data = {}
        for r_id in links['data']:
            new_data[r_id] = self.list_to_sp_mat(links['data'][r_id])
        links['data'] = new_data
        return links

    def load_links_task(self):
        ground_truth_source=[]
        ground_truth_des=[]
        with open(os.path.join(self.path, 'link_task.dat'), 'r', encoding='utf-8') as f:
            for line in f:
                th = line.split('\t')
                h_id, t_id, r_id, link_weight = int(th[0]), int(th[1]), int(th[2]), float(th[3])
                ground_truth_source.append(h_id)
                ground_truth_des.append(t_id)
        return ground_truth_source, ground_truth_des


    def load_nodes(self):
        """
        return nodes dict
            total: total number of nodes
            count: a dict of int, number of nodes for each type
            attr: a dict of np.array (or None), attribute matrices for each type of nodes
            shift: node_id shift for each type. You can get the id range of a type by 
                        [ shift[node_type], shift[node_type]+count[node_type] )
        """
        nodes = {'total':0, 'count':Counter(), 'attr':{}, 'shift':{}}
        with open(os.path.join(self.path, 'node.dat'), 'r', encoding='utf-8') as f:
            for line in f:
                th = line.split('\t')
                if len(th) == 4:
                    # Then this line of node has attribute
                    node_id, node_name, node_type, node_attr = th
                    node_id = int(node_id)
                    node_type = int(node_type)
                    node_attr = list(map(float, node_attr.split(',')))
                    nodes['count'][node_type] += 1
                    nodes['attr'][node_id] = node_attr
                    nodes['total'] += 1
                elif len(th) == 3:
                    # Then this line of node doesn't have attribute
                    node_id, node_name, node_type = th
                    node_id = int(node_id)
                    node_type = int(node_type)
                    nodes['count'][node_type] += 1
                    nodes['total'] += 1
                else:
                    raise Exception("Too few information to parse!")
        shift = 0
        attr = {}
        for i in range(len(nodes['count'])):
            nodes['shift'][i] = shift
            if shift in nodes['attr']:
                mat = []
                for j in range(shift, shift+nodes['count'][i]):
                    mat.append(nodes['attr'][j])
                attr[i] = np.array(mat)
            else:
                attr[i] = None
            shift += nodes['count'][i]
        nodes['attr'] = attr
        return nodes

###################

################### data.py文件

def load_data(prefix='DBLP', shotnum=10,tasknum=2,index=None):
    
    import os
    dl = data_loader(os.path.join(global_dir_path,'data',prefix))
    index_dir=os.path.join(global_dir_path,'data',prefix,index)

    print(index_dir)
    index_exist=os.path.exists(index_dir)
    if index_exist==False:
        print("Please Generate Few shot tasks first,using SHGN")
        sys.exit()
    print("##################")
    print("index for ",shotnum,"shots ",tasknum,"tasks exists: ",index_exist)

    features = []
    for i in range(len(dl.nodes['count'])):
        th = dl.nodes['attr'][i]
        if th is None:
            features.append(sp.eye(dl.nodes['count'][i]))
        else:
            features.append(th)
    adjM = sum(dl.links['data'].values())
    #labels = np.zeros((dl.nodes['count'][0], dl.labels_train['num_classes']), dtype=int)
    print("Loading Few Shot Tasks")
    save_train = np.load(os.path.join(index_dir, "train_index.npy"), allow_pickle=True)
    save_val = np.load(os.path.join(index_dir, "val_index.npy"), allow_pickle=True)
    save_test = np.load(os.path.join(index_dir, "test_index.npy"), allow_pickle=True)
    save_train_labels = np.load(os.path.join(index_dir, "train_labels.npy"), allow_pickle=True)
    save_val_labels = np.load(os.path.join(index_dir, "val_labels.npy"), allow_pickle=True)

    train_val_test_idx = {}
    train_val_test_idx['train_idx'] = save_train
    train_val_test_idx['val_idx'] = save_val
    train_val_test_idx['test_idx'] = save_test
    multi_task_labels = {}
    multi_task_labels['train'] = save_train_labels
    multi_task_labels['val'] = save_val_labels
    return features,\
           adjM, \
           multi_task_labels,\
           train_val_test_idx,\
            dl


def load_data_lp(prefix='DBLP', shotnum=10,tasknum=2,index=None):
    import os
    dl = data_loader(os.path.join(global_dir_path,'data',prefix))
    index_dir=os.path.join(global_dir_path,'data',prefix,index)

    print(index_dir)
    index_exist=os.path.exists(index_dir)
    if index_exist==False:
        print("Please Generate Few shot tasks first,using SHGN")
        sys.exit()
    print("##################")
    print("index for ",shotnum,"shots ",tasknum,"tasks exists: ",index_exist)

    features = []
    for i in range(len(dl.nodes['count'])):
        th = dl.nodes['attr'][i]
        if th is None:
            features.append(sp.eye(dl.nodes['count'][i]))
        else:
            features.append(th)
    adjM = sum(dl.links['data'].values())
    print("Loading Few Shot Tasks")
    save_train = np.load(os.path.join(index_dir, "train_index.npy"), allow_pickle=True)
    save_val = np.load(os.path.join(index_dir, "val_index.npy"), allow_pickle=True)
    save_test = np.load(os.path.join(index_dir, "test_index.npy"), allow_pickle=True)
    save_train_labels = np.load(os.path.join(index_dir, "train_labels.npy"), allow_pickle=True)
    save_val_labels = np.load(os.path.join(index_dir, "val_labels.npy"), allow_pickle=True)

    train_val_test_idx = {}
    train_val_test_idx['train_idx'] = save_train
    train_val_test_idx['val_idx'] = save_val
    train_val_test_idx['test_idx'] = save_test
    multi_task_labels = {}
    multi_task_labels['train'] = save_train_labels
    multi_task_labels['val'] = save_val_labels

    task_source=dl.links_lp_task_s
    task_des=dl.links_lp_task_d

    return features,\
           adjM, \
           multi_task_labels,\
           train_val_test_idx,\
            dl, task_source,task_des


def load_pretrain_data(prefix='DBLP'):
    #from scripts.data_loader import data_loader
    import os
    dl = data_loader(os.path.join(global_dir_path,'data',prefix))


    features = []
    for i in range(len(dl.nodes['count'])):
        th = dl.nodes['attr'][i]
        if th is None:
            features.append(sp.eye(dl.nodes['count'][i]))
        else:
            features.append(th)
    adjM = sum(dl.links['data'].values())

    return features,\
           adjM, \
            dl


def load_pretrain_data_lp(prefix='DBLP'):
    import os
    dl = data_loader(os.path.join(global_dir_path,'data',prefix))

    #from scripts.data_loader import data_loader
    features = []
    for i in range(len(dl.nodes['count'])):
        th = dl.nodes['attr'][i]
        if th is None:
            features.append(sp.eye(dl.nodes['count'][i]))
        else:
            features.append(th)
    adjM = sum(dl.links['data'].values())
    return features,\
           adjM, \
            dl


def load_acm(feat_type=0,shotnum=10,tasknum=2,index=None):

    import os
    dl = data_loader(os.path.join(global_dir_path,'data','ACM'))
    index_dir=os.path.join(global_dir_path,'data','ACM',index)


    index_exist=os.path.exists(index_dir)
    if index_exist==False:
        print("Please Generate Few shot tasks first,using SHGN")
        sys.exit()
    print("##################")
    print("index for ",shotnum,"shots ",tasknum,"tasks exists: ",index_exist)

    link_type_dic = {0: 'pp', 1: '-pp', 2: 'pa', 3: 'ap', 4: 'ps', 5: 'sp', 6: 'pt', 7: 'tp'}
    paper_num = dl.nodes['count'][0]
    data_dic = {}
    for link_type in dl.links['data'].keys():
        src_type = str(dl.links['meta'][link_type][0])
        dst_type = str(dl.links['meta'][link_type][1])
        data_dic[(src_type, link_type_dic[link_type], dst_type)] = dl.links['data'][link_type].nonzero()
    hg = dgl.heterograph(data_dic)

    # paper feature
    if feat_type == 0:
        '''preprocessed feature'''
        features = th.FloatTensor(dl.nodes['attr'][0])
    else:
        '''one-hot'''
        features = th.FloatTensor(np.eye(paper_num))

    # paper labels

    labels = dl.labels_test['data'][:paper_num] + dl.labels_train['data'][:paper_num]
    labels = [np.argmax(l) for l in labels]  # one-hot to value
    labels = th.LongTensor(labels)

    num_classes = 3

    print("Loading Few Shot Tasks")
    save_train = np.load(os.path.join(index_dir, "train_index.npy"), allow_pickle=True)
    save_val = np.load(os.path.join(index_dir, "val_index.npy"), allow_pickle=True)
    save_test = np.load(os.path.join(index_dir, "test_index.npy"), allow_pickle=True)
    save_train_labels = np.load(os.path.join(index_dir, "train_labels.npy"), allow_pickle=True)
    save_val_labels = np.load(os.path.join(index_dir, "val_labels.npy"), allow_pickle=True)

    train_val_test_idx = {}
    train_val_test_idx['train_idx'] = save_train
    train_val_test_idx['val_idx'] = save_val
    train_val_test_idx['test_idx'] = save_test
    multi_task_labels = {}
    multi_task_labels['train'] = save_train_labels
    multi_task_labels['val'] = save_val_labels

    train_valid_mask = dl.labels_train['mask'][:paper_num]
    test_mask = dl.labels_test['mask'][:paper_num]
    train_valid_indices = np.where(train_valid_mask == True)[0]
    split_index = int(0.7 * np.shape(train_valid_indices)[0])
    train_indices = train_valid_indices[:split_index]
    valid_indices = train_valid_indices[split_index:]
    train_mask = copy.copy(train_valid_mask)
    valid_mask = copy.copy(train_valid_mask)
    train_mask[valid_indices] = False
    valid_mask[train_indices] = False

    meta_paths = [['pp', 'ps', 'sp'], ['-pp', 'ps', 'sp'], ['pa', 'ap'], ['ps', 'sp'], ['pt', 'tp']]
    return hg, features, labels, num_classes, train_val_test_idx, multi_task_labels, meta_paths

#################


def sp_to_spt(mat):
    coo = mat.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))

def mat2tensor(mat):
    if type(mat) is np.ndarray:
        return torch.from_numpy(mat).type(torch.FloatTensor)
    return sp_to_spt(mat)

def prompt_pretrain_sample(adj,n):
    nodenum=adj.shape[0]
    indices=adj.indices
    indptr=adj.indptr
    res=np.zeros((nodenum,1+n))
    whole=np.array(range(nodenum))
    print("#############")
    print("start sampling disconnected tuples")
    for i in tqdm.trange(nodenum):
        nonzero_index_i_row=indices[indptr[i]:indptr[i+1]]
        zero_index_i_row=np.setdiff1d(whole,nonzero_index_i_row)
        np.random.shuffle(nonzero_index_i_row)
        np.random.shuffle(zero_index_i_row)
        if np.size(nonzero_index_i_row)==0:
            res[i][0] = i
        else:
            res[i][0]=nonzero_index_i_row[0]
        res[i][1:1+n]=zero_index_i_row[0:n]
    return res.astype(int)

def prompt_pretrain_sample_target_node(dataset,dl,adj,n_unrelated,target_node_tuple_num):
    nodenum=adj.shape[0]
    indices=adj.indices
    indptr=adj.indptr
    whole=np.array(range(nodenum))
    print("#############")
    print("start sampling target nodes' disconnected tuples")

    if dataset=="ACM":
        res = np.zeros((np.sum(target_node_tuple_num), 2 + n_unrelated))
        count=0
        for i in tqdm.trange(dl.nodes['shift'][0], dl.nodes['shift'][0] + dl.nodes['count'][0]):
            nonzero_index_i_row=indices[indptr[i]:indptr[i+1]]
            zero_index_i_row=np.setdiff1d(whole,nonzero_index_i_row)
            np.random.shuffle(nonzero_index_i_row)
            np.random.shuffle(zero_index_i_row)
            for j in np.arange(target_node_tuple_num[i]):
                res[count][0] = i
                if np.size(nonzero_index_i_row)==0:
                    res[count][1] = i
                else:
                    res[count][1]=nonzero_index_i_row[j]
                res[count][2:2+n_unrelated]=zero_index_i_row[j+j*n_unrelated:j+(j+1)*n_unrelated]
    return res.astype(int)

def hprompt_pretrain_sample(dataset,dl,adj,n_unrelated):
    #target nodes are labeled nodes
    print("#############")
    print("start hetero-sampling tuples")
    nodenum=adj.shape[0]
    indices=adj.indices
    indptr=adj.indptr
    target_node_tuple_num=None
    if dataset=="ACM":
        paper_nodes= np.arange(dl.nodes['shift'][0],dl.nodes['shift'][0]+dl.nodes['count'][0])
        author_nodes= np.arange(dl.nodes['shift'][1],dl.nodes['shift'][1]+dl.nodes['count'][1])
        subject_nodes= np.arange(dl.nodes['shift'][2],dl.nodes['shift'][2]+dl.nodes['count'][2])
        term_nodes= np.arange(dl.nodes['shift'][3],dl.nodes['shift'][3]+dl.nodes['count'][3])
        target_node_tuple_num=np.zeros_like(paper_nodes)
        whole=[]
        whole.append(paper_nodes)
        whole.append(author_nodes)
        whole.append(subject_nodes)
        whole.append(term_nodes)
        res = None
        first_res=True
        count=0
        paper_nodes_dont_have_type={}
        paper_nodes_dont_have_type[0]=[]
        paper_nodes_dont_have_type[1]=[]
        paper_nodes_dont_have_type[2]=[]
        paper_nodes_dont_have_type[3]=[]
        isolated_nodes=[]
        temp=np.zeros((1,2+n_unrelated))
        for w in whole:
            print('\nedge type: ', count)
            for i in tqdm.trange(dl.nodes['shift'][0],dl.nodes['shift'][0]+dl.nodes['count'][0]):
                nonzero_index_i_row=indices[indptr[i]:indptr[i+1]]
                nonzero_corresponding_type_index_i_row=np.intersect1d(nonzero_index_i_row,w)
                if not nonzero_corresponding_type_index_i_row.any():
                    paper_nodes_dont_have_type[count].append(i)
                    continue
                target_node_tuple_num[i]+=1
                zero_index_i_row=np.setdiff1d(w,nonzero_index_i_row)
                np.random.shuffle(nonzero_index_i_row)
                np.random.shuffle(zero_index_i_row)
                temp[0][0] = i
                temp[0][1] = nonzero_corresponding_type_index_i_row[0]
                temp[0][2:2 + n_unrelated] = zero_index_i_row[0:n_unrelated]
                if first_res:
                    res=temp
                    first_res=False
                else:
                    res=np.concatenate((res,temp),axis=0)
            count+=1
            # for (target,pos_candidate) in tqdm(edge2type):
        print('isolated nodes:', isolated_nodes)
        print('paper_nodes_dont_have_type 0 num', len(paper_nodes_dont_have_type[0]))
        print('paper_nodes_dont_have_type 1 num',len(paper_nodes_dont_have_type[1]))
        print('paper_nodes_dont_have_type 2 num',len(paper_nodes_dont_have_type[2]))
        print('paper_nodes_dont_have_type 3 num',len(paper_nodes_dont_have_type[3]))

    elif dataset=="DBLP":
        paper_nodes= np.arange(dl.nodes['shift'][1],dl.nodes['shift'][1]+dl.nodes['count'][1])
        target_node_tuple_num=np.zeros_like(paper_nodes)
        whole=[]
        whole.append(paper_nodes)
        res = None
        first_res=True
        ##each relation
        count=0
        author_nodes_dont_have_type={}
        author_nodes_dont_have_type[0]=[]
        isolated_nodes=[]
        temp=np.zeros((1,2+n_unrelated))
        for w in whole:
            ##each paper node
            print('\nedge type: ', count)
            #Is there some author node that is isolated?
            for i in tqdm.trange(dl.nodes['shift'][0],dl.nodes['shift'][0]+dl.nodes['count'][0]):
                #need to select the corresponding type of node
                nonzero_index_i_row=indices[indptr[i]:indptr[i+1]]
                ##ACM has no isolated paper_node
                nonzero_corresponding_type_index_i_row=np.intersect1d(nonzero_index_i_row,w)
                #some paper nodes dont have several relation
                if not nonzero_corresponding_type_index_i_row.any():
                    author_nodes_dont_have_type[count].append(i)
                    continue
                zero_index_i_row=np.setdiff1d(w,nonzero_index_i_row)
                np.random.shuffle(nonzero_index_i_row)
                np.random.shuffle(zero_index_i_row)
                temp[0][0] = i
                temp[0][1] = nonzero_corresponding_type_index_i_row[0]
                temp[0][2:2 + n_unrelated] = zero_index_i_row[0:n_unrelated]
                if first_res:
                    res=temp
                    first_res=False
                else:
                    res=np.concatenate((res,temp),axis=0)
            count+=1
        print('isolated nodes:', isolated_nodes)
        print('paper_nodes_dont_have_type 0 num', len(author_nodes_dont_have_type[0]))
    np.random.shuffle(res)
    return res.astype(int),target_node_tuple_num

def mygather(feature, index):
    input_size=index.size(0)
    index = index.flatten()
    index = index.reshape(len(index), 1)
    index = torch.broadcast_to(index, (len(index), feature.size(1)))
    # print(tuples)
    res = torch.gather(feature, dim=0, index=index)
    return res.reshape(input_size,-1,feature.size(1))

#tuple[i][0] represent pos node
def compareloss(feature,tuples,temperature,device):
    h_tuples=mygather(feature,tuples)
    temp = torch.arange(0, len(tuples))
    temp = temp.reshape(-1, 1)
    temp = torch.broadcast_to(temp, (temp.size(0), tuples.size(1)))
    temp=temp.to(device)
    h_i = mygather(feature, temp)
    sim = F.cosine_similarity(h_i, h_tuples, dim=2)
    exp = torch.exp(sim)
    exp = exp / temperature
    exp = exp.permute(1, 0)
    numerator = exp[0].reshape(-1, 1)
    denominator = exp[1:exp.size(0)]
    denominator = denominator.permute(1, 0)
    denominator = denominator.sum(dim=1, keepdim=True)
    res = -1 * torch.log(numerator / denominator)
    return res.mean()

#tuple[i][0] represent target node
def tcompareloss(feature,tuples,temperature,device):
    h_tuples=mygather(feature,tuples)
    temp=h_tuples.permute(1,0,2)[0]
    temp=temp.unsqueeze(0)
    h_i = temp.permute(1,0,2)
    h_tuples=h_tuples.permute(1,0,2)[1:]
    h_tuples=h_tuples.permute(1,0,2)
    sim = F.cosine_similarity(h_i, h_tuples, dim=2)
    exp = torch.exp(sim)
    exp = exp / temperature
    exp = exp.permute(1, 0)
    numerator = exp[0].reshape(-1, 1)
    denominator = exp[1:exp.size(0)]
    denominator = denominator.permute(1, 0)
    denominator = denominator.sum(dim=1, keepdim=True)
    res = -1 * torch.log(numerator / denominator)
    return res.mean()

#tuple[i][0] represent target node, add relation to similarity
def hcompareloss(feature,tuples,temperature,device):
    h_tuples=mygather(feature,tuples)
    temp=h_tuples.permute(1,0,2)[0]
    temp=temp.unsqueeze(0)
    h_i = temp.permute(1,0,2)
    h_tuples=h_tuples.permute(1,0,2)[1:]
    h_tuples=h_tuples.permute(1,0,2)
    sim = F.cosine_similarity(h_i, h_tuples, dim=2)
    exp = torch.exp(sim)
    exp = exp / temperature
    exp = exp.permute(1, 0)
    numerator = exp[0].reshape(-1, 1)
    denominator = exp[1:exp.size(0)]
    denominator = denominator.permute(1, 0)
    denominator = denominator.sum(dim=1, keepdim=True)
    res = -1 * torch.log(numerator / denominator)
    return res.mean()

def two_hop_subgraph_nodelist(g,subgraphs_dir,max_neigbour_num=10):
    if os.path.exists(subgraphs_dir) == False:
        os.mkdir(subgraphs_dir)
    if os.path.exists(os.path.join(subgraphs_dir,'0.npy'))==False:
        nodenum = g.number_of_nodes()
        subgraph_list = []
        for i in range(nodenum):
            neighbors = g.successors(i).numpy().tolist()
            if len(neighbors)>max_neigbour_num:
                neighbors=random.sample(neighbors, max_neigbour_num)
            two_hop_neighbors = []
            for neighbor in neighbors:
                two_hop_neighbors.extend(g.successors(neighbor).numpy().tolist())
            subgraph_nodes = [i] + neighbors + two_hop_neighbors
            subgraph_nodes = np.array(list(set(subgraph_nodes)))
            subgraph_dir = os.path.join(subgraphs_dir, str(i))
            np.save(subgraph_dir, subgraph_nodes)
            subgraph_list.append(torch.tensor(subgraph_nodes))
    else:
        # Load subgraphs list
        subgraph_list = []
        file_names = [file_name for file_name in os.listdir(subgraphs_dir) if file_name.endswith('.npy')]
        sorted_file_names = sorted(file_names, key=lambda x: int(x.split('.')[0]))
        for file_name in sorted_file_names:
            file_path = os.path.join(subgraphs_dir, file_name)
            np_array = np.load(file_path)
            subgraph_nodes = torch.tensor(np_array)
            subgraph_list.append(subgraph_nodes)
    return subgraph_list




def pretrain_DBLP(args):
    feats_type = args.feats_type
    features_list, adjM, dl = load_pretrain_data(args._dataset)
    
    # if args.device==1:
    #     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # elif args.device==0:
    #     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # else:
    #     device=torch.device('cpu')

    device = global_device
    
    features_list = [mat2tensor(features).to(device) for features in features_list]
    if feats_type == 0:
        in_dims = [features.shape[1] for features in features_list]
    elif feats_type == 1 or feats_type == 5:
        save = 0 if feats_type == 1 else 2
        in_dims = []#[features_list[0].shape[1]] + [10] * (len(features_list) - 1)
        for i in range(0, len(features_list)):
            if i == save:
                in_dims.append(features_list[i].shape[1])
            else:
                in_dims.append(10)
                features_list[i] = torch.zeros((features_list[i].shape[0], 10)).to(device)
    elif feats_type == 2 or feats_type == 4:
        save = feats_type - 2
        in_dims = [features.shape[0] for features in features_list]
        for i in range(0, len(features_list)):
            if i == save:
                in_dims[i] = features_list[i].shape[1]
                continue
            dim = features_list[i].shape[0]
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list[i] = torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device)
    elif feats_type == 3:
        in_dims = [features.shape[0] for features in features_list]
        for i in range(len(features_list)):
            dim = features_list[i].shape[0]
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list[i] = torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device)
    edge2type = {}
    #edge feature:0,1,2,3,...
    for k in dl.links['data']:
        for u,v in zip(*dl.links['data'][k].nonzero()):
            edge2type[(u,v)] = k
    # print(dl.links['count'])
    for i in range(dl.nodes['total']):
        if (i,i) not in edge2type:
            edge2type[(i,i)] = len(dl.links['count'])
    for k in dl.links['data']:
        for u,v in zip(*dl.links['data'][k].nonzero()):
            if (v,u) not in edge2type:
                edge2type[(v,u)] = k+1+len(dl.links['count'])

    samples_dir=os.path.join(global_dir_path,'data',args._dataset,"".join([str(args.tuple_neg_disconnected_num),'neg_samples']))
    load_samples_dir=os.path.join(global_dir_path,'data',args._dataset,"".join([str(args.tuple_neg_disconnected_num),'neg_samples.npy']))
    target_nodes_samples_dir=os.path.join(global_dir_path,'data',args._dataset,"".join([str(args.target_tuple_neg_disconnected_num),'target_neg_samples']))
    target_nodes_load_samples_dir=os.path.join(global_dir_path,'data',args._dataset,"".join([str(args.target_tuple_neg_disconnected_num),'target_neg_samples.npy']))
    hsamples_dir=os.path.join(global_dir_path,'data',args._dataset,"".join([str(args.tuple_neg_unrelated_num),'neg_unrelated_samples']))
    hload_samples_dir=os.path.join(global_dir_path,'data',args._dataset,"".join([str(args.tuple_neg_unrelated_num),'neg_unrelated_samples.npy']))
    target_nodes_tuple_num_dir=os.path.join(global_dir_path,'data',args._dataset,"".join([str(args.tuple_neg_unrelated_num),'target_nodes_tuple_num']))
    load_target_nodes_tuple_num_dir=os.path.join(global_dir_path,'data',args._dataset,"".join([str(args.tuple_neg_unrelated_num),'target_nodes_tuple_num.npy']))

    g = dgl.DGLGraph(adjM+(adjM.T))
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    g = g.to(device)
    trans_g=dgl.reverse(g)

    torch_sparse_adj=g.adj()
    #   修改处
    torch_sparse_adj = torch.sparse_coo_tensor(torch_sparse_adj.indices(),
                                               torch_sparse_adj.val,
                                               torch_sparse_adj.shape)


    torch_sparse_adj = torch_sparse_adj.to(device)

    if os.path.exists(load_samples_dir)==False:
        if args._dataset=='Freebase':
            samples=prompt_pretrain_sample(adjM+adjM.T, args.tuple_neg_disconnected_num)
        else:
            samples = prompt_pretrain_sample(adjM, args.tuple_neg_disconnected_num)
        np.save(samples_dir, samples)
    else:
        print("load sampleing tuples")
        samples=np.load(load_samples_dir,allow_pickle=True)
    samples=torch.tensor(samples,dtype=int)


    if args.target_pretrain:
        target_samples=target_samples.to(device)
    else:
        samples = samples.to(device)

    if args.hetero_pretrain:
        hsamples=hsamples.to(device)

    e_feat = []
    for u, v in zip(*g.edges()):
        u = u.cpu().item()
        v = v.cpu().item()
        e_feat.append(edge2type[(u,v)])
    e_feat = torch.tensor(e_feat, dtype=torch.long).to(device)

    train_time=0
    early_stop=False
    for _ in range(args.repeat):
        num_classes = dl.labels_train['num_classes']
        heads = [args.num_heads] * args.num_layers + [1]
        if args.model_type == 'gat':
            if args.hetero_subgraph:
                if args._dataset == 'DBLP':
                    net = dblp_hGAT(g, in_dims, args.hidden_dim, num_classes, args.num_layers, heads, F.elu, args.dropout, args.dropout, args.slope, False)
            else:
                net = GAT(g, in_dims, args.hidden_dim, num_classes, args.num_layers, heads, F.elu, args.dropout, args.dropout, args.slope, False)
        elif args.model_type == 'gcn':
            if args.hetero_subgraph:
                if args._dataset =='ACM':
                    if args.semantic_weight:
                        net=acm_sem_hGCN(g, in_dims, args.hidden_dim, num_classes, args.num_layers, F.elu, args.dropout)
                    elif args.each_loss:
                        net = acm_hGCN_each_loss(g, in_dims, args.hidden_dim, num_classes, args.num_layers, F.elu, args.dropout)
                    else:
                        net = acm_hGCN(g, in_dims, args.hidden_dim, num_classes, args.num_layers, F.elu, args.dropout)
                elif args._dataset=='Freebase':
                    if args.freebase_type==0:
                        net=freebase_source_hGCN(g, in_dims, args.hidden_dim, num_classes, args.num_layers, F.elu, args.dropout)
                    elif args.freebase_type==1:
                        net=freebase_des_hGCN(g, in_dims, args.hidden_dim, num_classes, args.num_layers, F.elu, args.dropout)
                    else:
                        net=freebase_bi_hGCN(g, in_dims, args.hidden_dim, num_classes, args.num_layers, F.elu, args.dropout)
            else:
                net = GCN(g, in_dims, args.hidden_dim, num_classes, args.num_layers, F.elu, args.dropout)
        elif args.model_type == 'gin':
            net = GIN(g, in_dims, args.hidden_dim, num_classes, args.num_layers, F.relu, args.dropout)
        elif args.model_type =='SHGN':
            num_classes = dl.labels_train['num_classes']
            heads = [args.num_heads] * args.num_layers + [1]
            net = myGAT(g, args.edge_feats, len(dl.links['count'])*2+1, in_dims, args.hidden_dim, num_classes, args.num_layers, heads, F.elu, args.dropout, args.dropout, args.slope, True, 0.05)
        else:
            raise Exception('{} model is not defined!'.format(args.model_type))
        net.to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # training loop
        net.train()
        if args.model_type=='SHGN':
        
            early_stopping = EarlyStopping(
                patience=args.patience, verbose=True,#   os.path.join(dir_path,)
                save_path=os.path.join(global_dir_path,'checkpoint','pretrain','shgn_checkpoint_{}_{}.pt'.format(args._dataset, args.num_layers))
                                           )
        else:
            if args._dataset=='Freebase':
                if args.hetero_pretrain:
                    if args.hetero_subgraph:
                        if args.semantic_weight:
                            early_stopping = EarlyStopping(patience=args.patience, verbose=True,
                                                           save_path=os.path.join(
                                                            global_dir_path,'checkpoint','pretrain','checkpoint_hsubgraph_semantic_{}_{}_{}_{}_{}_{}_{}.pt'.
                                                           format(args._dataset, args.model_type, args.subgraph_hop_num,
                                                                  args.loss_weight, args.feats_type,args.tuple_neg_disconnected_num,args.tuple_neg_unrelated_num)
                                                                  
                                                                  )
                                                                  )
                        elif args.each_loss:
                            early_stopping = EarlyStopping(patience=args.patience, verbose=True,
                                                           save_path=os.path.join(global_dir_path,'checkpoint','pretrain','checkpoint_hsubgraph_each_loss_{}_{}_{}_{}_{}_{}_{}.pt'.
                                                           format(args._dataset, args.model_type, args.subgraph_hop_num,
                                                                  args.loss_weight, args.feats_type,args.tuple_neg_disconnected_num,args.tuple_neg_unrelated_num)))

                        else:
                            early_stopping = EarlyStopping(patience=args.patience, verbose=True,
                                                           save_path=os.path.join(global_dir_path,'checkpoint','pretrain','checkpoint_hsubgraph_{}_{}_{}_{}_{}_{}_{}.pt'.
                                                           format(args._dataset, args.model_type, args.subgraph_hop_num,
                                                                  args.loss_weight, args.feats_type,args.tuple_neg_disconnected_num,args.tuple_neg_unrelated_num)))
                    else:
                        early_stopping = EarlyStopping(patience=args.patience, verbose=True,
                                                       save_path=os.path.join(global_dir_path,'checkpoint','pretrain','checkpoint_{}_{}_{}_{}_{}_{}_{}.pt'.
                                                       format(args._dataset, args.model_type, args.subgraph_hop_num,
                                                              args.loss_weight, args.feats_type,args.tuple_neg_disconnected_num,args.tuple_neg_unrelated_num)))
                else:
                    if args.hetero_subgraph:
                        if args.semantic_weight:
                            early_stopping = EarlyStopping(patience=args.patience, verbose=True,
                                                           save_path=os.path.join(global_dir_path,'checkpoint','pretrain','checkpoint_hsubgraph_semantic_{}_{}_{}_{}_{}_{}.pt'.
                                                           format(args._dataset, args.model_type, args.subgraph_hop_num,
                                                                  args.feats_type,args.tuple_neg_disconnected_num,args.tuple_neg_unrelated_num)))
                        elif args.each_loss:
                            early_stopping = EarlyStopping(patience=args.patience, verbose=True,
                                                           save_path=os.path.join(global_dir_path,'checkpoint','pretrain','checkpoint_hsubgraph_each_loss_{}_{}_{}_{}_{}_{}_{}.pt'.
                                                           format(args._dataset, args.model_type, args.subgraph_hop_num,
                                                                  args.loss_weight, args.feats_type,args.tuple_neg_disconnected_num,args.tuple_neg_unrelated_num)))
                        else:
                            early_stopping = EarlyStopping(patience=args.patience, verbose=True,
                                                           save_path=os.path.join(global_dir_path,'checkpoint','pretrain','checkpoint_hsubgraph_{}_{}_{}_{}_{}_{}.pt'.
                                                           format(args._dataset, args.model_type, args.subgraph_hop_num,
                                                                  args.feats_type,args.tuple_neg_disconnected_num,args.tuple_neg_unrelated_num)))
                    else:
                        early_stopping = EarlyStopping(patience=args.patience, verbose=True,
                                                       save_path=os.path.join(global_dir_path,'checkpoint','pretrain','checkpoint_{}_{}_{}_{}_{}_{}.pt'.
                                                       format(args._dataset, args.model_type, args.subgraph_hop_num,
                                                              args.feats_type,args.tuple_neg_disconnected_num,args.tuple_neg_unrelated_num)))

            else:
                if args.hetero_pretrain:
                    if args.hetero_subgraph:
                        if args.semantic_weight:
                            early_stopping = EarlyStopping(patience=args.patience, verbose=True,
                                                           save_path=os.path.join(global_dir_path,'checkpoint','pretrain','checkpoint_hsubgraph_semantic_{}_{}_{}_{}_{}.pt'.
                                                           format(args._dataset, args.model_type, args.subgraph_hop_num,
                                                                  args.loss_weight, args.feats_type)))
                        elif args.each_loss:
                            early_stopping = EarlyStopping(patience=args.patience, verbose=True,
                                                           save_path=os.path.join(global_dir_path,'checkpoint','pretrain','checkpoint_hsubgraph_each_loss_{}_{}_{}_{}_{}.pt'.
                                                           format(args._dataset, args.model_type, args.subgraph_hop_num,
                                                                  args.loss_weight, args.feats_type)))

                        else:
                            early_stopping = EarlyStopping(patience=args.patience, verbose=True,
                                                           save_path=os.path.join(global_dir_path,'checkpoint','pretrain','checkpoint_hsubgraph_{}_{}_{}_{}_{}.pt'.
                                                           format(args._dataset, args.model_type, args.subgraph_hop_num,
                                                                  args.loss_weight, args.feats_type)))
                    else:
                        early_stopping = EarlyStopping(patience=args.patience, verbose=True,
                                                       save_path=os.path.join(global_dir_path,'checkpoint','pretrain','checkpoint_{}_{}_{}_{}_{}.pt'.
                                                       format(args._dataset, args.model_type,args.subgraph_hop_num,args.loss_weight,args.feats_type)))
                else:
                    if args.hetero_subgraph:
                        if args.semantic_weight:
                            early_stopping = EarlyStopping(patience=args.patience, verbose=True,
                                                           save_path=os.path.join(global_dir_path,'checkpoint','pretrain','checkpoint_hsubgraph_semantic_{}_{}_{}_{}.pt'.
                                                           format(args._dataset, args.model_type, args.subgraph_hop_num,
                                                                  args.feats_type)))
                        elif args.each_loss:
                            early_stopping = EarlyStopping(patience=args.patience, verbose=True,
                                                           save_path=os.path.join(global_dir_path,'checkpoint','pretrain','checkpoint_hsubgraph_each_loss_{}_{}_{}_{}_{}.pt'.
                                                           format(args._dataset, args.model_type, args.subgraph_hop_num,
                                                                  args.loss_weight, args.feats_type)))
                        else:
                            early_stopping = EarlyStopping(patience=args.patience, verbose=True,
                                                           save_path=os.path.join(global_dir_path,'checkpoint','pretrain','checkpoint_hsubgraph_{}_{}_{}_{}.pt'.
                                                           format(args._dataset, args.model_type, args.subgraph_hop_num,
                                                                  args.feats_type)))
                    else:
                        early_stopping = EarlyStopping(patience=args.patience, verbose=True,
                                                       save_path=os.path.join(global_dir_path,'checkpoint','pretrain','checkpoint_{}_{}_{}_{}.pt'.
                                                       format(args._dataset, args.model_type,args.subgraph_hop_num,args.feats_type)))

        for epoch in range(args.epoch):
            t_start = time.time()
            # training
            net.train()
            if args.model_type=='SHGN':
                logits = net(features_list, e_feat)
                if args.subgraph_hop_num != 0:
                    for i in range(args.subgraph_hop_num):
                        logits = torch.sparse.mm(torch_sparse_adj, logits)
            else:
                if args.hetero_subgraph:
                    if args.each_loss:
                        logits,logits0,logits2,logits4,logits6 = net(g, features_list, e_feat)
                    else:
                        if args._dataset=='Freebase':
                            if args.freebase_type==2:
                                logits = net(g, trans_g, features_list, e_feat)
                            else:
                                logits = net(g, features_list, e_feat)
                        else:
                            logits = net(g,features_list,e_feat)
                else:
                    logits = net(features_list)
                #subgraph embedding
                    if args.subgraph_hop_num != 0:
                        for i in range(args.subgraph_hop_num):
                            logits = torch.sparse.mm(torch_sparse_adj, logits)
            if args.hetero_pretrain:
                htrain_loss=hcompareloss(logits,hsamples,args.temperature,device)
                htrain_loss.requires_grad_(True)
                total_loss=htrain_loss
            else:
                if args.each_loss:
                    train_loss = compareloss(logits, samples, args.temperature, device) + compareloss(logits0, samples,
                                                                                                      args.temperature,
                                                                                                      device) \
                                 + compareloss(logits2, samples, args.temperature, device) + compareloss(logits4,
                                                                                                         samples,
                                                                                                         args.temperature,
                                                                                                         device) \
                                 + compareloss(logits6, samples, args.temperature, device)
                else:
                    if args.target_pretrain:
                        train_loss = tcompareloss(logits, target_samples, args.temperature, device)
                    else:
                        train_loss = compareloss(logits, samples, args.temperature, device)
                    train_loss.requires_grad_(True)
                    total_loss=train_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            t_end = time.time()
            train_time+=t_end-t_start

            print('Epoch {:05d} | Train_Loss: {:.4f} | Time: {:.4f}'.format(epoch, total_loss.item(), t_end-t_start))

            early_stopping(total_loss, net)
            if early_stopping.early_stop:
                print('Early stopping!')
                train_time = train_time / epoch
                early_stop = True
                break
        if early_stop == False:
            train_time = train_time / args.epoch

    print("####################################################")
    print('pretrain train time per epoch:',train_time)



########################        run.py文件





def subgraph_nodelist(g,subgraphs_dir):
    if os.path.exists(subgraphs_dir) == False:
        os.mkdir(subgraphs_dir)

    if os.path.exists(os.path.join(subgraphs_dir,'0.npy'))==False:
        nodenum = g.number_of_nodes()
        subgraph_list = []
        for i in range(nodenum):
            neighbors = g.successors(i).numpy().tolist()
            two_hop_neighbors = []
            for neighbor in neighbors:
                two_hop_neighbors.extend(g.successors(neighbor).numpy().tolist())
            subgraph_nodes = [i] + neighbors + two_hop_neighbors
            subgraph_nodes = np.array(list(set(subgraph_nodes)))
            subgraph_dir = os.path.join(subgraphs_dir, str(i))
            np.save(subgraph_dir, subgraph_nodes)
            subgraph_list.append(torch.tensor(subgraph_nodes))
    else:
        # Load subgraphs list
        subgraph_list = []
        file_names = [file_name for file_name in os.listdir(subgraphs_dir) if file_name.endswith('.npy')]

        sorted_file_names = sorted(file_names, key=lambda x: int(x.split('.')[0]))

        for file_name in sorted_file_names:
            file_path = os.path.join(subgraphs_dir, file_name)
            np_array = np.load(file_path)
            subgraph_nodes = torch.tensor(np_array)
            subgraph_list.append(subgraph_nodes)
    return subgraph_list


def run_model_DBLP_downstream(args):
    torch.manual_seed(args.seed_down)
    np.random.seed(args.seed_down)
    feats_type = args.feats_type_down
    index_dir=[str(args.shotnum_down),'shots',str(args.tasknum_down),'tasks']
    index = "".join(index_dir)
    features_list, adjM, labels, train_val_test_idx, dl = load_data(args.dataset_down,args.tasknum_down,args.shotnum_down,index)

    # if args.device_down==1:
    #     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # elif args.device_down==0:
    #     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = global_device

    features_list = [mat2tensor(features).to(device) for features in features_list]
    if feats_type == 0:
        in_dims = [features.shape[1] for features in features_list]
    elif feats_type == 1 or feats_type == 5:
        save = 0 if feats_type == 1 else 2
        in_dims = []#[features_list[0].shape[1]] + [10] * (len(features_list) - 1)
        for i in range(0, len(features_list)):
            if i == save:
                in_dims.append(features_list[i].shape[1])
            else:
                in_dims.append(10)
                features_list[i] = torch.zeros((features_list[i].shape[0], 10)).to(device)
    elif feats_type == 2 or feats_type == 4:
        save = feats_type - 2
        in_dims = [features.shape[0] for features in features_list]
        for i in range(0, len(features_list)):
            if i == save:
                in_dims[i] = features_list[i].shape[1]
                continue
            dim = features_list[i].shape[0]
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list[i] = torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device)
    elif feats_type == 3:
        in_dims = [features.shape[0] for features in features_list]
        for i in range(len(features_list)):
            dim = features_list[i].shape[0]
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list[i] = torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device)


    edge2type = {}
    for k in dl.links['data']:
        for u,v in zip(*dl.links['data'][k].nonzero()):
            edge2type[(u,v)] = k
    for i in range(dl.nodes['total']):
        if (i,i) not in edge2type:
            edge2type[(i,i)] = len(dl.links['count'])
    for k in dl.links['data']:
        for u,v in zip(*dl.links['data'][k].nonzero()):
            if (v,u) not in edge2type:
                edge2type[(v,u)] = k+1+len(dl.links['count'])
    g = dgl.DGLGraph(adjM+(adjM.T))
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    g = g.to(device)
    trans_g=dgl.reverse(g)

    coo_adj=adjM.tocoo()
    values = coo_adj.data
    indices = np.vstack((coo_adj.row, coo_adj.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo_adj.shape

    e_feat = []
    for u, v in zip(*g.edges()):
        u = u.cpu().item()
        v = v.cpu().item()
        e_feat.append(edge2type[(u,v)])
    e_feat = torch.tensor(e_feat, dtype=torch.long).to(device)

    eval_result={}
    eval_result['micro-f1']=[]
    eval_result['macro-f1']=[]
    train_time=0
    test_time=0

    for count in range(args.tasknum_down):
        train_time_one_task = 0
        early_stop=False
        train_labels=labels['train'][count]
        val_labels=labels['val'][count]
        train_labels = torch.LongTensor(train_labels).to(device)
        val_labels = torch.LongTensor(val_labels).to(device)
        train_idx = train_val_test_idx['train_idx'][count]
        val_idx = train_val_test_idx['val_idx'][count]
        test_idx = train_val_test_idx['test_idx']
        test_idx = np.sort(test_idx)

        for _ in range(args.repeat_down):
            num_classes = dl.labels_train['num_classes']
            if args.model_type_down == 'gat':
                heads = [args.num_heads_down] * args.num_layers_down + [1]
                net = GAT(g, in_dims, args.hidden_dim_down, num_classes, args.num_layers_down, heads, F.elu, args.dropout_down, args.dropout_down, args.slope_down, False)
            elif args.model_type_down == 'gcn':
                if args.pretrain_semantic_down:
                    net = semantic_GCN(g, in_dims, args.hidden_dim_down, num_classes, args.num_layers_down, F.elu, args.dropout_down)
                else:
                    net = GCN(g, in_dims, args.hidden_dim_down, num_classes, args.num_layers_down, F.elu, args.dropout_down)   #   初始化GNN
            elif args.model_type_down == 'gin':
                net = GIN(g, in_dims, args.hidden_dim_down, num_classes, args.num_layers_down, F.relu, args.dropout_down)
            elif args.model_type_down=='SHGN':
                num_classes = dl.labels_train['num_classes']
                heads = [args.num_heads_down] * args.num_layers_down + [1]
                net = myGAT(g, args.edge_feats_down, len(dl.links['count']) * 2 + 1, in_dims, args.hidden_dim_down, num_classes,
                            args.num_layers_down, heads, F.elu, args.dropout_down, args.dropout_down, args.slope_down, True, 0.05)

            else:
                raise Exception('{} model is not defined!'.format(args.model_type_down))
            if args.model_type_down == 'SHGN':
                net.load_state_dict(torch.load(os.path.join(global_dir_path,'checkpoint/pretrain/shgn_checkpoint_{}_{}.pt'.format(
                                                   args.dataset_down, args.num_layers_down))))
            else:
                if args.dataset_down=='Freebase':
                    if args.load_pretrain_down:
                        if args.hetero_pretrain_down:
                            if args.hetero_pretrain_subgraph_down:
                                if args.pretrain_semantic_down:
                                    net.load_state_dict(
                                        torch.load(os.path.join(global_dir_path,'checkpoint/pretrain/checkpoint_hsubgraph_semantic_{}_{}_{}_{}_{}_{}_{}.pt'.
                                                   format(args.dataset_down, args.model_type_down, args.subgraph_hop_num_down,
                                                          args.pre_loss_weight_down, args.feats_type_down,args.tuple_neg_disconnected_num_down,args.tuple_neg_unrelated_num_down))))
                                elif args.pretrain_each_loss_down:
                                    net.load_state_dict(
                                        torch.load(
                                            os.path.join(global_dir_path,'checkpoint/pretrain/checkpoint_hsubgraph_each_loss_{}_{}_{}_{}_{}_{}_{}.pt'.
                                            format(args.dataset_down, args.model_type_down, args.subgraph_hop_num_down,
                                                   args.pre_loss_weight_down, args.feats_type_down,args.tuple_neg_disconnected_num_down,args.tuple_neg_unrelated_num_down))))

                                else:
                                    net.load_state_dict(
                                        torch.load(os.path.join(global_dir_path,'checkpoint/pretrain/checkpoint_hsubgraph_{}_{}_{}_{}_{}_{}_{}.pt'.
                                                   format(args.dataset_down, args.model_type_down, args.subgraph_hop_num_down,
                                                          args.pre_loss_weight_down, args.feats_type_down,args.tuple_neg_disconnected_num_down,args.tuple_neg_unrelated_num_down))))
                            else:
                                net.load_state_dict(
                                    torch.load(os.path.join(global_dir_path,'checkpoint/pretrain/checkpoint_{}_{}_{}_{}_{}_{}_{}.pt'.
                                               format(args.dataset_down, args.model_type_down, args.subgraph_hop_num_down,
                                                      args.pre_loss_weight_down, args.feats_type_down,args.tuple_neg_disconnected_num_down,args.tuple_neg_unrelated_num_down))))
                        else:
                            if args.hetero_pretrain_subgraph_down:
                                if args.pretrain_semantic_down:
                                    net.load_state_dict(
                                        torch.load(os.path.join(global_dir_path,'checkpoint/pretrain/checkpoint_hsubgraph_semantic_{}_{}_{}_{}_{}_{}.pt'.
                                                   format(args.dataset_down, args.model_type_down, args.subgraph_hop_num_down,
                                                          args.feats_type_down,args.tuple_neg_disconnected_num_down,args.tuple_neg_unrelated_num_down))))
                                elif args.pretrain_each_loss_down:
                                    net.load_state_dict(
                                        torch.load(
                                            os.path.join(global_dir_path,'checkpoint/pretrain/checkpoint_hsubgraph_each_loss_{}_{}_{}_{}_{}_{}_{}.pt'.
                                            format(args.dataset_down, args.model_type_down, args.subgraph_hop_num_down,
                                                   args.pre_loss_weight_down, args.feats_type_down,args.tuple_neg_disconnected_num_down,args.tuple_neg_unrelated_num_down))))
                                else:
                                    net.load_state_dict(
                                        torch.load(os.path.join(global_dir_path,'checkpoint/pretrain/checkpoint_hsubgraph_{}_{}_{}_{}_{}_{}.pt'.
                                                   format(args.dataset_down, args.model_type_down, args.subgraph_hop_num_down,
                                                          args.feats_type_down,args.tuple_neg_disconnected_num_down,args.tuple_neg_unrelated_num_down))))
                            else:
                                net.load_state_dict(
                                    torch.load(os.path.join(global_dir_path,'checkpoint/pretrain/checkpoint_{}_{}_{}_{}_{}_{}.pt'.
                                               format(args.dataset_down, args.model_type_down, args.subgraph_hop_num_down,
                                                      args.feats_type_down,args.tuple_neg_disconnected_num_down,args.tuple_neg_unrelated_num_down))))
    ##################### end of if freebase#############
                else:
                    if args.load_pretrain_down:
                        if args.hetero_pretrain_down:
                            if args.hetero_pretrain_subgraph_down:
                                if args.pretrain_semantic_down:
                                    net.load_state_dict(
                                        torch.load(os.path.join(global_dir_path,'checkpoint/pretrain/checkpoint_hsubgraph_semantic_{}_{}_{}_{}_{}.pt'.
                                                   format(args.dataset_down, args.model_type_down, args.subgraph_hop_num_down,
                                                          args.pre_loss_weight_down, args.feats_type_down))))
                                elif args.pretrain_each_loss_down:
                                    net.load_state_dict(
                                        torch.load(os.path.join(global_dir_path,'checkpoint/pretrain/checkpoint_hsubgraph_each_loss_{}_{}_{}_{}_{}.pt'.
                                                   format(args.dataset_down, args.model_type_down, args.subgraph_hop_num_down,
                                                          args.pre_loss_weight_down, args.feats_type_down))))

                                else:
                                    net.load_state_dict(
                                        torch.load(os.path.join(global_dir_path,'checkpoint/pretrain/checkpoint_hsubgraph_{}_{}_{}_{}_{}.pt'.
                                                   format(args.dataset_down, args.model_type_down, args.subgraph_hop_num_down,
                                                          args.pre_loss_weight_down, args.feats_type_down))))
                            else:
                                net.load_state_dict(
                                    torch.load(os.path.join(global_dir_path,'checkpoint/pretrain/checkpoint_{}_{}_{}_{}_{}.pt'.
                                                       format(args.dataset_down, args.model_type_down,args.subgraph_hop_num_down,args.pre_loss_weight_down,args.feats_type_down))))
                        else:
                            if args.hetero_pretrain_subgraph_down:
                                if args.pretrain_semantic_down:
                                    net.load_state_dict(
                                        torch.load(os.path.join(global_dir_path,'checkpoint/pretrain/checkpoint_hsubgraph_semantic_{}_{}_{}_{}.pt'.
                                                   format(args.dataset_down, args.model_type_down, args.subgraph_hop_num_down,
                                                          args.feats_type_down))))
                                elif args.pretrain_each_loss_down:
                                    net.load_state_dict(
                                        torch.load(os.path.join(global_dir_path,'checkpoint/pretrain/checkpoint_hsubgraph_each_loss_{}_{}_{}_{}_{}.pt'.
                                                   format(args.dataset_down, args.model_type_down, args.subgraph_hop_num_down,
                                                          args.pre_loss_weight_down, args.feats_type_down))))
                                else:
                                    net.load_state_dict(
                                        torch.load(os.path.join(global_dir_path,'checkpoint/pretrain/checkpoint_hsubgraph_{}_{}_{}_{}.pt'.
                                                   format(args.dataset_down, args.model_type_down, args.subgraph_hop_num_down, args.feats_type_down))))
                            else:
                                net.load_state_dict(
                                    torch.load(os.path.join(
                                                            global_dir_path,'checkpoint','pretrain','checkpoint_{}_{}_{}_{}.pt'.
                                                       format(args.dataset_down, args.model_type_down,args.subgraph_hop_num_down,args.feats_type_down))
                                                       
                                            )
                                                       
                                                       )
            net.to(device)
            if args.pretrain_semantic_down:
                prelogits,semantic_weight=net(features_list)
            else:
                if args.model_type_down=='SHGN':
                    prelogits = net(features_list,e_feat)
                else:
                    prelogits = net(features_list)

            if args.tuning_down=='linear':
                classify=torch.nn.Linear(args.hidden_dim_down,num_classes)
            elif args.tuning_down=='gcn':
                classify=GraphConv(args.hidden_dim_down, num_classes)
            elif args.tuning_down in ('weight-sum','weight-sum-center-fixed','bottle-net'):
                if args.model_type_down=='SHGN':
                    hidden_dim=args.shgn_hidden_dim_down
                else:
                    hidden_dim=args.hidden_dim_down
                if args.add_edge_info2prompt_down:
                    classify = hnode_prompt_layer_feature_weighted_sum(hidden_dim)
                    if args.each_type_subgraph_down:
                        if args.dataset_down=='ACM':
                            if args.pretrain_semantic_down:
                                classify = acm_hnode_prompt_layer_feature_weighted_sum(hidden_dim, semantic_weight)
                            elif args.pretrain_each_loss_down:
                                classify=acm_eachloss_hnode_prompt_layer_feature_weighted_sum(hidden_dim, semantic_weight)
                            elif args.semantic_prompt_down==1:
                                classify=acm_hnode_semantic_prompt_layer_feature_weighted_sum(hidden_dim,semantic_prompt_weight=args.semantic_prompt_weight_down)
                            else:
                                classify = acm_hnode_prompt_layer_feature_weighted_sum(hidden_dim)
                        elif args.dataset_down=='DBLP':
                            if args.semantic_prompt_down==1:
                                classify=dblp_hnode_semantic_prompt_layer_feature_weighted_sum(hidden_dim,semantic_prompt_weight=args.semantic_prompt_weight_down)
                            else:
                                classify = dblp_hnode_prompt_layer_feature_weighted_sum(hidden_dim)
                        elif args.dataset_down=='Freebase':
                            if args.semantic_prompt_down==1:
                                classify=freebase_bidirection_semantic_hnode_prompt_layer_feature_weighted_sum(hidden_dim,semantic_prompt_weight=args.semantic_prompt_weight_down)
                            else:
                                if args.freebase_type_down==2:
                                    classify=freebase_bidirection_hnode_prompt_layer_feature_weighted_sum(hidden_dim)
                                elif args.freebase_type_down==1:
                                    classify=freebase_des_hnode_prompt_layer_feature_weighted_sum(hidden_dim)
                                else:
                                    classify=freebase_source_hnode_prompt_layer_feature_weighted_sum(hidden_dim)
                else:
                    if args.tuning_down=='bottle-net':
                        print("##############    bottel-net   ###############")
                        classify=node_bottle_net(args.hidden_dim_down,args.bottle_net_hidden_dim_down,args.bottle_net_output_dim_down)
                    else:
                        classify = node_prompt_layer_feature_weighted_sum(hidden_dim)
            elif args.tuning_down in ('cat'):
                classify=node_prompt_layer_feature_cat(args.cat_prompt_dim_down)
            elif args.tuning_down in ('sum'):
                if args.add_edge_info2prompt_down:
                    classify = hnode_prompt_layer_feature_sum()
                else:
                    classify=node_prompt_layer_feature_sum()
            elif args.tuning_down in ('cat_edge'):
                if args.add_edge_info2prompt_down:
                    classify = hnode_prompt_layer_feature_cat_edge(args.cat_prompt_dim_down,args.cat_hprompt_dim_down)
                else:
                    classify=node_prompt_layer_feature_cat_edge(args.cat_prompt_dim_down)
            elif args.tuning_down in ('prompt_gcn'):
                if args.add_edge_info2prompt_down:
                    classify = hprompt_gcn(args.hidden_dim_down)
                else:
                    classify=prompt_gcn(args.hidden_dim_down)

            else:
                print('tuning model does not exist')
                sys.exit()
            classify.to(device)

            if args.tuning_down!='sum':
                optimizer = torch.optim.AdamW(classify.parameters(),
                                              lr=args.lr_down, weight_decay=args.weight_decay_down)

            # training loop
            classify.train()
            early_stopping_classify = EarlyStopping(patience=args.patience_down, verbose=True,
                                           save_path=os.path.join(
                                                            global_dir_path,'checkpoint','checkpoint_{}_{}_{}_freeze_classify.pt'.format(args.dataset_down,
                                                                                             args.model_type_down,args.tuning_down,args.shotnum_down))
                                    )
            for epoch in range(args.epoch_down):
                t_start = time.time()
                # training
                net.train()
                classify.train()
                if args.add_edge_info2prompt_down:
                    if args.tuning_down == 'gcn':
                        logits = classify(g, prelogits)
                    elif args.tuning_down in ('weight-sum','weight-sum-center-fixed','bottle-net','sum','cat','cat_edge','prompt_gcn'):
                        if args.dataset_down=='Freebase':
                            if args.freebase_type_down==2:
                                logits=classify(g,trans_g,prelogits,e_feat)
                            else:
                                logits = classify(g, prelogits, e_feat)
                        else:
                            logits = classify(g, prelogits, e_feat)
                    elif args.tuning_down=='linear':
                        logits = classify(prelogits)
                else:
                    if args.tuning_down in ('gcn','weighted-sum','weight-sum-center-fixed','bottle-net','cat','sum','cat_edge','prompt_gcn'):
                        logits = classify(g, prelogits)
                    else:#linear, weight-sum-center-fixed,cat
                        logits = classify(prelogits)
                if args.tuning_down in ('gcn','linear'):
                    embedding = logits
                    logp = F.log_softmax(embedding, 1)
                    train_loss = F.nll_loss(logp[train_idx], train_labels)
                else:#args.tuning_down in ('weight-sum','weight-sum-center-fixed','cat')
                    embedding = logits[train_idx]
                    c_embedding = center_embedding(embedding, train_labels, num_classes)
                    distance = distance2center(embedding, c_embedding)
                    logp = F.log_softmax(distance, dim=1)
                    train_loss = F.nll_loss(logp, train_labels)

                if torch.isnan(train_loss).any():
                    print('embedding',embedding)
                    print('c_embedding',c_embedding)
                    print('distance',distance)
                    print('logp',logp)
                    raise ValueError("Tensor contains NaN values. Program terminated.")
                # autograd
                if args.tuning_down != 'sum':
                    optimizer.zero_grad()
                train_loss.backward(retain_graph=True)
                if args.tuning_down != 'sum':
                    optimizer.step()
                t_end = time.time()
                train_time_one_task+=t_end-t_start

                if args.tuning_down in ('weight-sum','weight-sum-center-fixed','bottle-net','cat','sum','cat_edge','prompt_gcn'):
                    _pred = torch.argmax(logp, dim=1, keepdim=False)
                    truth = train_labels.cpu().numpy()
                    output = _pred.cpu().numpy()
                    microf1 = f1_score(truth, output, average='micro')
                    macrof1 = f1_score(truth, output, average='macro')
                else:
                    _pred = torch.argmax(logp[train_idx], dim=1, keepdim=False)
                    truth = train_labels.cpu().numpy()
                    output = _pred.cpu().numpy()
                    microf1 = f1_score(truth, output, average='micro')
                    macrof1 = f1_score(truth, output, average='macro')

                # print training info
                print('Epoch {:05d} | Train_Loss {:.4f} | Microf1 {:.4f} | Macrof1 {:.4f} | Time(s) {:.4f}'.format(
                    epoch, train_loss.item(), microf1,macrof1,t_end - t_start))

                t_start = time.time()
                # validation
                classify.eval()
                with torch.no_grad():
                    # add edge info to prompt, edge info is added same as SHGN
                    if args.add_edge_info2prompt_down:
                        if args.tuning_down == 'gcn':
                            logits = classify(g, prelogits)
                        elif args.tuning_down in ('weight-sum','weight-sum-center-fixed','bottle-net','cat','sum','cat_edge','prompt_gcn'):
                            if args.dataset_down == 'Freebase':
                                if args.freebase_type_down == 2:
                                    logits = classify(g, trans_g, prelogits, e_feat)
                                else:
                                    logits = classify(g, prelogits, e_feat)
                            else:
                                logits = classify(g, prelogits, e_feat)
                        else:
                            logits = classify(prelogits)
                    else:
                        if args.tuning_down in ('gcn', 'weighted-sum', 'weight-sum-center-fixed', 'bottle-net','cat', 'sum','cat_edge','prompt_gcn'):
                            if args.pretrain_each_loss_down:
                                if args.dataset_down=='ACM':
                                    logits,logits0,logits2,logits4,logits6 = classify(g, prelogits)
                            else:
                                logits = classify(g, prelogits)
                        else:
                            logits = classify(prelogits)
                    if args.tuning_down == 'gcn':
                        logits = logits
                        logp = F.log_softmax(logits, 1)
                        val_loss = F.nll_loss(logp[val_idx], val_labels)
                    if args.tuning_down == 'linear':
                        logits = logits
                        logp = F.log_softmax(logits, 1)
                        val_loss = F.nll_loss(logp[val_idx], val_labels)
                    if args.tuning_down == 'weight-sum':
                        embedding = logits[val_idx]
                        c_embedding = center_embedding(embedding, val_labels, num_classes)
                        distance = distance2center(embedding, c_embedding)
                        #distance = 1 / F.normalize(distance, dim=1)
                        logp = F.log_softmax(distance, dim=1)
                        val_loss = F.nll_loss(logp, val_labels)
                    if args.tuning_down in ('weight-sum-center-fixed','cat','bottle-net','sum','cat_edge','prompt_gcn'):
                        #This part hasn't been finished
                        if args.pretrain_each_loss_down:
                            if args.dataset_down == 'ACM':
                                embedding = logits[val_idx]
                                distance = distance2center(embedding, c_embedding)
                                # distance = 1 / F.normalize(distance, dim=1)
                                logp = F.log_softmax(distance, dim=1)
                                embedding0 = logits0[val_idx]
                                distance0 = distance2center(embedding0, c_embedding0)
                                # distance = 1 / F.normalize(distance, dim=1)
                                logp0 = F.log_softmax(distance0, dim=1)
                                val_loss = F.nll_loss(logp, val_labels)
                        else:
                            embedding = logits[val_idx]
                            distance = distance2center(embedding, c_embedding)
                            #distance = 1 / F.normalize(distance, dim=1)
                            logp = F.log_softmax(distance, dim=1)
                            val_loss = F.nll_loss(logp, val_labels)

                    t_end = time.time()

                if args.tuning_down in ('weight-sum','weight-sum-center-fixed','bottle-net','cat','sum','cat_edge','prompt_gcn'):
                    _pred = torch.argmax(logp, dim=1, keepdim=False)
                    truth = val_labels.cpu().numpy()
                    output = _pred.cpu().numpy()
                    microf1 = f1_score(truth, output, average='micro')
                    macrof1 = f1_score(truth, output, average='macro')
                else:
                    _pred = torch.argmax(logp[val_idx], dim=1, keepdim=False)
                    truth = val_labels.cpu().numpy()
                    output = _pred.cpu().numpy()
                    microf1 = f1_score(truth, output, average='micro')
                    macrof1 = f1_score(truth, output, average='macro')

                # print validation info
                print('Epoch {:05d} | Val_Loss {:.4f} | Microf1 {:.4f} | Macrof1 {:.4f} | Time(s) {:.4f}'.format(
                    epoch, val_loss.item(), microf1,macrof1,t_end - t_start))
                # early stopping
                early_stopping_classify(val_loss,classify)
                if early_stopping_classify.early_stop:
                    print('Early stopping!')
                    train_time_one_task = train_time_one_task / epoch
                    early_stop = True
                    break
            if early_stop == False:
                train_time_one_task = train_time_one_task / args.epoch_down
            train_time += train_time_one_task
            if args.tuning_down == 'weight-sum-center-fixed':
                save_c_emb = c_embedding.detach().cpu().numpy()
                np.save(os.path.join(global_dir_path,'checkpoint','{}_task_center_embedding'.format(count)), save_c_emb)
            elif args.tuning_down == 'bottle-net':
                save_c_emb = c_embedding.detach().cpu().numpy()
                np.save(os.path.join(global_dir_path,'checkpoint','{}_task_center_embedding'.format(count)), save_c_emb)
            elif args.tuning_down == 'cat':
                save_c_emb = c_embedding.detach().cpu().numpy()
                np.save(os.path.join(global_dir_path,'checkpoint','{}_task_center_embedding_cat'.format(count)), save_c_emb)
            elif args.tuning_down == 'sum':
                save_c_emb = c_embedding.detach().cpu().numpy()
                np.save(os.path.join(global_dir_path,'checkpoint','{}_task_center_embedding_sum'.format(count)), save_c_emb)
            elif args.tuning_down == 'cat_edge':
                save_c_emb = c_embedding.detach().cpu().numpy()
                np.save(os.path.join(global_dir_path,'checkpoint','{}_task_center_embedding_cat_edge'.format(count)), save_c_emb)
            elif args.tuning_down == 'prompt_gcn':
                save_c_emb = c_embedding.detach().cpu().numpy()
                np.save(os.path.join(global_dir_path,'checkpoint','{}_task_center_embedding_prompt_gcn'.format(count)), save_c_emb)


            # testing with evaluate_results_nc
            classify.load_state_dict(torch.load(os.path.join(global_dir_path,'checkpoint','checkpoint_{}_{}_{}_freeze_classify.pt'.format(args.dataset_down, args.model_type_down,args.tuning_down,args.shotnum_down))
                                                )
                                     )

            start_test_time=time.time()
            classify.eval()
            test_logits = []
            with torch.no_grad():
                if args.add_edge_info2prompt_down:
                    if args.tuning_down=='gcn':
                        logits=classify(g,prelogits)
                    elif args.tuning_down in ('weight-sum','weight-sum-center-fixed','sum','cat','cat_edge','prompt_gcn'):
                        if args.dataset_down == 'Freebase':
                            if args.freebase_type_down == 2:
                                logits = classify(g, trans_g, prelogits, e_feat)
                            else:
                                logits = classify(g, prelogits, e_feat)
                        else:
                            logits = classify(g, prelogits, e_feat)
                    else:
                        logits=classify(prelogits)
                else:
                    if args.tuning_down in ('gcn','weighted-sum','weight-sum-center-fixed','bottle-net','cat','sum','cat_edge','prompt_gcn'):
                        logits = classify(g, prelogits)
                    else:
                        logits=classify(prelogits)
                if args.tuning_down == 'gcn':
                    logits = logits
                    test_logits = logits[test_idx]
                    pred = test_logits.cpu().numpy().argmax(axis=1)
                    onehot = np.eye(num_classes, dtype=np.int32)
                    dl.gen_file_for_evaluate(test_idx=test_idx, label=pred, file_name=f"{args.dataset_down}_{args.run_down}.txt")
                    pred = onehot[pred]
                    res = dl.evaluate(pred)
                elif args.tuning_down=='linear':
                    logits = logits
                    test_logits = logits[test_idx]
                    pred = test_logits.cpu().numpy().argmax(axis=1)
                    onehot = np.eye(num_classes, dtype=np.int32)
                    dl.gen_file_for_evaluate(test_idx=test_idx, label=pred, file_name=f"{args.dataset_down}_{args.run_down}.txt")
                    pred = onehot[pred]
                    res = dl.evaluate(pred)
                elif args.tuning_down == 'weight-sum':
                    test_label = dl.labels_test['data'][dl.labels_test['mask']]
                    if args.dataset_down != 'IMDB':
                        test_label = test_label.argmax(axis=1)
                    test_label = torch.tensor(test_label, device=device)
                    embedding = logits[test_idx]
                    c_embedding = center_embedding(embedding, test_label, num_classes)
                    distance = distance2center(embedding, c_embedding)
                    #distance = 1 / F.normalize(distance, dim=1)
                    pred = F.log_softmax(distance, dim=1)
                    _pred = torch.argmax(pred, dim=1, keepdim=False)
                    truth = test_label.cpu().numpy()
                    output = _pred.cpu().numpy()
                    res = {}
                    res['micro-f1'] = f1_score(truth, output, average='micro')
                    res['macro-f1'] = f1_score(truth, output, average='macro')
                elif args.tuning_down == 'weight-sum-center-fixed':
                    load_c_emb = np.load(os.path.join(global_dir_path,'checkpoint','{}_task_center_embedding.npy'.format(count)), allow_pickle=True)
                    c_embedding=torch.tensor(load_c_emb,device=device)
                    test_label = dl.labels_test['data'][dl.labels_test['mask']]
                    if args.dataset_down != 'IMDB':
                        test_label = test_label.argmax(axis=1)
                    test_label = torch.tensor(test_label, device=device)
                    embedding = logits[test_idx]
                    distance = distance2center(embedding, c_embedding)
                    #distance = 1 / F.normalize(distance, dim=1)
                    pred = F.log_softmax(distance, dim=1)
                    _pred = torch.argmax(pred, dim=1, keepdim=False)
                    truth = test_label.cpu().numpy()
                    output = _pred.cpu().numpy()
                    res = {}
                    res['micro-f1'] = f1_score(truth, output, average='micro')
                    res['macro-f1'] = f1_score(truth, output, average='macro')
                elif args.tuning_down == 'bottle-net':
                    load_c_emb = np.load(os.path.join(global_dir_path,'checkpoint','{}_task_center_embedding.npy'.format(count)), allow_pickle=True)
                    c_embedding=torch.tensor(load_c_emb,device=device)
                    test_label = dl.labels_test['data'][dl.labels_test['mask']]
                    if args.dataset_down != 'IMDB':
                        test_label = test_label.argmax(axis=1)
                    test_label = torch.tensor(test_label, device=device)
                    embedding = logits[test_idx]
                    distance = distance2center(embedding, c_embedding)
                    #distance = 1 / F.normalize(distance, dim=1)
                    pred = F.log_softmax(distance, dim=1)
                    _pred = torch.argmax(pred, dim=1, keepdim=False)
                    truth = test_label.cpu().numpy()
                    output = _pred.cpu().numpy()
                    res = {}
                    res['micro-f1'] = f1_score(truth, output, average='micro')
                    res['macro-f1'] = f1_score(truth, output, average='macro')
                elif args.tuning_down == 'cat':
                    load_c_emb = np.load(os.path.join(global_dir_path,'checkpoint','{}_task_center_embedding_cat.npy'.format(count)), allow_pickle=True)
                    c_embedding=torch.tensor(load_c_emb,device=device)
                    test_label = dl.labels_test['data'][dl.labels_test['mask']]
                    if args.dataset_down != 'IMDB':
                        test_label = test_label.argmax(axis=1)
                    test_label = torch.tensor(test_label, device=device)
                    embedding = logits[test_idx]
                    distance = distance2center(embedding, c_embedding)
                    #distance = 1 / F.normalize(distance, dim=1)
                    pred = F.log_softmax(distance, dim=1)
                    _pred = torch.argmax(pred, dim=1, keepdim=False)
                    truth = test_label.cpu().numpy()
                    output = _pred.cpu().numpy()
                    res = {}
                    res['micro-f1'] = f1_score(truth, output, average='micro')
                    res['macro-f1'] = f1_score(truth, output, average='macro')
                elif args.tuning_down == 'sum':
                    load_c_emb = np.load(os.path.join(global_dir_path,'checkpoint','{}_task_center_embedding_sum.npy'.format(count)), allow_pickle=True)
                    c_embedding=torch.tensor(load_c_emb,device=device)
                    test_label = dl.labels_test['data'][dl.labels_test['mask']]
                    if args.dataset_down != 'IMDB':
                        test_label = test_label.argmax(axis=1)
                    test_label = torch.tensor(test_label, device=device)
                    embedding = logits[test_idx]
                    distance = distance2center(embedding, c_embedding)
                    #distance = 1 / F.normalize(distance, dim=1)
                    pred = F.log_softmax(distance, dim=1)
                    _pred = torch.argmax(pred, dim=1, keepdim=False)
                    truth = test_label.cpu().numpy()
                    output = _pred.cpu().numpy()
                    res = {}
                    res['micro-f1'] = f1_score(truth, output, average='micro')
                    res['macro-f1'] = f1_score(truth, output, average='macro')
                elif args.tuning_down == 'cat_edge':
                    load_c_emb = np.load(os.path.join(global_dir_path,'checkpoint','{}_task_center_embedding_cat_edge.npy'.format(count)), allow_pickle=True)
                    c_embedding=torch.tensor(load_c_emb,device=device)
                    test_label = dl.labels_test['data'][dl.labels_test['mask']]
                    if args.dataset_down != 'IMDB':
                        test_label = test_label.argmax(axis=1)
                    test_label = torch.tensor(test_label, device=device)
                    embedding = logits[test_idx]
                    distance = distance2center(embedding, c_embedding)
                    #distance = 1 / F.normalize(distance, dim=1)
                    pred = F.log_softmax(distance, dim=1)
                    _pred = torch.argmax(pred, dim=1, keepdim=False)
                    truth = test_label.cpu().numpy()
                    output = _pred.cpu().numpy()
                    res = {}
                    res['micro-f1'] = f1_score(truth, output, average='micro')
                    res['macro-f1'] = f1_score(truth, output, average='macro')
                elif args.tuning_down == 'prompt_gcn':
                    load_c_emb = np.load(os.path.join(global_dir_path,'checkpoint','{}_task_center_embedding_prompt_gcn.npy'.format(count)), allow_pickle=True)
                    c_embedding=torch.tensor(load_c_emb,device=device)
                    test_label = dl.labels_test['data'][dl.labels_test['mask']]
                    if args.dataset_down != 'IMDB':
                        test_label = test_label.argmax(axis=1)
                    test_label = torch.tensor(test_label, device=device)
                    embedding = logits[test_idx]
                    distance = distance2center(embedding, c_embedding)
                    #distance = 1 / F.normalize(distance, dim=1)
                    pred = F.log_softmax(distance, dim=1)
                    _pred = torch.argmax(pred, dim=1, keepdim=False)
                    truth = test_label.cpu().numpy()
                    output = _pred.cpu().numpy()
                    res = {}
                    res['micro-f1'] = f1_score(truth, output, average='micro')
                    res['macro-f1'] = f1_score(truth, output, average='macro')

            end_test_time=time.time()
            test_time+=end_test_time-start_test_time
            eval_result['micro-f1'].append(res['micro-f1'])
            eval_result['macro-f1'].append(res['macro-f1'])
    eval_result['micro-f1']=np.array(eval_result['micro-f1'])
    eval_result['macro-f1']=np.array(eval_result['macro-f1'])
    print("####################################################")
    print('microf mean: ', np.mean(eval_result['micro-f1']), 'acc std: ', np.std(eval_result['micro-f1']))
    print('macroF mean: ', np.mean(eval_result['macro-f1']), 'macroF std: ', np.std(eval_result['macro-f1']))
    print('downstream train time per epoch:',train_time/args.tasknum_down)
    print('downstream test time per epoch:',test_time/args.tasknum_down)



