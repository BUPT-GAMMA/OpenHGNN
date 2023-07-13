import os

import numpy as np
import scipy.sparse as sp
import torch
import dgl
from dgl.data import download, extract_archive

from . import BaseDataset, register_dataset


@register_dataset('mag_dataset')
class MagDataset(BaseDataset):
    def __init__(self, *args, **kwargs):
        train_percent = kwargs.pop('train_percent', 0.1)
        super(MagDataset, self).__init__(*args, **kwargs)
        self.data_path = 'openhgnn/dataset/data/mag.zip'
        self.raw_dir = 'openhgnn/dataset/data/'
        self.g_path = 'openhgnn/dataset/data/mag/mag/'
        self.name = 'mag'
        self.url = 'https://raw.githubusercontent.com/MoonLight-Sherry/OpenHGNN/main/openhgnn/dataset/data/test/mag.zip'
        # self.url = 'http://localhost:8000/mag.zip'
        self.train_percent = train_percent
        self.download()
        self.load_odbmag_4017(self.train_percent)



    def load_odbmag_4017(self, train_percent):
        feats = np.load(self.g_path + 'feats.npz', allow_pickle=True)
        p_ft = feats['p_ft']
        a_ft = feats['a_ft']
        i_ft = feats['i_ft']
        f_ft = feats['f_ft']

        ft_dict = {}
        ft_dict['p'] = torch.FloatTensor(p_ft)
        ft_dict['a'] = torch.FloatTensor(a_ft)
        ft_dict['i'] = torch.FloatTensor(i_ft)
        ft_dict['f'] = torch.FloatTensor(f_ft)

        p_label = np.load(self.g_path + 'p_label.npy', allow_pickle=True)
        p_label = torch.LongTensor(p_label)

        idx_train_p, idx_val_p, idx_test_p = train_val_test_split(p_label.shape[0], train_percent)

        label = {}
        label['p'] = [p_label, idx_train_p, idx_val_p, idx_test_p]

        sp_a_i = sp.load_npz(self.g_path + 'norm_sp_a_i.npz')
        sp_i_a = sp.load_npz(self.g_path + 'norm_sp_i_a.npz')
        sp_a_p = sp.load_npz(self.g_path + 'norm_sp_a_p.npz')
        sp_p_a = sp.load_npz(self.g_path + 'norm_sp_p_a.npz')
        sp_p_f = sp.load_npz(self.g_path + 'norm_sp_p_f.npz')
        sp_f_p = sp.load_npz(self.g_path + 'norm_sp_f_p.npz')
        sp_p_cp = sp.load_npz(self.g_path + 'norm_sp_p_cp.npz')
        sp_cp_p = sp.load_npz(self.g_path + 'norm_sp_cp_p.npz')

        adj_dict = {'p': {}, 'a': {}, 'i': {}, 'f': {}}
        adj_dict['a']['i'] = sp_coo_2_sp_tensor(sp_a_i.tocoo())
        adj_dict['a']['p'] = sp_coo_2_sp_tensor(sp_a_p.tocoo())
        adj_dict['i']['a'] = sp_coo_2_sp_tensor(sp_i_a.tocoo())
        adj_dict['f']['p'] = sp_coo_2_sp_tensor(sp_f_p.tocoo())
        adj_dict['p']['a'] = sp_coo_2_sp_tensor(sp_p_a.tocoo())
        adj_dict['p']['f'] = sp_coo_2_sp_tensor(sp_p_f.tocoo())
        adj_dict['p']['citing_p'] = sp_coo_2_sp_tensor(sp_p_cp.tocoo())
        adj_dict['p']['cited_p'] = sp_coo_2_sp_tensor(sp_cp_p.tocoo())

        self.label = label
        self.ft_dict = ft_dict
        self.adj_dict = adj_dict

        graph_data = {}

        # Add entries for 'a' -> 'i' relationship
        source_type = 'a'
        edge_type = 'i'
        target_type = 'i'
        coo_matrix = sp_a_i.tocoo()
        source_nodes = torch.from_numpy(coo_matrix.row)
        target_nodes = torch.from_numpy(coo_matrix.col)
        graph_data[(source_type, edge_type, target_type)] = (source_nodes, target_nodes)

        # Add entries for 'a' -> 'p' relationship
        source_type = 'a'
        edge_type = 'p'
        target_type = 'p'
        coo_matrix = sp_a_p.tocoo()
        source_nodes = torch.from_numpy(coo_matrix.row)
        target_nodes = torch.from_numpy(coo_matrix.col)
        graph_data[(source_type, edge_type, target_type)] = (source_nodes, target_nodes)

        # Add entries for 'i' -> 'a' relationship
        source_type = 'i'
        edge_type = 'a'
        target_type = 'a'
        coo_matrix = sp_i_a.tocoo()
        source_nodes = torch.from_numpy(coo_matrix.row)
        target_nodes = torch.from_numpy(coo_matrix.col)
        graph_data[(source_type, edge_type, target_type)] = (source_nodes, target_nodes)

        # Add entries for 'f' -> 'p' relationship
        source_type = 'f'
        edge_type = 'p'
        target_type = 'p'
        coo_matrix = sp_f_p.tocoo()
        source_nodes = torch.from_numpy(coo_matrix.row)
        target_nodes = torch.from_numpy(coo_matrix.col)
        graph_data[(source_type, edge_type, target_type)] = (source_nodes, target_nodes)

        # Add entries for 'p' -> 'a' relationship
        source_type = 'p'
        edge_type = 'a'
        target_type = 'a'
        coo_matrix = sp_p_a.tocoo()
        source_nodes = torch.from_numpy(coo_matrix.row)
        target_nodes = torch.from_numpy(coo_matrix.col)
        graph_data[(source_type, edge_type, target_type)] = (source_nodes, target_nodes)

        # Add entries for 'p' -> 'f' relationship
        source_type = 'p'
        edge_type = 'f'
        target_type = 'f'
        coo_matrix = sp_p_f.tocoo()
        source_nodes = torch.from_numpy(coo_matrix.row)
        target_nodes = torch.from_numpy(coo_matrix.col)
        graph_data[(source_type, edge_type, target_type)] = (source_nodes, target_nodes)

        # Add entries for 'p' -> 'citing_p' relationship
        source_type = 'p'
        edge_type = 'citing_p'
        target_type = 'p'
        coo_matrix = sp_p_cp.tocoo()
        source_nodes = torch.from_numpy(coo_matrix.row)
        target_nodes = torch.from_numpy(coo_matrix.col)
        graph_data[(source_type, edge_type, target_type)] = (source_nodes, target_nodes)

        # Add entries for 'p' -> 'cited_p' relationship
        source_type = 'p'
        edge_type = 'cited_p'
        target_type = 'p'
        coo_matrix = sp_cp_p.tocoo()
        source_nodes = torch.from_numpy(coo_matrix.row)
        target_nodes = torch.from_numpy(coo_matrix.col)
        graph_data[(source_type, edge_type, target_type)] = (source_nodes, target_nodes)

        # 转换成 DGL 的异构图数据格式
        self.g = dgl.heterograph(graph_data)
        # 将特征数据添加到异构图的节点上
        for ntype in self.ft_dict:
            self.g.nodes[ntype].data['features'] = self.ft_dict[ntype]

    def download(self):
        # download raw data to local disk
        # path to store the file
        if os.path.exists(self.data_path):  # pragma: no cover
           pass
        else:
            file_path = os.path.join(self.raw_dir)
            # download file
            download(self.url, path=file_path)
        extract_archive(self.data_path, os.path.join(self.raw_dir, self.name))



def train_val_test_split(label_shape, train_percent):
    rand_idx = np.random.permutation(label_shape)
    val_percent = (1.0 - train_percent) / 2
    idx_train = torch.LongTensor(rand_idx[int(label_shape * 0.0): int(label_shape * train_percent)])
    idx_val = torch.LongTensor(
        rand_idx[int(label_shape * train_percent): int(label_shape * (train_percent + val_percent))])
    idx_test = torch.LongTensor(rand_idx[int(label_shape * (train_percent + val_percent)): int(label_shape * 1.0)])
    return idx_train, idx_val, idx_test

def sp_coo_2_sp_tensor(sp_coo_mat):
    indices = torch.from_numpy(np.vstack((sp_coo_mat.row, sp_coo_mat.col)).astype(np.int64))
    values = torch.from_numpy(sp_coo_mat.data)
    shape = torch.Size(sp_coo_mat.shape)
    return torch.sparse.FloatTensor(indices, values, shape)