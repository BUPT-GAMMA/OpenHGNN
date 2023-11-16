import datetime
import dgl
import errno
import numpy as np
import os
import pickle
import random
import torch as th

from dgl import sparse as dglsp
from dgl.data.utils import download, get_download_dir, _get_dgl_url
from pprint import pprint
from scipy import sparse
from scipy import io as sio
import dgl.sparse

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = th.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = th.from_numpy(sparse_mx.data)
    shape = th.Size(sparse_mx.shape)
    return th.sparse.FloatTensor(indices, values, shape)


"""
It's the dataset from HAN.
Refer to https://github.com/dmlc/dgl/blob/master/examples/pytorch/han/utils.py
"""


def load_acm(remove_self_loop):
    url = 'dataset/ACM3025.pkl'
    data_path = './openhgnn/dataset/ACM3025.pkl'
    download(_get_dgl_url(url), path=data_path)

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    labels, features = th.from_numpy(data['label'].todense()).long(), \
                       th.from_numpy(data['feature'].todense()).float()
    num_classes = labels.shape[1]
    labels = labels.nonzero()[:, 1]

    if remove_self_loop:
        num_nodes = data['label'].shape[0]
        data['PAP'] = sparse.csr_matrix(data['PAP'] - np.eye(num_nodes))
        data['PLP'] = sparse.csr_matrix(data['PLP'] - np.eye(num_nodes))

    # Adjacency matrices for meta path based neighbors
    # (Mufei): I verified both of them are binary adjacency matrices with self loops
    author_g = dgl.from_scipy(data['PAP'])
    subject_g = dgl.from_scipy(data['PLP'])
    gs = [author_g, subject_g]

    train_idx = th.from_numpy(data['train_idx']).long().squeeze(0)
    val_idx = th.from_numpy(data['val_idx']).long().squeeze(0)
    test_idx = th.from_numpy(data['test_idx']).long().squeeze(0)

    num_nodes = author_g.number_of_nodes()
    train_mask = get_binary_mask(num_nodes, train_idx)
    val_mask = get_binary_mask(num_nodes, val_idx)
    test_mask = get_binary_mask(num_nodes, test_idx)

    print('dataset loaded')
    pprint({
        'dataset': 'ACM',
        'train': train_mask.sum().item() / num_nodes,
        'val': val_mask.sum().item() / num_nodes,
        'test': test_mask.sum().item() / num_nodes
    })

    return gs, features, labels, num_classes, train_idx, val_idx, test_idx, \
           train_mask, val_mask, test_mask


def load_acm_raw(remove_self_loop):
    assert not remove_self_loop
    url = 'dataset/ACM.mat'
    data_path = './openhgnn/dataset/ACM.mat'
    if not os.path.exists(data_path):
        download(_get_dgl_url(url), path=data_path)

    data = sio.loadmat(data_path)
    p_vs_l = data['PvsL']  # paper-field?
    p_vs_a = data['PvsA']  # paper-author
    p_vs_t = data['PvsT']  # paper-term, bag of words
    p_vs_c = data['PvsC']  # paper-conference, labels come from that

    # We assign
    # (1) KDD papers as class 0 (data mining),
    # (2) SIGMOD and VLDB papers as class 1 (database),
    # (3) SIGCOMM and MOBICOMM papers as class 2 (communication)
    conf_ids = [0, 1, 9, 10, 13]
    label_ids = [0, 1, 2, 2, 1]

    p_vs_c_filter = p_vs_c[:, conf_ids]
    p_selected = (p_vs_c_filter.sum(1) != 0).A1.nonzero()[0]
    p_vs_l = p_vs_l[p_selected]
    p_vs_a = p_vs_a[p_selected]
    p_vs_t = p_vs_t[p_selected]
    p_vs_c = p_vs_c[p_selected]

    hg = dgl.heterograph({
        ('paper', 'pa', 'author'): p_vs_a.nonzero(),
        ('author', 'ap', 'paper'): p_vs_a.transpose().nonzero(),
        ('paper', 'pf', 'field'): p_vs_l.nonzero(),
        ('field', 'fp', 'paper'): p_vs_l.transpose().nonzero()
    })

    features = th.FloatTensor(p_vs_t.toarray())

    pc_p, pc_c = p_vs_c.nonzero()
    labels = np.zeros(len(p_selected), dtype=np.int64)
    for conf_id, label_id in zip(conf_ids, label_ids):
        labels[pc_p[pc_c == conf_id]] = label_id
    labels = th.LongTensor(labels)

    num_classes = 3

    float_mask = np.zeros(len(pc_p))
    for conf_id in conf_ids:
        pc_c_mask = (pc_c == conf_id)
        float_mask[pc_c_mask] = np.random.permutation(np.linspace(0, 1, pc_c_mask.sum()))
    train_idx = np.where(float_mask <= 0.2)[0]
    val_idx = np.where((float_mask > 0.2) & (float_mask <= 0.3))[0]
    test_idx = np.where(float_mask > 0.3)[0]

    num_nodes = hg.number_of_nodes('paper')
    train_mask = get_binary_mask(num_nodes, train_idx)
    val_mask = get_binary_mask(num_nodes, val_idx)
    test_mask = get_binary_mask(num_nodes, test_idx)
    hg.nodes['paper'].data['h'] = features
    hg.nodes['paper'].data['labels'] = labels
    hg.nodes['paper'].data['train_mask'] = train_mask
    hg.nodes['paper'].data['val_mask'] = val_mask
    hg.nodes['paper'].data['test_mask'] = test_mask

    return hg, 'paper', num_classes, features.shape[1]


def get_binary_mask(total_size, indices):
    mask = th.zeros(total_size)
    mask[indices] = 1
    return mask.to(th.bool)


def generate_random_hg(num_nodes_dict, num_edges_dict):
    data_dict = {}
    for etype, num in num_edges_dict.items():
        data_dict[etype] = th.randint(low=0, high=num_nodes_dict[etype[0]], size=(num,)), \
                           th.randint(low=0, high=num_nodes_dict[etype[2]], size=(num,))
    return dgl.heterograph(data_dict=data_dict, num_nodes_dict=num_nodes_dict)

def to_symmetric(self):
    N = max(self.shape[0], self.shape[1])

    row, col = self.coo()
    idx = col.new_full((2 * col.numel() + 1,), -1)
    idx[1:row.numel() + 1] = row
    idx[row.numel() + 1:] = col
    idx[1:] *= N
    idx[1:row.numel() + 1] += col
    idx[row.numel() + 1:] += row

    idx, perm = idx.sort()
    mask = idx[1:] > idx[:-1]
    perm = perm[1:].sub_(1)
    idx = perm[mask]

    new_row = th.cat([row, col], dim=0)[idx]
    new_col = th.cat([col, row], dim=0)[idx]
    return dglsp.spmatrix(th.stack((new_row, new_col)), shape = (self.shape[0], self.shape[1]))
def row_norm(self) -> dgl.sparse.SparseMatrix:
    rownum, colnum = self.shape
    nodenum = self.nnz
    row = self.row
    rowptr, colind ,_= self.csr()
    rowcnt = rowptr[1:] - rowptr[:-1]
    nw = 0
    val = th.ones(nodenum)
    for k in range(rownum):
        val[nw:nw + rowcnt[k].item()] = 1. / rowcnt[k]
        nw += rowcnt[k].item()
    return dglsp.from_csr(rowptr, colind, val, shape = (rownum, colnum))