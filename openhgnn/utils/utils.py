import dgl
from dgl import backend as F
import torch as th
from scipy.sparse import coo_matrix
import numpy as np
import random
from . import load_HIN, load_KG


def get_nodes_dict(hg):
    n_dict = {}
    for n in hg.ntypes:
        n_dict[n] = hg.num_nodes(n)
    return n_dict

def get_idx(hg, g, category):
    train_mask = hg.nodes[category].data.pop('train_mask')
    test_mask = hg.nodes[category].data.pop('test_mask')
    train_idx = th.nonzero(train_mask, as_tuple=False).squeeze()
    test_idx = th.nonzero(test_mask, as_tuple=False).squeeze()
    labels = hg.nodes[category].data.pop('labels')
    # get target category id
    node_ids = th.arange(g.number_of_nodes())
    category_id = len(hg.ntypes)
    for i, ntype in enumerate(hg.ntypes):
        if ntype == category:
            category_id = i
    # find out the target node ids in g
    node_tids = g.ndata[dgl.NTYPE]
    loc = (node_tids == category_id)
    target_idx = node_ids[loc]
    return target_idx, train_idx, test_idx, labels


def build_dataset(model_name, dataset_name):
    # load the graph(HIN or KG)
    if model_name in ['GTN', 'NSHE', 'HetGNN']:
        g, category, num_classes = load_HIN(dataset_name)
    elif model_name in ['RSHN', 'RGCN', 'CompGCN']:
        g, category, num_classes = load_KG(dataset_name)
    return g, category, num_classes


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)

def com_mult(a, b):
    r1, i1 = a[..., 0], a[..., 1]
    r2, i2 = b[..., 0], b[..., 1]
    return th.stack([r1 * r2 - i1 * i2, r1 * i2 + i1 * r2], dim=-1)


def conj(a):
    a[..., 1] = -a[..., 1]
    return a


def ccorr(a, b):
    """
    Compute circular correlation of two tensors.
    Parameters
    ----------
    a: Tensor, 1D or 2D
    b: Tensor, 1D or 2D
    Notes
    -----
    Input a and b should have the same dimensions. And this operation supports broadcasting.
    Returns
    -------
    Tensor, having the same dimension as the input a.
    """
    import torch.fft as fft
    return th.irfft(com_mult(conj(th.rfft(a, 1)), th.rfft(b, 1)), 1, signal_sizes=(a.shape[-1],))


def extract_edge_with_id_edge(g):
    # input a homogeneous graph
    # return tensor with shape of [2,num_edges]
    edges = g.edges()
    edata = g.edata['_TYPE']
    num_edge_type = th.max(edata).item()
    ctx = F.context(edges[0])
    dtype = F.dtype(edges[0])
    A = []
    for i in range(num_edge_type + 1):
        index = th.nonzero(edata == i).squeeze()
        e_0 = edges[0][index]
        e_1 = edges[1][index]  # edges is  tuple
        e = th.stack((e_0, e_1), dim=0)
        # turn the edge type(tuple) to tensor
        values = th.ones(e.shape[1], device=ctx)
        A.append((e, values))
    x = th.arange(0, g.num_nodes(), dtype=dtype, device=ctx)
    id_edge = th.stack((x, x), dim=0)
    values = th.ones(id_edge.shape[1], device=ctx)
    A.append((id_edge, values))
    return A


def extract_mtx_with_id_edge(g):
    # input a homogeneous graph
    # return tensor with shape of [2,num_edges]
    edges = g.edges()
    edata = g.edata['_TYPE']
    num_edge_type = th.max(edata).item()
    ctx = F.context(edges[0])
    dtype = F.dtype(edges[0])
    A = []
    num_nodes = g.num_nodes()
    for i in range(num_edge_type + 1):
        index = th.nonzero(edata == i).squeeze()
        e_0 = edges[0][index].to('cpu').numpy()
        e_1 = edges[1][index].to('cpu').numpy()
        values = np.ones(e_0.shape[0])
        m = coo_matrix((values, (e_0, e_1)), shape=(num_nodes, num_nodes))
        m = th.from_numpy(m.todense()).type(th.FloatTensor).unsqueeze(0)
        if 0 == i:
            A = m
        else:
            A = th.cat([A, m], dim=0)
    m = th.eye(num_nodes).unsqueeze(0)
    A = th.cat([A, m], dim=0)
    return A.to(ctx)


def h2dict(h, hdict):
    pre = 0
    for i, value in hdict.items():
        hdict[i] = h[pre:value.shape[0] + pre]
        pre += value.shape[0]
    return hdict


def print_dict(d, end_string='\n\n'):
    for key in d.keys():
        if isinstance(d[key], dict):
            print('\n', end='')
            print_dict(d[key], end_string='')
        elif isinstance(d[key], int):
            print('{}: {:04d}'.format(key, d[key]), end=', ')
        elif isinstance(d[key], float):
            print('{}: {:.4f}'.format(key, d[key]), end=', ')
        else:
            print('{}: {}'.format(key, d[key]), end=', ')
    print(end_string, end='')
