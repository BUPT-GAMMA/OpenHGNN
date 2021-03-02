import dgl
from dgl import backend as F
import torch as th
from scipy.sparse import coo_matrix
import numpy as np

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
        e_1 = edges[1][index]   # edges is  tuple
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
        hdict[i] = h[pre:value.shape[0]+pre]
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
