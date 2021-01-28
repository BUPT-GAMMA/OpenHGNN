import torch as th
import dgl
import numpy as np


def load_HIN():
    data_path = '../data/acm/edge.txt'
    with open(data_path) as f:
        lines = f.readlines()
        relation_dict = {}
        for line in lines:
            token = line.strip('\n').split("\t")
            relation = token[2]
            if relation not in relation_dict:
                relation_dict[relation] = [0, 1]
                relation_dict[relation][0] = []
                relation_dict[relation][1] = []
            relation_dict[relation][0].append(int(token[0]))
            relation_dict[relation][1].append(int(token[1]))
    data_dict = {}
    for i in relation_dict:
        break
    for re, values in relation_dict.items():
        s = (re[0], re, re[-1])
        data_dict[s] = (th.tensor(values[0]), th.tensor(values[1]))
    dw_features = np.load('../data/acm/dw_emb_features.npy')
    g = dgl.heterograph(data_dict, idtype=th.int64, device='cpu')
    y = g.nodes('a')
    z = g.nodes['s'].data['h'] = th.ones(60, 1)
    x = th.from_numpy((dw_features))[0]
    y = th.tensor(dw_features)[:4019]
    g.ndata['h'] = {'p': th.tensor(dw_features)[:4019], 'a': th.tensor(dw_features)[4019:11186], 's':th.tensor(dw_features)[11186:]}
    return g