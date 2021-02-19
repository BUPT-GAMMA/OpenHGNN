import torch as th
import dgl
import numpy as np
from dgl.data.utils import load_graphs


def load_dgl_graph(path_file):
    g, _ = load_graphs(path_file)
    return g[0]
def load_HIN(dataset):
    if dataset == 'acm':
        data_path = './openhgnn/dataset/acm_graph.bin'
    elif dataset == 'imdb':
        data_path = './openhgnn/dataset/imdb_graph.bin'
    elif dataset == 'acm1':
        data_path = './openhgnn/dataset/acm_graph1.bin'
    g = load_dgl_graph(data_path)
    return g