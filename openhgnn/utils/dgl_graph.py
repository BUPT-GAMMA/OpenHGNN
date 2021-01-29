import torch as th
import dgl
import numpy as np
from dgl.data.utils import load_graphs


def load_dgl_graph(path_file):
    g,_ = load_graphs(path_file)
    return g[0]
def load_HIN():
    data_path = './openhgnn/dataset/acm_graph.bin'
    g = load_dgl_graph(data_path)
    return g