from abc import ABC

import dgl
from dgl.dataloading import *
from dgl import transform
import numpy as np
import pickle
import os
import torch as th
from torch.utils.data import Dataset, DataLoader
import warnings
from operator import itemgetter
from openhgnn.models.MAGNN import mp_instance_sampler, mini_mp_instance_sampler

class MAGNN_sampler(Dataset):
    def __init__(self, g, n_layers, metapath_list, dataset_name='imdb4MAGNN'):
        super().__init__()
        self.dataset_name = dataset_name
        self.metapath_list = metapath_list
        self.n_layers = n_layers
        self.mp_inst = mp_instance_sampler(g, self.metapath_list, self.dataset_name)

    def __getitem__(self, idx):

        # TODO: Too many loops in sample_frontier(), may need some optimization

        graph_data = {}
        mini_mp_inst = mini_mp_instance_sampler(seed_nodes=seed_nodes, mp_instances=self.mp_inst)
        for metapath in mini_mp_inst.keys():
            _mini_mp_inst = th.tensor(mini_mp_inst[metapath])
            for i in range(len(metapath) - 1): # TODO: THREE NEST LOOPS!!! CAN IT BE OPTIMIZED?
                edges_idx = th.unique(_mini_mp_inst[:, [i, i + 1]], dim=0)
                graph_data[(metapath[i + 1], metapath[i + 1] + '-' + metapath[i], metapath[i])] = \
                           (edges_idx[:, 1], edges_idx[:, 0])
                del edges_idx
            del _mini_mp_inst

        num_nodes_dict = {}
        for ntype in g.ntypes:
            num_nodes_dict[ntype] = g.number_of_nodes(ntype)

        frontier = dgl.heterograph(graph_data, num_nodes_dict)
        for ntype in frontier.ntypes:
            frontier.nodes[ntype].data.update(g.nodes[ntype].data)
        for etype in frontier.etypes:
            edge_ids = g.edge_ids(frontier.edges(etype=etype)[0],
                                  frontier.edges(etype=etype)[1],
                                  etype=etype)
            frontier.edges[etype].data[dgl.EID] = edge_ids
        return frontier


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    dataset_name = 'imdb4MAGNN'
    g_addr = 'openhgnn/dataset_name/' + '{}'.format(dataset_name) + '/graph.bin'
    metapath_list = ['MDM', 'MAM', 'DMD', 'DMAMD', 'AMA', 'AMDMA']

    g, _ = dgl.load_graphs(g_addr)
    g = g[0]
    nids = {'M': th.tensor([20, 2, 4, 1, 10, 6])}

    sampler = MAGNN_sampler(g, n_layers=2, metapath_list=metapath_list, dataset_name='imdb4MAGNN')
    dataloader = dgl.dataloading.NodeDataLoader(
        g=g, nids=nids, block_sampler=sampler, batch_size=4, shuffle=True, drop_last=False,
        num_workers=1
    )
    input_nodes, output_nodes, block = next(iter(dataloader))
    print(1)
    # TODO: test if exclude_eids make sense





