import dgl
from dgl.dataloading import *
from dgl import transform
import numpy as np
import pickle
import os
import torch as th
import warnings
from operator import itemgetter
from openhgnn.models.MAGNN import mp_instance_sampler

class MAGNN_Sampler(dgl.dataloading.BlockSampler):
    def __init__(self, g, n_layers, metapath_list, dataset='imdb4MAGNN', return_eids=False):
        super().__init__(n_layers, return_eids=return_eids)
        self.dataset = dataset
        self.metapath_list = metapath_list
        self.mp_inst = mp_instance_sampler(g, self.metapath_list, self.dataset)

    def sample_frontier(self, block_id, g, seed_nodes):
        '''

        Parameters
        ----------
        block_id : None
        g :
            the dgl heterogenous graph

        seed_nodes : dict
            {node_type: node_idx_list}, e.g {'M': [0, 1, 2]}. Because the MAGNN is just applied on the node
            classification of only one node type, so the key in `seed_nodes` should be only one.

        Returns
        -------
        frontier
            The dgl heterogenous graph.

        Notes
        -----
        '''

        # TODO: Too many loops in sample_frontier(), may need some optimization

        graph_data = {}
        # obtain the target ntype
        ntypes = list(seed_nodes.keys())
        # obtain the metapath types with ntype as dst node type
        for ntype in ntypes:
            target_mp_types = np.array(self.metapath_list)[[metapath[0] == ntype for metapath in self.metapath_list]]

            for metapath in target_mp_types:  # the metapath instances of the certain metapath
                _mp_inst = np.isin(self.mp_inst[metapath][:, 0], seed_nodes[ntype])
                _mp_inst = th.tensor(self.mp_inst[metapath][_mp_inst])

                for i in range(len(metapath) - 1):  # traverse the metapath instances to build graph
                    edges_idx = th.unique(_mp_inst[:, [i, i+1]], dim=0)
                    graph_data[(metapath[i+1], metapath[i+1] + '-' + metapath[i], metapath[i])] = \
                        (edges_idx[:, 1], edges_idx[:, 0])
                    del edges_idx

                del _mp_inst

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
    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        '''

        Here we're trying to overwrite the original method in BlockSampler, which is not fitted here.
        Because there's one line of codes `block = transform.to_block(frontier, seed_nodes)` specifying the dst nodes,
        which makes the `block` loss all n-hop (n > 2) neighbors. So here we'd like to cancel this specification.

        The method is basically the same as the original method with `block = transform.to_block(frontier)` as the only
        exception
        '''

        blocks = []
        exclude_eids = (
            _tensor_or_dict_to_numpy(exclude_eids) if exclude_eids is not None else None)
        for block_id in reversed(range(self.num_layers)):
            frontier = self.sample_frontier(block_id, g, seed_nodes)

            # Removing edges from the frontier for link prediction training falls
            # into the category of frontier postprocessing
            if exclude_eids is not None:
                parent_eids = frontier.edata[EID]
                parent_eids_np = _tensor_or_dict_to_numpy(parent_eids)
                located_eids = _locate_eids_to_exclude(parent_eids_np, exclude_eids)
                if not isinstance(located_eids, Mapping):
                    # (BarclayII) If frontier already has a EID field and located_eids is empty,
                    # the returned graph will keep EID intact.  Otherwise, EID will change
                    # to the mapping from the new graph to the old frontier.
                    # So we need to test if located_eids is empty, and do the remapping ourselves.
                    if len(located_eids) > 0:
                        frontier = transform.remove_edges(
                            frontier, located_eids, store_ids=True)
                        frontier.edata[EID] = F.gather_row(parent_eids, frontier.edata[EID])
                else:
                    # (BarclayII) remove_edges only accepts removing one type of edges,
                    # so I need to keep track of the edge IDs left one by one.
                    new_eids = parent_eids.copy()
                    for k, v in located_eids.items():
                        if len(v) > 0:
                            frontier = transform.remove_edges(
                                frontier, v, etype=k, store_ids=True)
                            new_eids[k] = F.gather_row(parent_eids[k], frontier.edges[k].data[EID])
                    frontier.edata[EID] = new_eids

            block = transform.to_block(frontier)

            if self.return_eids:
                assign_block_eids(block, frontier)

            # seed_nodes = block.srcdata[NID]
            # Here because dgl makes dst nodes forcibly appear in src nodes, actually block.srcdata[EID] = block.data[EID]
            seed_nodes = {ntype: block.srcnodes[ntype].data[NID] for ntype in block.srctypes}

            blocks.insert(0, block)
        return blocks

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    dataset = 'imdb4MAGNN'
    g_addr = 'openhgnn/dataset/' + '{}'.format(dataset) + '/graph.bin'
    metapath_list = ['MDM', 'MAM', 'DMD', 'DMAMD', 'AMA', 'AMDMA']

    g, _ = dgl.load_graphs(g_addr)
    g = g[0]
    nids = {'M': th.tensor([20, 2, 4, 1, 10, 6])}

    sampler = MAGNN_Sampler(g, n_layers=1, metapath_list=metapath_list, dataset='imdb4MAGNN', return_eids=True)
    dataloader = dgl.dataloading.NodeDataLoader(
        g=g, nids=nids, block_sampler=sampler, batch_size=2, shuffle=True, drop_last=False,
        num_workers=1
    )
    input_nodes, output_nodes, block = next(iter(dataloader))

    # TODO: test if exclude_eids make sense





