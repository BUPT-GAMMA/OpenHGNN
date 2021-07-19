import dgl
import numpy as np
import torch as th
from torch.utils.data import Dataset
from openhgnn.models.MAGNN import mp_instance_sampler, mini_mp_instance_sampler


class MAGNN_sampler(Dataset):
    def __init__(self, g, n_layers, category, metapath_list, dataset_name='imdb4MAGNN'):
        self.g = g
        self.dataset_name = dataset_name
        self.metapath_list = metapath_list
        self.n_layers = n_layers
        self.category = category
        self.mp_inst = mp_instance_sampler(g, self.metapath_list, self.dataset_name)

    def __getitem__(self, idx):
        '''
        Parameters
        ----------
        idx : int
            The node ids of ntype as self.category, that is, the original seed_nodes is {self.category: idx}.

        Returns
        -------
        dict
            The nids sampled by the MAGNN_sampler from the original graph. The nids are of original version.
        dict
            The metapath instances corresponding to the sampled graph. The nids in mp instances are of original version.
        DGLGraph
            The original heterogeneous subgraph.
        '''
        # TODO: Too many loops in sample_frontier(), may need some optimization

        # def convert_mp_nids(old_metapath_dict, old_nids):
        #     # convert the old_nids in metapath instances to the new_nids in the subgraph
        #     for meta in old_metapath_dict.keys():
        #         for i, ntype in enumerate(meta):
        #             old_metapath_dict[meta][:, i] = \
        #                 np.array(
        #                     list(map(lambda x: np.argwhere(old_nids[ntype] == x)[0][0], old_metapath_dict[meta][:, i]))
        #                 )
        #
        #     return old_metapath_dict

        seed_nodes = {self.category: idx}
        if self.n_layers < 1:
            raise ValueError("Wrong value of number of layers.")
        for _ in range(self.n_layers):
            mini_mp_inst = mini_mp_instance_sampler(seed_nodes=seed_nodes, mp_instances=self.mp_inst)
            seed_nodes = {}
            for metapath in mini_mp_inst.keys():
                _mini_mp_inst = mini_mp_inst[metapath]
                for i in range(0, len(metapath)):
                    if not metapath[i] in seed_nodes.keys():
                        seed_nodes[metapath[i]] = _mini_mp_inst[:, i]
                    else:
                        seed_nodes[metapath[i]] = np.concatenate((seed_nodes[metapath[i]], _mini_mp_inst[:, i]),
                                                                 axis=0)
            for ntype in seed_nodes.keys():
                seed_nodes[ntype] = np.unique(seed_nodes[ntype])

        # mini_mp_inst = convert_mp_nids(mini_mp_inst, seed_nodes)

        # Here seed_nodes are the nodes sampled from original graph with {self.category: idx} as seed_nodes
        # while mini_mp_inst is corresponding mini metapath instances
        return seed_nodes, mini_mp_inst, self.g

    def __len__(self):
        return len(self.mp_inst)

def collate_fn(batch):
    '''

    Parameters
    ----------
    batch : tuple
        The mini batch data, including seed_nodes, mini_mp_inst and g.
        The seed_nodes is the nids sampled by the MAGNN_sampler from the original graph.
        The mini_mp_inst is the metapath instances corresponding to the sampled graph.
        The g is the sampled graph.

    Returns
    -------
    tuple
        the sampled subgraph and corresponding mini metapath instances
    '''

    def convert_mp_nids(old_metapath_dict, old_nids):
        # convert the old_nids in metapath instances into the new_nids in the subgraph
        for meta in old_metapath_dict.keys():
            for i, ntype in enumerate(meta):
                old_metapath_dict[meta][:, i] = \
                    np.array(
                        list(map(lambda x: np.argwhere(old_nids[ntype] == x)[0][0], old_metapath_dict[meta][:, i]))
                    )

        return old_metapath_dict

    nids = {}
    mini_mp_inst = {}

    ntypes = []
    meta_types = []
    [ntypes.extend(list(_batch[0].keys())) for _batch in batch]
    [meta_types.extend(list(_batch[1].keys())) for _batch in batch]
    ntypes, meta_types = set(ntypes), set(meta_types)

    for _batch in batch:
        # mini_mp_inst = convert_mp_nids(_batch[1], _batch[0])
        for ntype in ntypes:
            if ntype not in _batch[0].keys():
                continue
            if ntype in nids.keys():
                nids[ntype] = np.concatenate((nids[ntype], _batch[0][ntype]), axis=0)
            else:
                nids[ntype] = _batch[0][ntype]
            nids[ntype] = np.unique(nids[ntype], axis=0)
        for meta_type in meta_types:
            if meta_type not in _batch[1].keys():
                continue
            if meta_type in mini_mp_inst.keys():
                mini_mp_inst[meta_type] = np.concatenate((mini_mp_inst[meta_type], _batch[1][meta_type]),
                                                         axis=0)
            else:
                mini_mp_inst[meta_type] = _batch[1][meta_type]
            mini_mp_inst[meta_type] = np.unique(mini_mp_inst[meta_type], axis=0)

    for ntype in nids.keys():
        nids[ntype] = np.sort(np.unique(nids[ntype], axis=0), axis=0)
    for meta_type in mini_mp_inst.keys():
        mini_mp_inst[meta_type] = np.unique(mini_mp_inst[meta_type], axis=0)

    mini_mp_inst = convert_mp_nids(mini_mp_inst, nids)
    _subgraph = dgl.node_subgraph(batch[0][2], nids)
    # feat_dict = {}
    # for ntype in nids.keys():
    #     feat_dict[ntype] = batch[0][2].nodes[ntype].data['feat'][nids[ntype]]

    # return tuple((feat_dict, mini_mp_inst))
    return _subgraph, mini_mp_inst
