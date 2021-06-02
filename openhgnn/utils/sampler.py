import os
import dgl
import pickle
import random
import torch as th
import numpy as np
from scipy.sparse import coo_matrix
from dgl.dataloading.negative_sampler import _BaseNegativeSampler
from dgl import backend as F
from dgl.data.utils import load_graphs, save_graphs
import dgl.sampling
from collections import Counter
from torch.utils.data import IterableDataset, DataLoader


class MP2vecCollator(object):
    def __init__(self, ntypes, num_nodes):
        # the new graph
        self.ntypes = ntypes
        self.num_nodes = num_nodes

    def build_hetero_graph(self, heads, tails):
        edge_dict = {}
        num_ntypes = len(self.ntypes)
        for i in range(num_ntypes):
            for j in range(num_ntypes):
                edge = (self.ntypes[i], self.ntypes[i] + '-' + self.ntypes[j], self.ntypes[j])
                mask = (heads[1] == i) & (tails[1] == j)
                edge_dict[edge] = (heads[0][mask], tails[0][mask])
        hg = dgl.heterograph(edge_dict, self.num_nodes)
        return hg

    def construct_from_item_pairs(self, heads, tails, neg_tails):
        pos_graph = self.build_hetero_graph(heads, tails)
        neg_graph = self.build_hetero_graph(heads, neg_tails)

        pos_graph, neg_graph = dgl.compact_graphs([pos_graph, neg_graph])
        # pos_nodes = pos_graph.ndata[dgl.NID]
        # seed_nodes = pos_nodes  # same with neg_nodes from neg_graph

        return pos_graph, neg_graph, [pos_graph]

    def collate_train(self, batches):
        heads, tails, neg_tails = batches[0]
        pos_graph, neg_graph, block = self.construct_from_item_pairs(heads, tails, neg_tails)
        return pos_graph, neg_graph, block



class randomwalk_on_heterograph(object):
    def __init__(self, hg):
        self.hg = hg
        self.g = dgl.to_homogeneous(hg).to('cpu')
        self.NID = self.g.ndata[dgl.NID]
        self.NTYPE = self.g.ndata[dgl.NTYPE]
        num_nodes = {}
        for i in range(th.max(self.NTYPE) + 1):
            num_nodes[self.hg.ntypes[i]] = int((self.NTYPE == i).sum())
        self.num_nodes = num_nodes
        self.weight_column = 'w'

    def randomwalk(self, length, walks, restart_prob):
        from dgl.sampling.neighbor import select_topk
        from dgl.sampling.pinsage import RandomWalkNeighborSampler
        #path, _ = dgl.sampling.random_walk(self.g, nodes, length=length)
        #traces, types = dgl.sampling.random_walk(self.g, nodes, length=length)
        edges = [[[[],[]], [[],[]], [[],[]]],
                 [[[],[]], [[],[]], [[],[]]],
                 [[[],[]], [[],[]], [[],[]]]]
        for i in range(self.g.number_of_nodes()):
            nodes = th.tensor([i]).repeat(walks)
            traces, types = dgl.sampling.random_walk(self.g, nodes, length=length, restart_prob=restart_prob)
            concat_vids, _, _, _ = dgl.sampling.pack_traces(traces, types)
            concat_types = th.index_select(self.NTYPE, 0, concat_vids)
            uid = concat_vids[0]
            utype = concat_types[0]
            for (vid, vtype) in zip(concat_vids, concat_types):
                # if vid == uid:
                #     pass
                # else:
                edges[int(utype)][int(vtype)][0].append(self.NID[uid])
                edges[int(utype)][int(vtype)][1].append(self.NID[vid])

        from dgl import convert
        from dgl import transform
        edge_dict = {}
        k = {}
        num_ntypes = self.NTYPE.max() + 1
        for i in range(num_ntypes):
            for j in range(num_ntypes):
                edge = (self.hg.ntypes[j], self.hg.ntypes[j]+'-'+self.hg.ntypes[i], self.hg.ntypes[i])
                edge_dict[edge] = (th.tensor(edges[i][j][1]), th.tensor(edges[i][j][0]))
                if j == 2:
                    k[edge] = 3
                else:
                    k[edge] = 10

        neighbor_graph = convert.heterograph(
            edge_dict,
            self.num_nodes
        )

        neighbor_graph = transform.to_simple(neighbor_graph, return_counts=self.weight_column)
        counts = neighbor_graph.edata[self.weight_column]
        neighbor_graph = select_topk(neighbor_graph, k, self.weight_column)
        from dgl.data.utils import load_graphs, save_graphs
        fname = './openhgnn/output/HetGNN/Academic.bin'
        save_graphs(fname, neighbor_graph)
        return

    def randomwalk_with_restart(self, nodes, length, restart_prob=None):

        path = dgl.sampling.random_walk(self.g, nodes, length=length, restart_prob=restart_prob)
        return path

    def full_rwr(self, walks, length, restart_prob):
        n = self.g.number_of_nodes()
        # expand to 'walks' times n
        nodes = th.arange(n).repeat(walks)
        path = self.randomwalk(nodes, length)
        return path

    def path_hetro(self):
        pass

    def neighbour_hetero(self, path):
        edges = {}
        for p in path[0]:
            center_node = p[0]
            neigh_nodes = p[1:]
            c_nid = int(self.NID[center_node])
            c_ntype = int(self.NTYPE[center_node])
            edges.setdefault(c_ntype, {})
            edges[c_ntype].setdefault(c_nid, {})
            for n in neigh_nodes:
                if n >= 0:
                    n_nid = self.NID[n]
                    n_ntype = int(self.NTYPE[n])
                    edges[c_ntype][c_nid].setdefault(n_ntype, [])
                    edges[c_ntype][c_nid][n_ntype].append(n_nid)
                else:
                    break
        import dgl.sampling.select_topk
        a = 1
        # src = F.reshape(paths[:, self.metapath_hops::self.metapath_hops], (-1,))
        # dst = F.repeat(paths[:, 0], self.num_traversals, 0)
        #
        # src_mask = (src != -1)
        # src = F.boolean_mask(src, src_mask)
        # dst = F.boolean_mask(dst, src_mask)
        # import dgl.convert as convert
        # # count the number of visits and pick the K-most frequent neighbors for each node
        # neighbor_graph = convert.heterograph(
        #     {(self.ntype, '_E', self.ntype): (src, dst)},
        #     {self.ntype: self.G.number_of_nodes(self.ntype)}
        # )
        # neighbor_graph = transform.to_simple(neighbor_graph, return_counts=self.weight_column)
        # counts = neighbor_graph.edata[self.weight_column]
        #
        # neighbor_graph = select_topk(neighbor_graph, self.num_neighbors, self.weight_column)
        # selected_counts = F.gather_row(counts, neighbor_graph.edata[EID])
        # neighbor_graph.edata[self.weight_column] = selected_counts



class pro_sampler(_BaseNegativeSampler):

    def __init__(self, k, p):
        self.k = k
        self.p = p

    def _generate(self, g, eids, canonical_etype):
        _, _, vtype = canonical_etype
        shape = F.shape(eids)
        dtype = F.dtype(eids)
        ctx = F.context(eids)
        shape = (shape[0] * self.k,)
        src, _ = g.find_edges(eids, etype=canonical_etype)
        src = F.repeat(src, self.k, 0)
        dst = np.random.choice(np.arange(0, g.number_of_nodes()), shape, replace=True, p=self.p)
        # dst = F.randint(shape, dtype, ctx, 0, g.number_of_nodes(vtype))
        dst = th.tensor(dst, dtype=dtype, device=ctx)
        return src, dst


def gen_neg_edges(g, num_neg, device):
    if not g.is_homogeneous:
        g_homo = dgl.to_homogeneous(g)
    else:
        g_homo = g
    node_degrees = g_homo.out_degrees().to('cpu').numpy()
    node_weights = np.power(node_degrees, 0.75)
    node_probs = node_weights / np.sum(node_weights)

    # neg_sampler = dgl.dataloading.negative_sampler.Uniform(num_neg)
    # neg_edges = neg_sampler(g_homo, th.arange(0, g_homo.num_edges(), dtype=th.int64, device=device))
    neg_sampler = pro_sampler(num_neg, node_probs)
    neg_edges = neg_sampler(g_homo, th.arange(0, g_homo.num_edges(), dtype=th.int64, device=device))
    # tensors used as indices must be long, byte or bool tensors， so it should be tensor.int64
    return neg_edges


def _get_neg_edge(g, epoch_seed, n_dataset, num_neg, device):
    fname = './openhgnn/output/NSHE/{}_NE-rate={:.0f}_seed={}.dat'.format(
        n_dataset, num_neg, epoch_seed)
    if os.path.exists(fname):
        # load the existing negative edge sampling file
        with open(fname, 'rb') as f:
            try:
                epoch_data = pickle.load(f)
                neg_edges = epoch_data['neg_edges']
            except EOFError:
                os.remove(fname)
                print(epoch_seed, fname)
    else:
        # sample
        neg_edges = gen_neg_edges(g, num_neg=num_neg, device=device)
        # save
        data_to_save = {'neg_edges': neg_edges}
        with open(fname, 'wb') as f:
            pickle.dump(data_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)
    return neg_edges

def gen_ns_instances(g, num_ns_neg):

    def _get_current_sample_types(prev_sampled_types, pos_ns):
        u_node_type = []
        v_node_type = []
        etypes = []
        for t in set(prev_sampled_types):
            for etype in g.canonical_etypes:
                if t in etype[0]:
                    if etype[2] not in pos_ns.keys():
                        u_node_type.append(etype[0])
                        v_node_type.append(etype[2])
                        etypes.append(etype[1])
        return u_node_type, etypes, v_node_type

    def _sample_pos_ns(i, target_t):
        pos_ns = {target_t: i}
        prev_sampled_types = [target_t]
        prev_sampled_types, etypes, current_sample_types = _get_current_sample_types(prev_sampled_types, pos_ns)
        prev_t = i
        while len(current_sample_types) > 0:
        #BFS
            for (u_ntype, etype, v_ntype) in zip(prev_sampled_types, etypes, current_sample_types):
                # find neighbors of type t
                prev_nid = pos_ns[u_ntype]
                t_neighbors = g.out_edges(prev_nid, etype=etype)[1]
                # random select one as postive ns_instance
                if len(t_neighbors) == 0:  # if there is no neighbor to select
                    print('Node {} has no {} type point!!'.format(id, v_ntype))
                    return None
                elif len(t_neighbors) == 1:  # if there is only one selection
                    r = 0
                elif len(t_neighbors) > 1:  # if there is only one selection
                    r = random.randint(0, len(t_neighbors) - 1)
                pos_ns[v_ntype] = t_neighbors[r]

            prev_sampled_types = current_sample_types
            prev_sampled_types, etypes, current_sample_types = _get_current_sample_types(prev_sampled_types, pos_ns)
        pos_ns['label'] = 1
        return pos_ns

    ns_ins_list = []
    for target_t in g.ntypes:
        # Sample using target_t
        target_nodes = g.nodes(target_t)
        num_target_node = g.num_nodes(target_t)
        ns_ins = []
        for i in target_nodes:
            # ================== pos ns instnace sampling ==================
            pos_ns = _sample_pos_ns(i, target_t)
            if pos_ns is not None:
                ns_ins.append(pos_ns)
            else:
                continue
            # ================== neg ns instnace sampling ==================
            for _ in range(num_ns_neg):
                neg_ns = pos_ns.copy()
                neg_node = target_nodes[random.randint(0, num_target_node - 1)]
                # replace type in schema instance
                neg_ns[target_t] = neg_node
                neg_ns['label'] = 0  # BCE_loss, negative samples label = 0
                ns_ins.append(neg_ns)
        ns_ins_dict = {}
        ns_ins_dict['target_type'] = target_t
        ns_ins_dict['label'] = th.tensor([x['label'] for x in ns_ins])
        ns_ins_dict[target_t] = th.tensor([x[target_t] for x in ns_ins])
        for ntype in g.ntypes:
            if ntype != target_t:
                ns_ins_dict[ntype] = th.tensor([x[ntype] for x in ns_ins])

        ns_ins_list.append(ns_ins_dict)
    return ns_ins_list


def _get_ns_instance(g, epoch_seed, n_dataset, num_ns_neg):
    fname = './openhgnn/output/NSHE/{}_NS-rate={:.0f}_seed={}.dat'.format(
        n_dataset, num_ns_neg, epoch_seed)
    if os.path.exists(fname):
        # load the existing negative network schemas sampling file
        with open(fname, 'rb') as f:
            try:
                epoch_data = pickle.load(f)
            except EOFError:
                print(epoch_seed, fname)
        ns_ins_list = epoch_data['ns_instances']
    else:

        ns_ins_list = gen_ns_instances(g, num_ns_neg)
        # save
        data_to_save = {
            'ns_instances': ns_ins_list}
        with open(fname, 'wb') as f:
            pickle.dump(data_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)

    return ns_ins_list


def get_epoch_samples(g, epoch, dataset, ns_neg, device):
    """
    Renew ns_instances and neg_edges in every epoch:
    1. get the seed for current epoch
    2. find using seed
        Y: load the file
        N: sample again and save
    """

    epoch_seed = np.random.randint(1000)
    np.random.seed(epoch_seed)
    neg_edges = _get_neg_edge(g, epoch_seed, dataset, 1, device)
    ns_samples = _get_ns_instance(g, epoch_seed, dataset, ns_neg)

    return neg_edges, ns_samples


