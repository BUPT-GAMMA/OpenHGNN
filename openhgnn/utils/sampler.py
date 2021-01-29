import os
import dgl
import pickle
import random
import torch as th
import numpy as np


def gen_neg_edges(g, num_neg):
    if not g.is_homogeneous:
        g_homo = dgl.to_homogeneous(g)
    neg_sampler = dgl.dataloading.negative_sampler.Uniform(num_neg)
    neg_edges = neg_sampler(g_homo, th.arange(0, g_homo.num_edges(), dtype=th.int64))
    return neg_edges


def _get_neg_edge(g, epoch_seed, n_dataset, num_neg=1):
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
        neg_edges = gen_neg_edges(g, num_neg=num_neg)
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
                    print('Node {} has no {} type point!!'.format(id, t))
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


def get_epoch_samples(g, epoch, dataset, ns_neg):
    """
    Renew ns_instances and neg_edges in every epoch:
    1. get the seed for current epoch
    2. find using seed
        Y: load the file
        N: sample again and save
    """
    epoch_seed = 0
    np.random.seed(epoch_seed)
    neg_edges = _get_neg_edge(g, epoch_seed, dataset, 1)
    ns_samples = _get_ns_instance(g, epoch_seed, dataset, ns_neg)

    return neg_edges, ns_samples