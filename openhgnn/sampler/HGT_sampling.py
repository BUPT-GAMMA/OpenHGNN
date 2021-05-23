import dgl
import dgl.function as fn
import numba
import numpy as np
import torch as th
import scipy.sparse as ssp
import array
import torch



class Budget(object):
    def __init__(self, hg, n_types, NS):
        self.n_types = {}
        for key, value in n_types.items():
            self.n_types[key] = th.zeros(value)
        self.NS = NS
        self.hg = hg
    def update(self, dst_type, idxs):
        for etype in self.hg.canonical_etypes:
            if dst_type == etype[2]:
                src_type = etype[0]
                #degree = self.hg.in_degrees(idx, etype=etype)
                for i in idxs:
                    src_idx = self.hg.predecessors(i, etype=etype)
                    #if src_idx.shape[0] > 0:
                    len = src_idx.shape[0]
                    if src_type in self.NS.keys():
                        src_idx = th.tensor([i for i in src_idx if i not in self.NS[src_type]])
                    if src_idx.shape[0] > 0:
                        self.n_types[src_type][src_idx] += 1 / len

    def pop(self, type, idx):
        self.n_types[type][idx] = 0


class HGTsampler(object):
    def __init__(self, hg, category, num_nodes_per_type, num_steps):
        self.n_types = {}
        for n in hg.ntypes:
            self.n_types[n] = hg.num_nodes(n)
        self.category = category
        self.num_nodes_per_type = num_nodes_per_type
        self.num_steps = num_steps
        self.hg = hg

    def sampler_subgraph(self, seed_nodes):
        OS = {self.category: th.stack(seed_nodes)}
        NS = OS
        B = Budget(self.hg, self.n_types, NS)
        for type, idxs in OS.items():
            B.update(type, idxs)
        for i in range(self.num_steps):
            prob = {}
            for src_type, p in B.n_types.items():
                #print(src_type)
                if p.max() > 0:
                    prob[src_type] = p / th.sum(p)
                    sampled_idx = th.multinomial(prob[src_type], self.num_nodes_per_type, replacement=False)
                    if not OS.__contains__(src_type):
                        OS[src_type] = sampled_idx
                    else:
                        OS[src_type] = th.cat((OS[src_type], sampled_idx))
                    B.update(src_type, sampled_idx)
                    B.pop(src_type, sampled_idx)
        sg = self.hg.subgraph(OS)
        return sg, OS




def HGT_preprocess4mag(hg, train_idx):
    hg = hg.to('cpu')
    edges = {etype: hg.edges(etype=etype) for etype in hg.canonical_etypes}
    edges.update({(v, e + '_inv', u): (dst, src) for (u, e, v), (src, dst) in edges.items()})
    hg2 = dgl.heterograph(edges)
    hg2 = dgl.to_simple(hg2)

    # Initialize year
    hg2.nodes['paper'].data['timestamp'] = hg.nodes['paper'].data['year'].squeeze()
    for ntype in hg.ntypes:
        if ntype != 'paper':
            hg2.nodes[ntype].data['timestamp'] = th.zeros(hg2.num_nodes(ntype), dtype=th.int64)

    # Aggregate bag-of-paper features
    hg2.nodes['paper'].data['feat'] = hg.nodes['paper'].data['feat']
    hg2.update_all(fn.copy_u('feat', 'm'), fn.mean('m', 'feat'), etype='has_topic')  # field_of_study
    hg2.update_all(fn.copy_u('feat', 'm'), fn.mean('m', 'feat'), etype='writes_inv')  # author
    hg2.update_all(fn.copy_u('feat', 'm'), fn.mean('m', 'feat'), etype='affiliated_with')  # institution

    # Attach log-degree to feature of each node type
    for ntype in hg2.ntypes:
        hg2.nodes[ntype].data['deg'] = th.zeros(hg2.num_nodes(ntype))
    for utype, etype, vtype in hg2.canonical_etypes:
        hg2.nodes[vtype].data['deg'] += hg2.in_degrees(etype=etype)
    for ntype in hg2.ntypes:
        hg2.nodes[ntype].data['feat'] = th.cat([
            hg2.nodes[ntype].data['feat'],
            th.log10(hg2.nodes[ntype].data['deg'][:, None])], 1)
        del hg2.nodes[ntype].data['deg']

    for ntype in hg2.ntypes:
        hg2.nodes[ntype].data['train_mask'] = torch.zeros(hg2.num_nodes(ntype), dtype=torch.bool)
        if ntype == 'paper':
            hg2.nodes[ntype].data['train_mask'][train_idx['paper']] = True

    # Convert to homogeneous graph and add self-loop
    g = dgl.to_homogeneous(hg2, ndata=['timestamp', 'feat'])
    g.edata['etype'] = g.edata[dgl.ETYPE]
    g.ndata['ntype'] = g.ndata[dgl.NTYPE]
    g.ndata['nid'] = g.ndata[dgl.NID]
    del g.edata[dgl.ETYPE]
    del g.edata[dgl.EID]
    del g.ndata[dgl.NTYPE]
    del g.ndata[dgl.NID]
    num_nodes = g.num_nodes()
    g = dgl.add_self_loop(g)
    g.edata['etype'][-num_nodes:] = len(hg2.etypes)

    return g