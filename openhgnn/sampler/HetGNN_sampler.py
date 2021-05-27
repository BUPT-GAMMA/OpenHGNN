import os
import dgl
import torch as th
import dgl.sampling
from torch.utils.data import IterableDataset
from dgl.sampling.neighbor import select_topk
from dgl.data.utils import load_graphs, save_graphs


class SkipGramBatchSampler(IterableDataset):
    def __init__(self, hg, batch_size, window_size, rw_len=None):
        self.hg = hg
        self.g = dgl.to_homogeneous(hg).to('cpu')
        self.NID = self.g.ndata[dgl.NID]
        self.NTYPE = self.g.ndata[dgl.NTYPE]
        num_nodes = {}
        for i in range(th.max(self.NTYPE) + 1):
            num_nodes[self.hg.ntypes[i]] = int((self.NTYPE == i).sum())
        self.num_nodes = num_nodes
        self.num_ntypes = len(self.num_nodes)
        # self.weights = {
        #     etype: hg.in_degrees(etype=etype).float() ** 0.75
        #     for _, etype, _ in hg.canonical_etypes
        # }
        self.batch_size = batch_size
        self.window_size = window_size
        if rw_len is not None:
            self.rw_len = rw_len
        else:
            self.rw_len = self.window_size * 2
        self.neg_hetero = True
        self.edge_dict = {}
        self.ntypes = hg.ntypes

    def __iter__(self):
        '''         u = []
                    v = []
                    for i in range(self.window_size):
                        a = traces[:self.window_size*2-i]
                        b = traces[i:]
                        u.append(a)
                        v.append(b)'''

        # random select heads
        # select tails through random walk skgram
        while True:
            heads = th.randint(0, self.g.number_of_nodes(), (self.batch_size,))
            traces, _ = dgl.sampling.random_walk(self.g, heads, length=self.rw_len)
            heads, tails = self.traces2pos(traces, self.window_size)

            heads = (self.NID[heads], self.NTYPE[heads])
            tails = (self.NID[tails], self.NTYPE[tails])
            import copy
            neg_tails = copy.deepcopy(tails)
            for i in range(self.num_ntypes):
                mask = (neg_tails[1] == i)
                ntype = self.hg.ntypes[i]
                neg_tails[0][mask] = th.randint(0, self.hg.number_of_nodes(ntype), size=neg_tails[0][mask].shape)
            yield heads, tails, neg_tails


    def pre_process(self):
        heads = th.arange(self.g.number_of_nodes())
        traces, _ = dgl.sampling.random_walk(self.g, heads, length=self.window_size)
        heads, tails = self.traces2pos(traces, self.window_size)

        heads = (self.NID[heads], self.NTYPE[heads])
        tails = (self.NID[tails], self.NTYPE[tails])
        for i in range(self.num_ntypes):
            for j in range(self.num_ntypes):
                mask_h = (heads[1] == i)
                mask_t = (tails[1] == j)
                edge = (self.ntypes[i], self.ntypes[i] + '-' + self.ntypes[j], self.ntypes[j])
                self.edge_dict[edge] = (heads[0][mask_h], tails[0][mask_t])


    def traces2pos(self, traces, window_size):
        '''
        sample positive edges through skip gram
        '''
        idx_list_u = []
        idx_list_v = []
        batch_size = traces.shape[0]
        for b in range(batch_size):
            walk = traces[b]
            if -1 in walk:
                #walk = traces[b]
                mask = (walk != -1)
                walk = walk[mask]
            for i in range(len(walk)):
                for j in range(i - window_size, i):
                    if j >= 0:
                        idx_list_u.append(walk[j])
                        idx_list_v.append(walk[i])
                for j in range(i + 1, i + 1 + window_size):
                    if j < len(walk):
                        idx_list_u.append(walk[j])
                        idx_list_v.append(walk[i])
                    else:
                        break

        # [num_pos * batch_size]
        u = th.LongTensor(idx_list_u)
        v = th.LongTensor(idx_list_v)

        return v, u


class NeighborSampler(object):
    def __init__(self, hg, ntypes, num_nodes, device):
        # the new graph
        self.hg = hg
        self.ntypes = ntypes
        self.num_nodes = num_nodes
        self.sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
        self.device = device


    def build_hetero_graph(self, heads, tails):
        edge_dict = {}
        num_ntypes = len(self.ntypes)
        for i in range(num_ntypes):
            for j in range(num_ntypes):
                edge = (self.ntypes[i], self.ntypes[i]+ '-' + self.ntypes[j], self.ntypes[j])
                mask = (heads[1] == i) & (tails[1] == j)
                edge_dict[edge] = (heads[0][mask], tails[0][mask])
        hg = dgl.heterograph(edge_dict, self.num_nodes)
        return hg

    def sample_from_item_pairs(self, heads, tails, neg_tails):
        # Create a graph with positive connections only and another graph with negative
        # connections only.
        pos_graph = self.build_hetero_graph(heads, tails)
        neg_graph = self.build_hetero_graph(heads, neg_tails)

        pos_graph, neg_graph = dgl.compact_graphs([pos_graph, neg_graph])
        pos_nodes = pos_graph.ndata[dgl.NID]
        seed_nodes = pos_nodes # same with neg_nodes from neg_graph

        blocks = self.sampler.sample_blocks(
            self.hg, seed_nodes, exclude_eids=None)
        return pos_graph, neg_graph, blocks


def assign_simple_node_features(ndata, g, ntypes, assign_id=False):
    """
    Copies data to the given block from the corresponding nodes in the original graph.
    """
    for ntype in ntypes:
        for col in g.nodes[ntype].data.keys():
            if not assign_id and col == dgl.NID:
                continue
            induced_nodes = ndata[dgl.NID][ntype]
            ndata[col] = {ntype: g.nodes[ntype].data[col][induced_nodes]}


def assign_features_to_blocks(blocks, g, ntypes):
    # For the first block (which is closest to the input), copy the features from
    # the original graph as well as the texts.
    # for ntype in ntypes:
    #     for col in g.nodes[ntype].data.keys():
    #         if not assign_id and col == dgl.NID:
    #             continue
    #         induced_nodes = blocks[0].srcnodes[ntype].data[dgl.NID]
    #         blocks[0].srcnodes[ntype].data[col] = g.nodes[ntype].data[col][induced_nodes]
    assign_simple_node_features(blocks[0].srcdata, g, ntypes)
    #assign_simple_node_features(blocks[-1].dstdata, g, ntypes)


class HetGNNCollator(object):
    def __init__(self, sampler, g):
        self.sampler = sampler
        self.g = g

    def collate_train(self, batches):
        heads, tails, neg_tails = batches[0]
        # Construct multilayer neighborhood via PinSAGE...
        pos_graph, neg_graph, blocks = self.sampler.sample_from_item_pairs(heads, tails, neg_tails)
        assign_features_to_blocks(blocks, self.g, self.g.ntypes)

        return pos_graph, neg_graph, blocks

    # def collate_test(self, samples):
    #     batch = th.LongTensor(samples)
    #     blocks = self.sampler.sample_blocks(batch)
    #     assign_features_to_blocks(blocks, self.g, self.g.ntypes)
    #     return blocks
class hetgnn_graph():
    """

    """
    def __init__(self, hg, n_dataset):
        self.hg = hg
        self.g = dgl.to_homogeneous(hg).to('cpu')
        self.NID = self.g.ndata[dgl.NID]
        self.NTYPE = self.g.ndata[dgl.NTYPE]
        num_nodes = {}
        for i in range(th.max(self.NTYPE) + 1):
            num_nodes[self.hg.ntypes[i]] = int((self.NTYPE == i).sum())
        self.num_nodes = num_nodes
        self.weight_column = 'w'
        self.n_dataset = n_dataset

    def get_hetgnn_graph(self, length, walks, restart_prob):
        fname = './openhgnn/output/HetGNN/{}_{}_{}.bin'.format(
            self.n_dataset, length, walks)
        if os.path.exists(fname):
            g, _ = load_graphs(fname)
            return g[0]
        else:
            g = self.build_hetgnn_graph(length, walks, restart_prob)
            save_graphs(fname, g)
            return g

    def build_hetgnn_graph(self, length, walks, restart_prob):
        #edges = [[[[],[]]] * len(self.num_nodes)] * len(self.num_nodes)
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
                edges[int(utype)][int(vtype)][0].append(self.NID[uid])
                edges[int(utype)][int(vtype)][1].append(self.NID[vid])

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

        neighbor_graph = dgl.heterograph(
            edge_dict,
            self.num_nodes
        )

        neighbor_graph = dgl.to_simple(neighbor_graph, return_counts=self.weight_column)
        counts = neighbor_graph.edata[self.weight_column]
        neighbor_graph = select_topk(neighbor_graph, k, self.weight_column)

        return neighbor_graph