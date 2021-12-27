import dgl
import numpy as np
from torch.utils.data import Dataset
import torch


class RandomWalkSampler(Dataset):
    def __init__(self, g, rw_walks: int, window_size: int, neg_size: int, metapath=None, rw_prob=None, rw_length=None):

        self.g = g
        self.start_idx = [0]
        for i, num_nodes in enumerate([g.num_nodes(ntype) for ntype in g.ntypes]):
            if i < len(g.ntypes) - 1:
                self.start_idx.append(num_nodes + self.start_idx[i])
        self.node_frequency = [0] * g.num_nodes()
        self.rw_prob = rw_prob
        self.rw_length = rw_length
        self.rw_walks = rw_walks
        self.window_size = window_size

        self.neg_pos = 0
        self.neg_size = neg_size

        if metapath is not None:
            start_ntype = self.g.to_canonical_etype(metapath[0])[0]
            self.start_nodes = list(range(self.g.num_nodes(start_ntype))) * self.rw_walks
        else:
            self.start_nodes = list(range(self.g.num_nodes())) * self.rw_walks
        self.metapath = metapath

        self.traces = []
        self.negatives = []
        self.discards = []
        self.trace_ntype = []
        self.traces_idx_of_all_types = []

        self.__generate_metapath()
        self.__generate_node_frequency()
        self.__generate_negatives()
        self.__generate_discards()

    def __generate_metapath(self):
        self.traces, self.trace_ntype = dgl.sampling.random_walk(g=self.g, nodes=self.start_nodes,
                                                                 metapath=self.metapath, prob=self.rw_prob,
                                                                 length=self.rw_length)

        self.traces_idx_of_all_types = torch.index_select(torch.tensor(self.start_idx), 0, self.trace_ntype).repeat(
            len(self.traces), 1) + self.traces

    def __generate_node_frequency(self):
        for trace in self.traces_idx_of_all_types:
            for idx in trace:
                self.node_frequency[idx] += 1

    def __generate_negatives(self):
        pow_frequency = np.array(self.node_frequency) ** 0.75
        nodes_pow = sum(pow_frequency)
        ratio = pow_frequency / nodes_pow
        count = np.round(ratio * 1e8)
        for i, c in enumerate(count):
            self.negatives += [i] * int(c)
        self.negatives = np.array(self.negatives)
        np.random.shuffle(self.negatives)
        self.sampling_prob = ratio

    def __generate_discards(self):
        t = 0.0001
        f = np.array(self.node_frequency) / sum(self.node_frequency)
        self.discards = np.sqrt(t / f) + (t / f)
        return

    def get_center_context_negatives(self, idx):
        # return all (center, context, n negatives) on one trace
        trace = self.traces_idx_of_all_types[idx]
        pair_catch = []
        for i, u in enumerate(trace):
            if np.random.rand() > self.discards[u]:
                continue
            for j, v in enumerate(
                    trace[max(i - self.window_size, 0):i + self.window_size]):
                if u < 0 or v < 0:
                    continue
                if u >= self.g.num_nodes() or v >= self.g.num_nodes():
                    continue

                pair_catch.append((int(u), int(v), self.__get_negatives()))
        return pair_catch

    def __get_negatives(self):
        response = self.negatives[self.neg_pos:self.neg_pos + self.neg_size]
        self.neg_pos = (self.neg_pos + self.neg_size) % len(self.negatives)
        if len(response) != self.neg_size:
            return np.concatenate((response, self.negatives[0:self.neg_pos]))
        return response

    def __getitem__(self, idx):
        return self.get_center_context_negatives(idx)

    def __len__(self):
        return len(self.traces_idx_of_all_types)

    @staticmethod
    def collate(batches):
        all_u = [u for batch in batches for u, _, _ in batch if len(batch) > 0]
        all_v = [v for batch in batches for _, v, _ in batch if len(batch) > 0]
        all_neg_v = [neg_v for batch in batches for _, _, neg_v in batch if len(batch) > 0]

        return torch.LongTensor(all_u), torch.LongTensor(all_v), torch.LongTensor(all_neg_v)
