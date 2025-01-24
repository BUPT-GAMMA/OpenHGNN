import multiprocessing
from functools import partial, reduce
import time
import numpy as np
import torch
import dgl


class NeighborSampler(object):
    def __init__(self, g, num_fanouts):
        self.g = g
        self.num_fanouts = num_fanouts

    def sample(self, pairs):
        heads, tails, types = zip(*pairs)
        seeds, head_invmap = torch.unique(torch.LongTensor(heads), return_inverse=True)
        blocks = []
        for fanout in reversed(self.num_fanouts):
            sampled_graph = dgl.sampling.sample_neighbors(self.g, seeds, fanout)
            sampled_block = dgl.to_block(sampled_graph, seeds)
            seeds = sampled_block.srcdata[dgl.NID]
            blocks.insert(0, sampled_block)
        return (
            blocks,
            torch.LongTensor(head_invmap),
            torch.LongTensor(tails),
            torch.LongTensor(types),
        )


def generate_pairs(all_walks, window_size, num_workers):
    # for each node, choose the first neighbor and second neighbor of it to form pairs
    # Get all worker processes
    start_time = time.time()
    print("We are generating pairs with {} cores.".format(num_workers))

    # Start all worker processes
    pool = multiprocessing.Pool(processes=num_workers)
    pairs = []
    skip_window = window_size // 2
    for layer_id, walks in enumerate(all_walks):
        block_num = len(walks) // num_workers
        if block_num > 0:
            walks_list = [
                walks[i * block_num: min((i + 1) * block_num, len(walks))]
                for i in range(num_workers)
            ]
        else:
            walks_list = [walks]
        tmp_result = pool.map(
            partial(
                generate_pairs_parallel, skip_window=skip_window, layer_id=layer_id
            ),
            walks_list,
        )
        pairs += reduce(lambda x, y: x + y, tmp_result)

    pool.close()
    end_time = time.time()
    print("Generate pairs end, use {}s.".format(end_time - start_time))
    return np.array([list(pair) for pair in set(pairs)])


def generate_pairs_parallel(walks, skip_window=None, layer_id=None):
    pairs = []
    for walk in walks:
        walk = walk.tolist()
        for i in range(len(walk)):
            for j in range(1, skip_window + 1):
                if i - j >= 0:
                    pairs.append((walk[i], walk[i - j], layer_id))
                if i + j < len(walk):
                    pairs.append((walk[i], walk[i + j], layer_id))
    return pairs
