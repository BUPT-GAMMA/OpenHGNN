import torch

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
