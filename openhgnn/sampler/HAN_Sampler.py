import dgl
import torch
from dgl.sampling import RandomWalkNeighborSampler
from ..utils import extract_metapaths


class HANSampler(dgl.dataloading.Sampler):
    r"""HANSampler.
    Sample blocks by meta paths.
    """

    def __init__(self, g, category, meta_paths_dict, num_neighbors):
        self.sampler_dict = {}
        self.category = category
        if meta_paths_dict is None:
            self.meta_paths_dict = extract_metapaths(g, category)
        else:
            self.meta_paths_dict = meta_paths_dict
        for meta_path_name, meta_path in self.meta_paths_dict.items():
            # note: random walk may get same route(same edge), which will be removed in the sampled graph.
            # So the sampled graph's edges may be less than num_random_walks(num_neighbors).
            self.sampler_dict[meta_path_name] = RandomWalkNeighborSampler(G=g,
                                                                          num_traversals=1,
                                                                          termination_prob=0,
                                                                          num_random_walks=num_neighbors,
                                                                          num_neighbors=num_neighbors,
                                                                          metapath=meta_path)

    def sample(self, g, seeds):
        input_nodes_dict = {}
        block_dict = {}
        category_seeds = seeds[self.category]
        for meta_path_name, sampler in self.sampler_dict.items():
            frontier = sampler(category_seeds)
            frontier = dgl.remove_self_loop(frontier)
            frontier.add_edges(torch.tensor(category_seeds), torch.tensor(category_seeds))
            block = dgl.to_block(frontier, category_seeds)
            block_dict[meta_path_name] = block
            input_nodes_dict[meta_path_name] = {self.category: block.srcdata[dgl.NID]}
        return input_nodes_dict, seeds, block_dict
