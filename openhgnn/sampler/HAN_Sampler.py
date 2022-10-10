import dgl
from dgl.sampling import RandomWalkNeighborSampler
from ..utils import extract_metapaths


class HANSampler(dgl.dataloading.Sampler):
    """HANSampler.
    Sample blocks by node types and meta paths.

    Parameters
    ----------
    g : DGLHeteroGraph
        The graph.
    ntypes : list[str]
        List of center node types.
    meta_paths_dict: dict[str, list[etype]]
        Dict from meta path name to meta path.
    num_neighbors: int
        Number of neighbors to sample.
    """

    def __init__(self, g, seed_ntypes, meta_paths_dict, num_neighbors):
        self.output_device = None  # as_edge_prediction_sampler requires this attribute

        self.ntype_mp_name_sampler_dict = {}
        self.seed_ntypes = seed_ntypes
        self.ntype_meta_paths_dict = {}

        # build ntype_meta_paths_dict
        for ntype in self.seed_ntypes:
            self.ntype_meta_paths_dict[ntype] = {}
            for meta_path_name, meta_path in meta_paths_dict.items():
                # a meta path starts with this node type
                if meta_path[0][0] == ntype:
                    self.ntype_meta_paths_dict[ntype][meta_path_name] = meta_path
        for ntype, meta_paths_dict in self.ntype_meta_paths_dict.items():
            if len(meta_paths_dict) == 0:
                self.ntype_meta_paths_dict[ntype] = extract_metapaths(ntype, g.canonical_etypes)

        for ntype, meta_paths_dict in self.ntype_meta_paths_dict.items():
            self.ntype_mp_name_sampler_dict[ntype] = {}
            for meta_path_name, meta_path in meta_paths_dict.items():
                # note: random walk may get same route(same edge), which will be removed in the sampled graph.
                # So the sampled graph's edges may be less than num_random_walks(num_neighbors).
                self.ntype_mp_name_sampler_dict[ntype][meta_path_name] = RandomWalkNeighborSampler(G=g,
                                                                                                   num_traversals=1,
                                                                                                   termination_prob=0,
                                                                                                   num_random_walks=num_neighbors,
                                                                                                   num_neighbors=num_neighbors,
                                                                                                   metapath=meta_path)

    def sample(self, g, seeds, exclude_eids=None):  # exclude_eids is for compatibility with link prediction
        """sample method.

        Returns
        -------
        dict[str, dict[str, Tensor]]
            Input node ids. Dict from node type to dict from meta path name to node ids.
        dict[str, Tensor]
            Seeds. Dict from node type to node ids
        dict[str, dict[str, DGLBlock]]
            Sampled blocks. Dict from node type to dict from meta path name to sampled blocks.
        """
        input_nodes_dict = {}
        ntype_mp_name_block_dict = {}
        for ntype, nid in seeds.items():
            if len(nid) == 0:
                continue
            input_nodes_dict[ntype] = {}
            ntype_mp_name_block_dict[ntype] = {}
            for meta_path_name, sampler in self.ntype_mp_name_sampler_dict[ntype].items():
                frontier = sampler(nid)
                frontier = dgl.remove_self_loop(frontier)
                frontier.add_edges(nid.clone().detach(), nid.clone().detach())
                block = dgl.to_block(frontier, nid)
                ntype_mp_name_block_dict[ntype][meta_path_name] = block
                input_nodes_dict[ntype][meta_path_name] = block.srcdata[dgl.NID]
        return input_nodes_dict, seeds, ntype_mp_name_block_dict
