"""
HTGformer Sampler
==================
Paper: HTGformer: Heterogeneous Temporal Graph Transformer (SIGIR 2025)

Sampling Strategy Explanation:
One of the core advantages of HTGformer is that it avoids the problem in existing methods where neighbor sampling needs to be performed independently for each time step. The sampling of HTGformer only involves:
1. Temporal Window Sampling:
Select a subgraph of the most recent T time slices
2. Node-id sampling for the target node (for the entire graph or mini-batch)
Note: For small graphs (such as Aminer), the entire graph can be used for training without sampling.
For large graphs (such as OGBN-MAG), mini-batch sampling is required.
"""

import torch
import dgl
import numpy as np
from torch.utils.data import DataLoader, Dataset


# ──────────────────────────────────────────────────────────────────────────────
# Mini-batch Node Sampler for HTGformer
# ──────────────────────────────────────────────────────────────────────────────
class HTGformerNodeSampler:
    """
    A mini-batch sampler for the target node.
    For each mini-batch:
    - Sample a batch of target node IDs
    - Return the complete T time slices of the graph (no sub-sampling of the graph, the graph is small)
    - Or perform k-hop neighbor sampling on the graph (when the graph is large)
    Args:
    graphs (list): List[DGLHeteroGraph], T time slices feat_dicts   (list): List[{ntype: Tensor}]
    target_ntype (str): Type of target node
    batch_size   (int): Number of target nodes per batch
    shuffle      (bool): Whether to shuffle
    use_khop     (bool): Whether to perform k-hop neighbor sub-sampling for each time slice
    num_neighbors(list): Number of neighbors per hop during k-hop sampling, e.g. [10, 5]
    """

    def __init__(self, graphs, feat_dicts, target_ntype,
                 batch_size=256, shuffle=True,
                 use_khop=False, num_neighbors=None):
        self.graphs = graphs
        self.feat_dicts = feat_dicts
        self.target_ntype = target_ntype
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.use_khop = use_khop
        self.num_neighbors = num_neighbors or [10, 5]

        # Total number of target nodes (based on the last time step)
        self.num_targets = graphs[-1].num_nodes(target_ntype)
        self.node_ids = torch.arange(self.num_targets)

    def __iter__(self):
        if self.shuffle:
            perm = torch.randperm(self.num_targets)
        else:
            perm = torch.arange(self.num_targets)

        for start in range(0, self.num_targets, self.batch_size):
            batch_ids = perm[start: start + self.batch_size]
            yield self._sample_batch(batch_ids)

    def __len__(self):
        return (self.num_targets + self.batch_size - 1) // self.batch_size

    def _sample_batch(self, target_ids):
        """
        For the given target node IDs，build mini-batch。

        Returns:
            batch_graphs   (list): T A time slice screenshot
            batch_feats    (list): T The corresponding node feature dictionary
            batch_target_ids (Tensor): The target ID node in the map ID
        """
        if not self.use_khop:
            # Do not perform k-hop sampling: Use the complete graph (small graph scenario)
            return self.graphs, self.feat_dicts, \
                   {self.target_ntype: target_ids}
        else:
            # k-hop Neighbor Sampling (Large Map Scenario)
            batch_graphs = []
            batch_feats = []

            for t, (g, feat) in enumerate(zip(self.graphs, self.feat_dicts)):
                # Using k-hop sampling with DGL
                sub_g, induced_nodes = self._khop_sample(
                    g, target_ids, self.num_neighbors
                )
                # Extract the corresponding features
                sub_feat = {}
                for ntype in feat.keys():
                    if ntype in induced_nodes:
                        sub_feat[ntype] = feat[ntype][induced_nodes[ntype]]
                batch_graphs.append(sub_g)
                batch_feats.append(sub_feat)

            # The ID of the target node in the subgraph (corresponding to target_ids)
            new_target_ids = torch.arange(len(target_ids))
            return batch_graphs, batch_feats, \
                   {self.target_ntype: new_target_ids}

    def _khop_sample(self, g, seed_nodes, fanouts):
        """
        Simple k-hop sampling, returning the subgraph and the set of introduced nodes.
        In actual deployment, dgl.sampling.sample_neighbors can be used.
        """
        frontier_nodes = {self.target_ntype: seed_nodes}

        for fanout in reversed(fanouts):
            new_frontier = {}
            for etype in g.canonical_etypes:
                src_t, rel_t, dst_t = etype
                if dst_t not in frontier_nodes:
                    continue
                dst_ids = frontier_nodes[dst_t]
                sampled = dgl.sampling.sample_neighbors(
                    g, {dst_t: dst_ids}, fanout,
                    edge_dir='in'
                )
                src_ids = sampled.edges(etype=rel_t)[0].unique()
                if src_t not in new_frontier:
                    new_frontier[src_t] = src_ids
                else:
                    new_frontier[src_t] = torch.cat(
                        [new_frontier[src_t], src_ids]).unique()
            # merge
            for k, v in frontier_nodes.items():
                if k not in new_frontier:
                    new_frontier[k] = v
                else:
                    new_frontier[k] = torch.cat(
                        [new_frontier[k], v]).unique()
            frontier_nodes = new_frontier

        # Construct subgraph
        sub_g = dgl.node_subgraph(g, frontier_nodes)
        return sub_g, frontier_nodes


# ──────────────────────────────────────────────────────────────────────────────
# Temporal Window Sampler
# Sample consecutive T time slices from the entire time sequence (for ultra-long time sequences)
# ──────────────────────────────────────────────────────────────────────────────
class TemporalWindowSampler:
    """
    Time window sampler.
    When the time series is very long (e.g., T = 304 days), only the last window_size slices are used.
    Args:
    total_timestamps  (int): Total number of time steps
    window_size       (int): Size of the time window used by the model
    step              (int): Step size for window sliding
    """

    def __init__(self, total_timestamps: int, window_size: int, step: int = 1):
        self.total_T = total_timestamps
        self.window_size = window_size
        self.step = step

        # 所有合法窗口的起始位置
        self.starts = list(range(
            0, total_timestamps - window_size + 1, step
        ))

    def __len__(self):
        return len(self.starts)

    def __iter__(self):
        for start in self.starts:
            yield start, start + self.window_size

    def get_window(self, graphs, feat_dicts, start_idx):
        """
        Extract the data from the specified window

        Args:
            graphs:     Complete list of pictures
            feat_dicts: Complete list of features
            start_idx:  Window starting position

        Returns:
            window_graphs (list): List of sliced images
            window_feats  (list): The list of features after slicing
        """
        end_idx = start_idx + self.window_size
        return graphs[start_idx:end_idx], feat_dicts[start_idx:end_idx]


# ──────────────────────────────────────────────────────────────────────────────
# HTGformer DataLoader Wrapper
# ──────────────────────────────────────────────────────────────────────────────
class HTGformerDataLoader:
    """
    Package the training data loading process of HTGformer.
    Support two modes:
    1. Full graph training (full_graph=True):
    During each iteration, return the complete T time slice graphs (applicable for small graphs)
    2. Mini-batch training (full_graph=False):
    Use HTGformerNodeSampler for node-level mini-batch
    Args:
    dataset:         An instance of the subclass HTGDatasetBase split:           'train' / 'val' / 'test'
    batch_size:      Mini-batch size (applicable when full_graph=False)
    full_graph:      Whether to train the entire graph
    num_workers:     Number of worker threads for DataLoader
    shuffle:         Whether to shuffle the training set
    """

    def __init__(self, dataset, split='train', batch_size=256,
                 full_graph=True, num_workers=0, shuffle=True):
        self.dataset = dataset
        self.split = split
        self.full_graph = full_graph
        self.batch_size = batch_size

        mask_map = {
            'train': dataset.train_mask,
            'val': dataset.val_mask,
            'test': dataset.test_mask,
        }
        self.mask = mask_map[split]
        self.node_ids = self.mask.nonzero(as_tuple=True)[0]

    def __iter__(self):
        if self.full_graph:
            # Full-screen mode: Returns all data at once
            yield {
                'graphs': self.dataset.graphs,
                'feat_dicts': self.dataset.feat_dicts,
                'labels': self.dataset.labels[self.node_ids],
                'target_ids': {
                    self.dataset.category: self.node_ids
                },
                'mask': self.mask,
            }
        else:
            # Mini-batch form
            perm = torch.randperm(len(self.node_ids))
            for start in range(0, len(self.node_ids), self.batch_size):
                batch_node_ids = self.node_ids[
                    perm[start:start + self.batch_size]
                ]
                yield {
                    'graphs': self.dataset.graphs,
                    'feat_dicts': self.dataset.feat_dicts,
                    'labels': self.dataset.labels[batch_node_ids],
                    'target_ids': {
                        self.dataset.category: batch_node_ids
                    },
                    'mask': None,
                }

    def __len__(self):
        if self.full_graph:
            return 1
        return (len(self.node_ids) + self.batch_size - 1) // self.batch_size
