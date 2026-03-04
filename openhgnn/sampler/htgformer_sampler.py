"""
HTGformer Sampler
==================
Paper: HTGformer: Heterogeneous Temporal Graph Transformer (SIGIR 2025)

采样策略说明：
  HTGformer 的核心优势之一是避免了现有方法中每个时间步都需要
  独立进行邻居采样的问题。HTGformer 的采样只需：
    1. 时间窗口采样 (Temporal Window Sampling)：
       选取最近 T 个时间切片的子图
    2. 针对目标节点做 node-id 采样（全图或 mini-batch）

  注：对于小图（如 Aminer），可以全图训练无需采样。
  对于大图（如 OGBN-MAG），需要使用 mini-batch 采样。
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
    针对目标节点的 mini-batch 采样器。

    对于每个 mini-batch：
      - 采样一批目标节点 ID
      - 返回完整的 T 个时间切片图（不对图做子采样，图较小）
      - 或对图做 k-hop 邻居采样（图较大时）

    Args:
        graphs       (list): List[DGLHeteroGraph], T 个时间切片
        feat_dicts   (list): List[{ntype: Tensor}]
        target_ntype (str):  目标节点类型
        batch_size   (int):  每批目标节点数
        shuffle      (bool): 是否打乱
        use_khop     (bool): 是否对每个时间切片做 k-hop 邻居子采样
        num_neighbors(list): k-hop 采样时每跳的邻居数，如 [10, 5]
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

        # 目标节点总数（以最后一个时间步为准）
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
        对给定的目标节点 IDs，构建 mini-batch。

        Returns:
            batch_graphs   (list): T 个时间切片子图
            batch_feats    (list): T 个对应的节点特征字典
            batch_target_ids (Tensor): 目标节点在子图中的 ID
        """
        if not self.use_khop:
            # 不做 k-hop 采样：使用完整图（小图场景）
            return self.graphs, self.feat_dicts, \
                   {self.target_ntype: target_ids}
        else:
            # k-hop 邻居采样（大图场景）
            batch_graphs = []
            batch_feats = []

            for t, (g, feat) in enumerate(zip(self.graphs, self.feat_dicts)):
                # 使用 DGL 的 k-hop 采样
                sub_g, induced_nodes = self._khop_sample(
                    g, target_ids, self.num_neighbors
                )
                # 提取对应特征
                sub_feat = {}
                for ntype in feat.keys():
                    if ntype in induced_nodes:
                        sub_feat[ntype] = feat[ntype][induced_nodes[ntype]]
                batch_graphs.append(sub_g)
                batch_feats.append(sub_feat)

            # 在子图中目标节点的 ID（与 target_ids 对应）
            new_target_ids = torch.arange(len(target_ids))
            return batch_graphs, batch_feats, \
                   {self.target_ntype: new_target_ids}

    def _khop_sample(self, g, seed_nodes, fanouts):
        """
        简单 k-hop 采样，返回子图和引入的节点集合。
        实际部署中可使用 dgl.sampling.sample_neighbors。
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
            # 合并
            for k, v in frontier_nodes.items():
                if k not in new_frontier:
                    new_frontier[k] = v
                else:
                    new_frontier[k] = torch.cat(
                        [new_frontier[k], v]).unique()
            frontier_nodes = new_frontier

        # 构建子图
        sub_g = dgl.node_subgraph(g, frontier_nodes)
        return sub_g, frontier_nodes


# ──────────────────────────────────────────────────────────────────────────────
# Temporal Window Sampler
# 从整个时序中采样连续 T 个时间切片（用于超长时序）
# ──────────────────────────────────────────────────────────────────────────────
class TemporalWindowSampler:
    """
    时间窗口采样器。
    当时间序列很长时（如 T=304 天），只使用最近 window_size 个切片。

    Args:
        total_timestamps  (int): 总时间步数
        window_size       (int): 模型使用的时间窗口大小
        step              (int): 窗口滑动步长
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
        提取指定窗口的数据。

        Args:
            graphs:     完整图列表
            feat_dicts: 完整特征列表
            start_idx:  窗口起始位置

        Returns:
            window_graphs (list): 切片后的图列表
            window_feats  (list): 切片后的特征列表
        """
        end_idx = start_idx + self.window_size
        return graphs[start_idx:end_idx], feat_dicts[start_idx:end_idx]


# ──────────────────────────────────────────────────────────────────────────────
# HTGformer DataLoader Wrapper
# ──────────────────────────────────────────────────────────────────────────────
class HTGformerDataLoader:
    """
    封装 HTGformer 的训练数据加载流程。

    支持两种模式：
      1. 全图训练 (full_graph=True)：
         每次迭代返回完整的 T 个时间切片图（小图适用）
      2. Mini-batch 训练 (full_graph=False)：
         使用 HTGformerNodeSampler 进行节点级 mini-batch

    Args:
        dataset:         HTGDatasetBase 子类实例
        split:           'train' / 'val' / 'test'
        batch_size:      mini-batch 大小（full_graph=False 时有效）
        full_graph:      是否全图训练
        num_workers:     DataLoader 工作线程数
        shuffle:         训练集是否打乱
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
            # 全图模式：一次性返回所有数据
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
            # Mini-batch 模式
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
