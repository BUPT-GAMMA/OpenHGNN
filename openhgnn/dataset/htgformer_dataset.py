"""
HTGformer Dataset Implementation
==================================
Paper: HTGformer: Heterogeneous Temporal Graph Transformer (SIGIR 2025)

支持的数据集：
  - Aminer (academic heterogeneous temporal graph, 论文使用)
  - COVID-19 (epidemiological HTG, node regression)
  - OGBN-MAG (large-scale academic HTG)
  - 自定义数据集接口 (CustomHTGDataset)

数据格式说明：
  HTG = Heterogeneous Temporal Graph
    - T 个时间切片 (snapshots)
    - 每个切片: DGLHeteroGraph + 各类型节点特征
    - 节点标签（用于分类/回归任务）
"""

import os
import numpy as np
import torch
import dgl
from dgl.data import DGLDataset
from dgl import save_graphs, load_graphs

# OpenHGNN 数据集基类（部署时取消注释）
# from openhgnn.dataset import BaseDataset, register_dataset


# ──────────────────────────────────────────────────────────────────────────────
# 基础类：异质时序图数据集
# ──────────────────────────────────────────────────────────────────────────────
class HTGDatasetBase(DGLDataset):
    """
    HTG（异质时序图）数据集基类。
    子类需实现:
      - _build_snapshots(): 返回 (graph_list, feat_list, labels, masks)
    """

    def __init__(self, name, raw_dir=None, save_dir=None,
                 force_reload=False, verbose=False):
        super().__init__(name=name, raw_dir=raw_dir, save_dir=save_dir,
                         force_reload=force_reload, verbose=verbose)

    def download(self):
        """子类可覆盖实现下载逻辑"""
        pass

    def process(self):
        """调用 _build_snapshots 处理原始数据"""
        result = self._build_snapshots()
        self.graphs = result['graphs']          # List[DGLHeteroGraph]
        self.feat_dicts = result['feat_dicts']  # List[{ntype: Tensor}]
        self.labels = result['labels']          # Tensor[N]
        self.train_mask = result['train_mask']  # Tensor[N] bool
        self.val_mask = result['val_mask']
        self.test_mask = result['test_mask']
        self.meta = result['meta']              # dict with metadata

    def _build_snapshots(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        return self.graphs, self.feat_dicts, self.labels

    def __len__(self):
        return 1  # 整个时序图视为一条数据

    def has_cache(self):
        return os.path.exists(os.path.join(self.save_dir, f'{self.name}.bin'))

    def save(self):
        save_graphs(os.path.join(self.save_dir, f'{self.name}.bin'), self.graphs)

    def load(self):
        self.graphs, _ = load_graphs(
            os.path.join(self.save_dir, f'{self.name}.bin'))

    @property
    def num_classes(self):
        return self.meta.get('num_classes', 2)

    @property
    def node_types(self):
        return self.meta.get('node_types', [])

    @property
    def num_timestamps(self):
        return len(self.graphs)

    @property
    def category(self):
        return self.meta.get('category', self.node_types[0])

    @property
    def in_dim_dict(self):
        return self.meta.get('in_dim_dict', {})


# ──────────────────────────────────────────────────────────────────────────────
# Aminer 数据集（学术异质时序图）
# 节点类型: paper, author, venue
# 边类型: (author, writes, paper), (paper, published_in, venue), etc.
# 任务: paper 节点分类（研究领域预测）
# ──────────────────────────────────────────────────────────────────────────────
class AminerHTGDataset(HTGDatasetBase):
    """
    Aminer 学术异质时序图数据集。

    数据集统计（来自论文）：
      - 节点类型: author, paper, venue
      - 时间切片: 按年份划分（如 2000-2020, T=20）
      - 任务: paper 节点分类（领域分类）

    注：由于 Aminer 数据集需要申请，这里提供合成数据生成器用于调试。
    实际使用时请替换 _load_real_data() 中的加载逻辑。
    """

    _URL = "https://www.aminer.cn/heterogeneous_dataset"

    def __init__(self, raw_dir='./data/aminer', num_timestamps=10,
                 use_synthetic=True):
        self._num_timestamps = num_timestamps
        self.use_synthetic = use_synthetic
        super().__init__(name='aminer_htg', raw_dir=raw_dir)
        if not hasattr(self, 'meta'):
            self.process()

    def _build_snapshots(self):
        if self.use_synthetic:
            return self._generate_synthetic_aminer()
        else:
            return self._load_real_aminer()

    def _generate_synthetic_aminer(self):
        """
        生成与 Aminer 结构相似的合成数据，用于调试和快速验证。
        节点规模: author=500, paper=1000, venue=50
        特征维度: author=128, paper=256, venue=64
        时间切片: T=10
        类别数: 4
        """
        T = self._num_timestamps
        num_authors = 500
        num_papers = 1000
        num_venues = 50
        feat_dim = {'author': 128, 'paper': 256, 'venue': 64}
        num_classes = 4

        graphs = []
        feat_dicts = []

        for t in range(T):
            # 模拟每个时间步的边（逐步增加连边）
            scale = 1.0 + 0.1 * t
            n_writes = int(2000 * scale)
            n_pub = int(800 * scale)
            n_cite = int(1500 * scale)

            writes_src = torch.randint(0, num_authors, (n_writes,))
            writes_dst = torch.randint(0, num_papers, (n_writes,))

            pub_src = torch.randint(0, num_papers, (n_pub,))
            pub_dst = torch.randint(0, num_venues, (n_pub,))

            cite_src = torch.randint(0, num_papers, (n_cite,))
            cite_dst = torch.randint(0, num_papers, (n_cite,))

            g = dgl.heterograph({
                ('author', 'writes', 'paper'): (writes_src, writes_dst),
                ('paper', 'written_by', 'author'): (writes_dst, writes_src),
                ('paper', 'published_in', 'venue'): (pub_src, pub_dst),
                ('venue', 'publishes', 'paper'): (pub_dst, pub_src),
                ('paper', 'cites', 'paper'): (cite_src, cite_dst),
            }, num_nodes_dict={
                'author': num_authors,
                'paper': num_papers,
                'venue': num_venues,
            })
            graphs.append(g)

            # 节点特征：随机初始化（实际中使用 text/attribute features）
            feat_dicts.append({
                'author': torch.randn(num_authors, feat_dim['author']),
                'paper': torch.randn(num_papers, feat_dim['paper']),
                'venue': torch.randn(num_venues, feat_dim['venue']),
            })

        # 标签：paper 节点的领域分类（4类）
        labels = torch.randint(0, num_classes, (num_papers,))

        # 划分 train/val/test = 60/20/20
        idx = torch.randperm(num_papers)
        n_train = int(0.6 * num_papers)
        n_val = int(0.2 * num_papers)

        train_mask = torch.zeros(num_papers, dtype=torch.bool)
        val_mask = torch.zeros(num_papers, dtype=torch.bool)
        test_mask = torch.zeros(num_papers, dtype=torch.bool)
        train_mask[idx[:n_train]] = True
        val_mask[idx[n_train:n_train + n_val]] = True
        test_mask[idx[n_train + n_val:]] = True

        meta = {
            'num_classes': num_classes,
            'node_types': ['author', 'paper', 'venue'],
            'category': 'paper',
            'in_dim_dict': feat_dim,
            'num_timestamps': T,
        }

        return dict(graphs=graphs, feat_dicts=feat_dicts,
                    labels=labels, train_mask=train_mask,
                    val_mask=val_mask, test_mask=test_mask,
                    meta=meta)

    def _load_real_aminer(self):
        """
        加载真实 Aminer 数据集。
        用户需从 https://www.aminer.cn/heterogeneous_dataset 下载数据，
        放置于 raw_dir 目录，并实现此函数。
        """
        raise NotImplementedError(
            "Please download Aminer dataset from https://www.aminer.cn/heterogeneous_dataset "
            "and implement _load_real_aminer(). Or use use_synthetic=True for debugging."
        )


# ──────────────────────────────────────────────────────────────────────────────
# COVID-19 数据集（流行病学 HTG，节点回归）
# 节点类型: state, county
# 边类型: (county, belongs_to, state), (state, adjacent, state)
# 任务: 预测各县/州的新增病例数
# ──────────────────────────────────────────────────────────────────────────────
class COVID19HTGDataset(HTGDatasetBase):
    """
    COVID-19 流行病学异质时序图数据集。

    数据集统计（来自 HTGExplainer 论文）：
      - 时间切片: 304 天（2020.05 - 2021.02）
      - 节点类型: state (51), county (3143)
      - 任务: 节点回归（预测新增病例）

    注：本实现使用合成数据，真实数据需从
    https://github.com/jliang993/HTGExplainer 下载。
    """

    def __init__(self, raw_dir='./data/covid19', num_timestamps=30,
                 use_synthetic=True):
        self._num_timestamps = num_timestamps
        self.use_synthetic = use_synthetic
        super().__init__(name='covid19_htg', raw_dir=raw_dir)

    def _build_snapshots(self):
        if self.use_synthetic:
            return self._generate_synthetic_covid()
        raise NotImplementedError

    def _generate_synthetic_covid(self):
        T = self._num_timestamps
        num_states = 51
        num_counties = 300   # 简化版
        feat_dim = {'state': 32, 'county': 32}

        graphs = []
        feat_dicts = []

        for t in range(T):
            # county → state 归属关系
            county_state_src = torch.arange(num_counties) % num_states
            county_state_dst = torch.arange(num_counties) % num_states

            # state 邻接关系（随机，模拟地理邻接）
            adj_pairs = torch.randint(0, num_states, (100, 2))
            state_adj_src, state_adj_dst = adj_pairs[:, 0], adj_pairs[:, 1]

            g = dgl.heterograph({
                ('county', 'belongs_to', 'state'): (
                    torch.arange(num_counties), county_state_src),
                ('state', 'has', 'county'): (
                    county_state_src, torch.arange(num_counties)),
                ('state', 'adjacent', 'state'): (
                    state_adj_src, state_adj_dst),
            }, num_nodes_dict={
                'state': num_states,
                'county': num_counties,
            })
            graphs.append(g)

            feat_dicts.append({
                'state': torch.randn(num_states, feat_dim['state']),
                'county': torch.randn(num_counties, feat_dim['county']),
            })

        # 标签：每个 county 每天的新增病例（连续值）
        labels = torch.randn(num_counties).abs() * 100

        idx = torch.randperm(num_counties)
        n_train = int(0.6 * num_counties)
        n_val = int(0.2 * num_counties)
        train_mask = torch.zeros(num_counties, dtype=torch.bool)
        val_mask = torch.zeros(num_counties, dtype=torch.bool)
        test_mask = torch.zeros(num_counties, dtype=torch.bool)
        train_mask[idx[:n_train]] = True
        val_mask[idx[n_train:n_train + n_val]] = True
        test_mask[idx[n_train + n_val:]] = True

        meta = {
            'num_classes': 1,   # 回归任务
            'node_types': ['state', 'county'],
            'category': 'county',
            'in_dim_dict': feat_dim,
            'num_timestamps': T,
            'task': 'regression',
        }
        return dict(graphs=graphs, feat_dicts=feat_dicts,
                    labels=labels, train_mask=train_mask,
                    val_mask=val_mask, test_mask=test_mask,
                    meta=meta)


# ──────────────────────────────────────────────────────────────────────────────
# 自定义数据集接口
# ──────────────────────────────────────────────────────────────────────────────
class CustomHTGDataset(HTGDatasetBase):
    """
    用户自定义数据集接口。
    传入已构建好的图列表、特征字典列表和标签，
    自动完成 train/val/test 划分。

    使用示例：
        graphs = [g0, g1, ..., gT]      # T 个 DGLHeteroGraph
        feat_dicts = [f0, f1, ..., fT]  # List of {ntype: Tensor}
        labels = torch.tensor([...])    # 目标节点标签

        dataset = CustomHTGDataset(
            name='my_dataset',
            graphs=graphs,
            feat_dicts=feat_dicts,
            labels=labels,
            category='paper',
            num_classes=4,
            split_ratio=(0.6, 0.2, 0.2),
        )
    """

    def __init__(self, name, graphs, feat_dicts, labels, category,
                 num_classes, split_ratio=(0.6, 0.2, 0.2)):
        self._graphs_input = graphs
        self._feat_dicts_input = feat_dicts
        self._labels_input = labels
        self._category_input = category
        self._num_classes_input = num_classes
        self._split_ratio = split_ratio
        super().__init__(name=name)

    def download(self):
        pass

    def process(self):
        graphs = self._graphs_input
        feat_dicts = self._feat_dicts_input
        labels = self._labels_input

        N = labels.shape[0]
        idx = torch.randperm(N)
        r = self._split_ratio
        n1, n2 = int(r[0] * N), int((r[0] + r[1]) * N)

        train_mask = torch.zeros(N, dtype=torch.bool)
        val_mask = torch.zeros(N, dtype=torch.bool)
        test_mask = torch.zeros(N, dtype=torch.bool)
        train_mask[idx[:n1]] = True
        val_mask[idx[n1:n2]] = True
        test_mask[idx[n2:]] = True

        node_types = list(feat_dicts[0].keys())
        in_dim_dict = {k: v.shape[-1] for k, v in feat_dicts[0].items()}

        self.graphs = graphs
        self.feat_dicts = feat_dicts
        self.labels = labels
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask
        self.meta = {
            'num_classes': self._num_classes_input,
            'node_types': node_types,
            'category': self._category_input,
            'in_dim_dict': in_dim_dict,
            'num_timestamps': len(graphs),
        }


# ──────────────────────────────────────────────────────────────────────────────
# 数据集注册表（供 OpenHGNN 调用）
# ──────────────────────────────────────────────────────────────────────────────
_DATASET_REGISTRY = {
    'aminer_htg': AminerHTGDataset,
    'covid19_htg': COVID19HTGDataset,
}


def build_dataset(name: str, **kwargs):
    """
    构建数据集的工厂函数。
    在 OpenHGNN 中集成时，注册为 @register_dataset('aminer_htg') 等。
    """
    if name not in _DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset '{name}'. "
                         f"Available: {list(_DATASET_REGISTRY.keys())}")
    return _DATASET_REGISTRY[name](**kwargs)
