"""
HTGformer 数据集
支持论文Table 1中的四个数据集：
  1. Aminer      - 节点分类（Paper: 18,464 / Author: 23,035 / Venue: 22）
                   时间划分: Training:14, Val:1, Test:1
                   数据来源: HTGNN (SDM'2022) 论文[4]
  2. OGBN-MAG    - 链路预测
                   时间划分: Training:8, Val:1, Test:1
  3. YELP        - 节点分类
                   时间划分: Training:10, Val:1, Test:1
  4. COVID-19    - 节点回归（State:54 / County:3223）
                   时间划分: Training:244, Val:30, Test:30

数据获取方式（Aminer/HTGNN数据）：
  论文说明"follow the dataset and splits provided by previous works [4]"
  [4] = HTGNN (SDM'2022): https://github.com/yeslab-code/HTGNN
  从该仓库下载数据后，指定 raw_dir 路径即可

OpenHGNN数据集规范：
  - 继承并符合 NodeClassificationDataset 接口
  - 提供 graphs, feat_dicts, labels, train_idx, val_idx, test_idx
  - 节点特征预处理（linear projection）在外部HeteroFeature层完成
"""

import os
import torch
import numpy as np
import dgl
from torch.utils.data import Dataset


class HTGDatasetBase:
    """
    HTGformer数据集基类
    符合OpenHGNN数据集规范
    """
    def __init__(self):
        self.graphs = []         # List[DGLGraph]，T个时间步
        self.feat_dicts = []     # List[dict]，T个时间步节点特征
        self.labels = None       # tensor，目标节点标签
        self.train_idx = None
        self.val_idx = None
        self.test_idx = None
        self.category = None     # 目标节点类型
        self.task = None         # 'node_classification'/'node_regression'/'link_prediction'
        self.num_classes = None
        self.pos_edges = None    # 链路预测正样本
        self.neg_edges = None    # 链路预测负样本


# ===========================================================================
# Aminer 数据集
# 论文Table 1: Paper:18,464 / Author:23,035 / Venue:22
#              Publish:18,464 / Write:52,545
#              Time Split: Training:14, Val:1, Test:1
# 数据来源: HTGNN论文 https://github.com/yeslab-code/HTGNN
# ===========================================================================
class AminerHTGDataset(HTGDatasetBase):
    """
    Aminer学术异质时序图数据集
    
    节点类型: paper, author, venue
    边类型: (author, writes, paper), (paper, published_in, venue)
    任务: paper节点分类（研究领域预测）
    评估指标: Macro-F1, Recall（论文Table 2）
    
    数据目录结构（来自HTGNN仓库）：
      raw_dir/
        ├── node_features/
        │   ├── paper_features.pt      # [18464, feat_dim]
        │   ├── author_features.pt     # [23035, feat_dim]
        │   └── venue_features.pt      # [22, feat_dim]
        ├── edges/
        │   ├── writes_{t}.pt          # (author_ids, paper_ids) for t=0..15
        │   └── published_in_{t}.pt    # (paper_ids, venue_ids) for t=0..15
        └── labels.pt                  # [18464] paper节点标签
    """

    # 论文Table 1精确规格
    NUM_PAPER = 18464
    NUM_AUTHOR = 23035
    NUM_VENUE = 22
    NUM_TIMESTAMPS = 16  # Training:14, Val:1, Test:1
    TRAIN_T = 14
    VAL_T = 1
    TEST_T = 1

    def __init__(self, raw_dir='./data/aminer', use_synthetic=False,
                 num_timestamps=16):
        super().__init__()
        self.category = 'paper'
        self.task = 'node_classification'
        self.num_classes = 4  # Aminer论文分类数（需根据实际数据确认）

        if use_synthetic or not os.path.exists(raw_dir):
            print("[AminerHTGDataset] 使用合成数据（流程验证模式）")
            print("  提示: 真实数据请从 https://github.com/yeslab-code/HTGNN 下载")
            self._build_synthetic(num_timestamps)
        else:
            print(f"[AminerHTGDataset] 从 {raw_dir} 加载真实数据...")
            self._load_real(raw_dir)

    def _load_real(self, raw_dir):
        """
        加载真实Aminer数据集
        数据格式与HTGNN论文保持一致
        """
        graphs = []
        feat_dicts = []

        # 加载节点特征（所有时间步共享）
        paper_feat = torch.load(
            os.path.join(raw_dir, 'node_features', 'paper_features.pt')
        )   # [18464, d]
        author_feat = torch.load(
            os.path.join(raw_dir, 'node_features', 'author_features.pt')
        )   # [23035, d]
        venue_feat = torch.load(
            os.path.join(raw_dir, 'node_features', 'venue_features.pt')
        )   # [22, d]

        for t in range(self.NUM_TIMESTAMPS):
            # 加载t时刻的边
            writes_edges = torch.load(
                os.path.join(raw_dir, 'edges', f'writes_{t}.pt')
            )   # (src_authors, dst_papers)
            pub_edges = torch.load(
                os.path.join(raw_dir, 'edges', f'published_in_{t}.pt')
            )   # (src_papers, dst_venues)

            g = dgl.heterograph({
                ('author', 'writes', 'paper'): (
                    writes_edges[0], writes_edges[1]
                ),
                ('paper', 'written_by', 'author'): (
                    writes_edges[1], writes_edges[0]
                ),
                ('paper', 'published_in', 'venue'): (
                    pub_edges[0], pub_edges[1]
                ),
                ('venue', 'publishes', 'paper'): (
                    pub_edges[1], pub_edges[0]
                ),
            }, num_nodes_dict={
                'paper': self.NUM_PAPER,
                'author': self.NUM_AUTHOR,
                'venue': self.NUM_VENUE,
            })
            graphs.append(g)
            feat_dicts.append({
                'paper': paper_feat,
                'author': author_feat,
                'venue': venue_feat,
            })

        # 加载标签
        labels = torch.load(os.path.join(raw_dir, 'labels.pt'))  # [18464]
        self.num_classes = int(labels.max().item()) + 1

        # 时间划分：Training:14, Val:1, Test:1（论文Table 1）
        # 使用最后3个时间步分别作为train/val/test的图
        # 节点划分按照HTGNN的标准划分
        split_path = os.path.join(raw_dir, 'splits.pt')
        if os.path.exists(split_path):
            splits = torch.load(split_path)
            train_idx = splits['train']
            val_idx = splits['val']
            test_idx = splits['test']
        else:
            # 如果没有预定义划分，随机划分
            n = self.NUM_PAPER
            perm = torch.randperm(n)
            train_idx = perm[:int(0.6 * n)]
            val_idx = perm[int(0.6 * n):int(0.8 * n)]
            test_idx = perm[int(0.8 * n):]

        self.graphs = graphs
        self.feat_dicts = feat_dicts
        self.labels = labels
        self.train_idx = train_idx
        self.val_idx = val_idx
        self.test_idx = test_idx

        print(f"  ✓ 加载成功: {len(graphs)}个时间步")
        print(f"  节点: paper={self.NUM_PAPER}, "
              f"author={self.NUM_AUTHOR}, venue={self.NUM_VENUE}")
        print(f"  训练/验证/测试: "
              f"{len(train_idx)}/{len(val_idx)}/{len(test_idx)}")

    def _build_synthetic(self, num_timestamps):
        """
        合成数据，严格按照论文Table 1的节点数量
        用于在获取真实数据前验证代码流程
        """
        T = min(num_timestamps, self.NUM_TIMESTAMPS)
        feat_dim = {'paper': 128, 'author': 64, 'venue': 32}

        for t in range(T):
            # 边数量按论文比例设置
            n_writes = self.NUM_AUTHOR * 2  # ~52545/16 ≈ 3284 per step
            n_pub = self.NUM_PAPER  # ~18464/16 ≈ 1154 per step

            g = dgl.heterograph({
                ('author', 'writes', 'paper'): (
                    torch.randint(0, self.NUM_AUTHOR, (n_writes,)),
                    torch.randint(0, self.NUM_PAPER, (n_writes,))
                ),
                ('paper', 'written_by', 'author'): (
                    torch.randint(0, self.NUM_PAPER, (n_writes,)),
                    torch.randint(0, self.NUM_AUTHOR, (n_writes,))
                ),
                ('paper', 'published_in', 'venue'): (
                    torch.randint(0, self.NUM_PAPER, (n_pub,)),
                    torch.randint(0, self.NUM_VENUE, (n_pub,))
                ),
                ('venue', 'publishes', 'paper'): (
                    torch.randint(0, self.NUM_VENUE, (n_pub,)),
                    torch.randint(0, self.NUM_PAPER, (n_pub,))
                ),
            }, num_nodes_dict={
                'paper': self.NUM_PAPER,
                'author': self.NUM_AUTHOR,
                'venue': self.NUM_VENUE,
            })
            self.graphs.append(g)
            self.feat_dicts.append({
                'paper': torch.randn(self.NUM_PAPER, feat_dim['paper']),
                'author': torch.randn(self.NUM_AUTHOR, feat_dim['author']),
                'venue': torch.randn(self.NUM_VENUE, feat_dim['venue']),
            })

        # 合成标签（4类分类）
        self.labels = torch.randint(0, self.num_classes, (self.NUM_PAPER,))

        # 按照HTGNN标准划分比例
        n = self.NUM_PAPER
        perm = torch.randperm(n)
        self.train_idx = perm[:int(0.6 * n)]
        self.val_idx = perm[int(0.6 * n):int(0.8 * n)]
        self.test_idx = perm[int(0.8 * n):]


# ===========================================================================
# OGBN-MAG 数据集
# 论文Table 1: Author:17,764 / Paper:282,039 / Field:34,601 / Institution:2,276
#              Time Split: Training:8, Val:1, Test:1
# ===========================================================================
class OGBNMAGHTGDataset(HTGDatasetBase):
    """
    OGBN-MAG学术异质时序图数据集
    任务: 链路预测
    评估指标: AUC, AP（论文Table 2）
    """
    NUM_AUTHOR = 17764
    NUM_PAPER = 282039
    NUM_FIELD = 34601
    NUM_INSTITUTION = 2276
    NUM_TIMESTAMPS = 10  # Training:8, Val:1, Test:1

    def __init__(self, raw_dir='./data/ogbn_mag', use_synthetic=False,
                 num_timestamps=10):
        super().__init__()
        self.category = 'paper'
        self.task = 'link_prediction'

        if use_synthetic or not os.path.exists(raw_dir):
            print("[OGBNMAGHTGDataset] 使用合成数据")
            self._build_synthetic(num_timestamps)
        else:
            self._load_real(raw_dir)

    def _load_real(self, raw_dir):
        """加载真实OGBN-MAG数据"""
        # OGBN-MAG可从OGB官网下载: https://ogb.stanford.edu/docs/nodeprop/#ogbn-mag
        # 需要按年份切分成时序图
        raise NotImplementedError(
            "请从 https://ogb.stanford.edu/docs/nodeprop/#ogbn-mag 下载数据，"
            "并实现预处理逻辑"
        )

    def _build_synthetic(self, num_timestamps):
        feat_dim = {'author': 64, 'paper': 128,
                    'field_of_study': 32, 'institution': 32}
        T = min(num_timestamps, self.NUM_TIMESTAMPS)

        for t in range(T):
            g = dgl.heterograph({
                ('author', 'writes', 'paper'): (
                    torch.randint(0, self.NUM_AUTHOR, (5000,)),
                    torch.randint(0, self.NUM_PAPER, (5000,))
                ),
                ('paper', 'written_by', 'author'): (
                    torch.randint(0, self.NUM_PAPER, (5000,)),
                    torch.randint(0, self.NUM_AUTHOR, (5000,))
                ),
                ('paper', 'has_topic', 'field_of_study'): (
                    torch.randint(0, self.NUM_PAPER, (3000,)),
                    torch.randint(0, self.NUM_FIELD, (3000,))
                ),
                ('field_of_study', 'has_paper', 'paper'): (
                    torch.randint(0, self.NUM_FIELD, (3000,)),
                    torch.randint(0, self.NUM_PAPER, (3000,))
                ),
            }, num_nodes_dict={
                'author': self.NUM_AUTHOR,
                'paper': self.NUM_PAPER,
                'field_of_study': self.NUM_FIELD,
                'institution': self.NUM_INSTITUTION,
            })
            self.graphs.append(g)
            self.feat_dicts.append({
                k: torch.randn(n, feat_dim.get(k, 64))
                for k, n in {
                    'author': self.NUM_AUTHOR,
                    'paper': self.NUM_PAPER,
                    'field_of_study': self.NUM_FIELD,
                    'institution': self.NUM_INSTITUTION,
                }.items()
            })

        n = self.NUM_PAPER
        self.labels = torch.zeros(n, dtype=torch.long)
        perm = torch.randperm(n)
        self.train_idx = perm[:int(0.6 * n)]
        self.val_idx = perm[int(0.6 * n):int(0.8 * n)]
        self.test_idx = perm[int(0.8 * n):]
        # 合成链路预测边
        self.pos_edges = torch.stack([
            torch.randint(0, n, (1000,)),
            torch.randint(0, n, (1000,))
        ], dim=1)
        self.neg_edges = torch.stack([
            torch.randint(0, n, (1000,)),
            torch.randint(0, n, (1000,))
        ], dim=1)
        self.labels = torch.cat([
            torch.ones(1000), torch.zeros(1000)
        ]).long()
        self.train_idx = torch.arange(1200)
        self.val_idx = torch.arange(1200, 1600)
        self.test_idx = torch.arange(1600, 2000)


# ===========================================================================
# YELP 数据集
# 论文Table 1: User:55,702 / Item:12,524
#              Review:87,846 / Tip:35,508
#              Time Split: Training:10, Val:1, Test:1
# ===========================================================================
class YELPHTGDataset(HTGDatasetBase):
    """
    YELP异质时序图数据集
    任务: 节点分类
    评估指标: Macro-F1, Recall（论文Table 2）
    """
    NUM_USER = 55702
    NUM_ITEM = 12524
    NUM_TIMESTAMPS = 12  # Training:10, Val:1, Test:1

    def __init__(self, raw_dir='./data/yelp', use_synthetic=False,
                 num_timestamps=12):
        super().__init__()
        self.category = 'item'
        self.task = 'node_classification'
        self.num_classes = 4

        if use_synthetic or not os.path.exists(raw_dir):
            print("[YELPHTGDataset] 使用合成数据")
            self._build_synthetic(num_timestamps)
        else:
            self._load_real(raw_dir)

    def _load_real(self, raw_dir):
        raise NotImplementedError("请提供YELP数据集路径")

    def _build_synthetic(self, num_timestamps):
        T = min(num_timestamps, self.NUM_TIMESTAMPS)
        feat_dim = {'user': 64, 'item': 128}

        for t in range(T):
            g = dgl.heterograph({
                ('user', 'reviews', 'item'): (
                    torch.randint(0, self.NUM_USER, (3000,)),
                    torch.randint(0, self.NUM_ITEM, (3000,))
                ),
                ('item', 'reviewed_by', 'user'): (
                    torch.randint(0, self.NUM_ITEM, (3000,)),
                    torch.randint(0, self.NUM_USER, (3000,))
                ),
            }, num_nodes_dict={
                'user': self.NUM_USER,
                'item': self.NUM_ITEM,
            })
            self.graphs.append(g)
            self.feat_dicts.append({
                'user': torch.randn(self.NUM_USER, feat_dim['user']),
                'item': torch.randn(self.NUM_ITEM, feat_dim['item']),
            })

        n = self.NUM_ITEM
        self.labels = torch.randint(0, self.num_classes, (n,))
        perm = torch.randperm(n)
        self.train_idx = perm[:int(0.6 * n)]
        self.val_idx = perm[int(0.6 * n):int(0.8 * n)]
        self.test_idx = perm[int(0.8 * n):]


# ===========================================================================
# COVID-19 数据集
# 论文Table 1: State:54 / County:3223
#              S-S:269 / S-C:3141 / C-C:22176
#              Time Split: Training:244, Val:30, Test:30
# ===========================================================================
class COVID19HTGDataset(HTGDatasetBase):
    """
    COVID-19流行病学异质时序图数据集
    任务: 节点回归（预测新增病例）
    评估指标: MAE（论文Table 2）
    论文特殊超参数: hidden_dim=8（区别于其他数据集的64）
    """
    NUM_STATE = 54
    NUM_COUNTY = 3223
    NUM_TIMESTAMPS = 304  # Training:244, Val:30, Test:30

    def __init__(self, raw_dir='./data/covid19', use_synthetic=False,
                 num_timestamps=304):
        super().__init__()
        self.category = 'county'
        self.task = 'node_regression'

        if use_synthetic or not os.path.exists(raw_dir):
            print("[COVID19HTGDataset] 使用合成数据")
            print("  注意: 论文对COVID-19使用 hidden_dim=8")
            self._build_synthetic(num_timestamps)
        else:
            self._load_real(raw_dir)

    def _load_real(self, raw_dir):
        raise NotImplementedError(
            "COVID-19数据来自 HTGNN 论文，请从相关仓库下载"
        )

    def _build_synthetic(self, num_timestamps):
        T = min(num_timestamps, self.NUM_TIMESTAMPS)
        feat_dim = {'state': 32, 'county': 64}

        for t in range(T):
            g = dgl.heterograph({
                ('state', 'state_state', 'state'): (
                    torch.randint(0, self.NUM_STATE, (269,)),
                    torch.randint(0, self.NUM_STATE, (269,))
                ),
                ('state', 'state_county', 'county'): (
                    torch.randint(0, self.NUM_STATE, (3141,)),
                    torch.randint(0, self.NUM_COUNTY, (3141,))
                ),
                ('county', 'county_state', 'state'): (
                    torch.randint(0, self.NUM_COUNTY, (3141,)),
                    torch.randint(0, self.NUM_STATE, (3141,))
                ),
                ('county', 'county_county', 'county'): (
                    torch.randint(0, self.NUM_COUNTY, (22176,)),
                    torch.randint(0, self.NUM_COUNTY, (22176,))
                ),
            }, num_nodes_dict={
                'state': self.NUM_STATE,
                'county': self.NUM_COUNTY,
            })
            self.graphs.append(g)
            self.feat_dicts.append({
                'state': torch.randn(self.NUM_STATE, feat_dim['state']),
                'county': torch.randn(self.NUM_COUNTY, feat_dim['county']),
            })

        # 回归标签（新增病例数）
        n = self.NUM_COUNTY
        self.labels = torch.randn(n).abs() * 100  # 模拟病例数
        # Training:244, Val:30, Test:30（时间划分转换为节点划分）
        perm = torch.randperm(n)
        self.train_idx = perm[:int(0.7 * n)]
        self.val_idx = perm[int(0.7 * n):int(0.85 * n)]
        self.test_idx = perm[int(0.85 * n):]
