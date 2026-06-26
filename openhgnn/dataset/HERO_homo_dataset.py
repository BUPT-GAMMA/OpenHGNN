import os
import pickle
import random
import numpy as np
import torch
import torch as th
import torch.nn.functional as F
import dgl
from dgl.data import DGLDataset
from dgl.data import (
    CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset,
    AmazonCoBuyPhotoDataset, AmazonCoBuyComputerDataset,
    CoauthorCSDataset, CoauthorPhysicsDataset,
    CoraFullDataset, WikiCSDataset
)
from ogb.nodeproppred import DglNodePropPredDataset


def pairwise_distance(x, y=None):
    if not isinstance(x, th.Tensor):
        x = th.tensor(x, dtype=th.float32)
    else:
        x = x.float()

    x = x.unsqueeze(0).permute(0, 2, 1)

    if y is None:
        y = x
    else:
        if not isinstance(y, th.Tensor):
            y = th.tensor(y, dtype=th.float32)
        else:
            y = y.float()
        y = y.unsqueeze(0).permute(0, 2, 1)

    y = y.permute(0, 2, 1)

    A = -2 * th.bmm(y, x)
    A += th.sum(y ** 2, dim=2, keepdim=True)
    A += th.sum(x ** 2, dim=1, keepdim=True)
    return A.squeeze()


def random_aug(feat, dfr=0.0):
    return feat.clone()

def setup_seed(seed=0):
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class HEROHomoDataset(DGLDataset):
    HOMO_DATASETS = {
        "cora4HERO": "cora",
        "citeseer4HERO": "citeseer",
        "pubmed4HERO": "pubmed",
        "photo4HERO": "photo",
        "computers4HERO": "computers",
        "cs4HERO": "cs",
        "physics4HERO": "physics",
        "corafull4HERO": "corafull",
        "wikics4HERO": "wikics",
        "ogbn-arxiv4HERO": "ogbn-arxiv",
    }

    def __init__(
        self,
        name,
        raw_dir=None,
        force_reload=False,
        verbose=True,
        seed=0,
        edge_rate=None,
        dfr=0.0,
    ):
        assert name in self.HOMO_DATASETS, \
            f"name must be one of {list(self.HOMO_DATASETS.keys())}"

        self.dataset_name = name
        self.base_name = self.HOMO_DATASETS[name]
        self.seed = seed
        self.edge_rate = edge_rate
        self.dfr = dfr

        # 与 HERODataset 对齐
        # .../openhgnn/dataset
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        # .../openhgnn/dataset/xxx4HERO
        self.dataset_dir = os.path.join(self.base_dir, name)
        self.raw_data_dir = os.path.join(self.dataset_dir, "raw")

        self.g_path = os.path.join(self.dataset_dir, "graph.bin")
        self.info_path = os.path.join(self.dataset_dir, "info.pkl")
        self.homo_data_path = os.path.join(self.dataset_dir, "homo_data.pkl")

        if raw_dir is None:
            raw_dir = self.base_dir

        super(HEROHomoDataset, self).__init__(
            name=name,
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose
        )

    def download(self):
        os.makedirs(self.dataset_dir, exist_ok=True)
        os.makedirs(self.raw_data_dir, exist_ok=True)

        if self.verbose:
            print(f"[{self.dataset_name}] dataset_dir: {self.dataset_dir}")
            print(f"[{self.dataset_name}] raw_data_dir: {self.raw_data_dir}")

    def _build_dataset(self, name):
        if name == 'cora':
            dataset = CoraGraphDataset(raw_dir=self.raw_data_dir)
        elif name == 'citeseer':
            dataset = CiteseerGraphDataset(raw_dir=self.raw_data_dir)
        elif name == 'pubmed':
            dataset = PubmedGraphDataset(raw_dir=self.raw_data_dir)
        elif name == 'photo':
            dataset = AmazonCoBuyPhotoDataset(raw_dir=self.raw_data_dir)
        elif name == 'computers':
            dataset = AmazonCoBuyComputerDataset(raw_dir=self.raw_data_dir)
        elif name == 'cs':
            dataset = CoauthorCSDataset(raw_dir=self.raw_data_dir)
        elif name == 'physics':
            dataset = CoauthorPhysicsDataset(raw_dir=self.raw_data_dir)
        elif name == 'corafull':
            dataset = CoraFullDataset(raw_dir=self.raw_data_dir)
        elif name == 'wikics':
            dataset = WikiCSDataset(raw_dir=self.raw_data_dir)
        elif name == 'ogbn-arxiv':
            dataset = DglNodePropPredDataset(name='ogbn-arxiv', root=self.raw_data_dir)
        else:
            raise ValueError(f"Unsupported homogeneous dataset: {name}")

        return dataset

    def _mask_to_index(self, mask):
        if mask.dim() == 1:
            return th.nonzero(mask, as_tuple=False).squeeze(1)

        if mask.dim() == 2:
            mask0 = mask[0]
            return th.nonzero(mask0, as_tuple=False).squeeze(1)

        raise ValueError(f"Unsupported mask shape: {mask.shape}")

    def _load_homogeneous_graph(self, name):
        dataset = self._build_dataset(name)

        if name == 'ogbn-arxiv':
            graph, labels = dataset[0]
            labels = labels.squeeze()
        else:
            graph = dataset[0]
            labels = graph.ndata['label']
            graph.ndata.pop('label')

        citegraph = ['cora', 'citeseer', 'pubmed']
        cograph = ['photo', 'computers', 'cs', 'physics', 'corafull', 'ogbn-arxiv']

        if name in citegraph:
            train_mask = graph.ndata.pop('train_mask')
            val_mask = graph.ndata.pop('val_mask')
            test_mask = graph.ndata.pop('test_mask')

            train_idx = th.nonzero(train_mask, as_tuple=False).squeeze(1)
            val_idx = th.nonzero(val_mask, as_tuple=False).squeeze(1)
            test_idx = th.nonzero(test_mask, as_tuple=False).squeeze(1)

        elif name in cograph:
            train_ratio = 0.1
            val_ratio = 0.1

            N = graph.number_of_nodes()
            train_num = int(N * train_ratio)
            val_num = int(N * (train_ratio + val_ratio))

            idx = np.arange(N)
            np.random.shuffle(idx)

            train_idx = th.tensor(idx[:train_num], dtype=th.long)
            val_idx = th.tensor(idx[train_num:val_num], dtype=th.long)
            test_idx = th.tensor(idx[val_num:], dtype=th.long)

        elif name == "wikics":
            train_mask = graph.ndata.pop('train_mask')
            val_mask = graph.ndata.pop('val_mask')
            test_mask = graph.ndata.pop('test_mask')

            train_idx = self._mask_to_index(train_mask)
            val_idx = self._mask_to_index(val_mask)
            test_idx = self._mask_to_index(test_mask)

        else:
            raise ValueError(f"Unsupported homogeneous dataset split rule: {name}")

        feat = graph.ndata.pop('feat')

        if hasattr(dataset, "num_classes"):
            num_class = dataset.num_classes
        else:
            num_class = int(labels.max().item() + 1)

        return graph, feat, labels.squeeze(), train_idx, val_idx, test_idx, num_class

    def process(self):
        setup_seed(self.seed)

        graph, feat, labels, train_idx, val_idx, test_idx, num_class = \
            self._load_homogeneous_graph(self.base_name)

        graph = dgl.add_self_loop(graph)

        feat1 = random_aug(feat, self.dfr).float()
        feat2 = random_aug(feat, self.dfr).float()

        # 特征距离图
        feature_distance = pairwise_distance(feat2)
        feature_distance = F.normalize(feature_distance, dim=1)

        # 按 edge_rate 稀疏化
        if self.edge_rate is not None and self.edge_rate > 0:
            total = feature_distance.numel()
            k = int(total * self.edge_rate)
            k = max(1, min(k, total))

            kthvalue = th.kthvalue(feature_distance.reshape(-1), k).values
            mask = (feature_distance > kthvalue).float()
            feature_distance = feature_distance * mask

        self._graph = graph
        self._features = [feat1, feat2]
        self._feature_distance = feature_distance
        self._labels = labels.long()
        self._train_idx = train_idx.long()
        self._val_idx = val_idx.long()
        self._test_idx = test_idx.long()

        self._info = {
            "dataset": self.dataset_name,
            "graph_type": "homogeneous",
            "num_classes": int(num_class),
            "num_nodes": int(graph.number_of_nodes()),
            "target_ntype": None,
        }

    def save(self):
        os.makedirs(self.dataset_dir, exist_ok=True)

        dgl.save_graphs(self.g_path, [self._graph])

        with open(self.homo_data_path, "wb") as f:
            pickle.dump({
                "features": self._features,
                "feature_distance": self._feature_distance,
                "labels": self._labels,
                "train_idx": self._train_idx,
                "val_idx": self._val_idx,
                "test_idx": self._test_idx,
            }, f)

        with open(self.info_path, "wb") as f:
            pickle.dump(self._info, f)

        if self.verbose:
            print(f"[{self.dataset_name}] homogeneous graph saved to {self.g_path}")
            print(f"[{self.dataset_name}] homogeneous data saved to {self.homo_data_path}")
            print(f"[{self.dataset_name}] info saved to {self.info_path}")

    def load(self):
        graphs, _ = dgl.load_graphs(self.g_path)
        self._graph = graphs[0]

        with open(self.homo_data_path, "rb") as f:
            homo_data = pickle.load(f)

        self._features = homo_data["features"]
        self._feature_distance = homo_data["feature_distance"]
        self._labels = homo_data["labels"]
        self._train_idx = homo_data["train_idx"]
        self._val_idx = homo_data["val_idx"]
        self._test_idx = homo_data["test_idx"]

        with open(self.info_path, "rb") as f:
            self._info = pickle.load(f)

        if self.verbose:
            print(f"[{self.dataset_name}] homogeneous graph loaded from {self.g_path}")
            print(f"[{self.dataset_name}] homogeneous data loaded from {self.homo_data_path}")

    def has_cache(self):
        return (
            os.path.exists(self.g_path)
            and os.path.exists(self.homo_data_path)
            and os.path.exists(self.info_path)
        )

    def __getitem__(self, idx):
        assert idx == 0, "This dataset has only one graph"
        return self._graph

    def __len__(self):
        return 1

    @property
    def graph(self):
        return self._graph

    @property
    def features(self):
        return self._features

    @property
    def feature_distance(self):
        return self._feature_distance

    @property
    def labels(self):
        return self._labels

    @property
    def train_idx(self):
        return self._train_idx

    @property
    def val_idx(self):
        return self._val_idx

    @property
    def test_idx(self):
        return self._test_idx

    @property
    def num_classes(self):
        return self._info["num_classes"]

    @property
    def info(self):
        return self._info