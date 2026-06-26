# 文件：openhgnn/dataset/HERO_dataset.py
import os
import pickle
import zipfile
import shutil
import urllib.request
import random
import numpy as np
import scipy.sparse as sp
import torch
import torch as th
import torch.nn.functional as F
import dgl
from dgl.data import DGLDataset
from dgl.data.utils import save_graphs, load_graphs

def preprocess_features(features, norm=True):
    if sp.issparse(features):
        features = features.toarray()
    if not isinstance(features, np.ndarray):
        features = np.array(features)
    features = features.copy()
    if norm:
        features[features > 0] = 1
    return th.tensor(features, dtype=th.float32)

def pairwise_distance(x, y=None):
    """
    计算平方欧氏距离矩阵
    """
    if not isinstance(x, th.Tensor):
        x = th.tensor(x, dtype=th.float32)
    else:
        x = x.float()
    x = x.unsqueeze(0).permute(0, 2, 1)  # [1, F, N]
    if y is None:
        y = x
    else:
        if not isinstance(y, th.Tensor):
            y = th.tensor(y, dtype=th.float32)
        else:
            y = y.float()
        y = y.unsqueeze(0).permute(0, 2, 1)  # [1, F, M]
    y = y.permute(0, 2, 1)  # [1, M, F]

    A = -2 * th.bmm(y, x)
    A += th.sum(y ** 2, dim=2, keepdim=True)
    A += th.sum(x ** 2, dim=1, keepdim=True)
    return A.squeeze()

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
    torch.backends.cudnn.enabled = True

class HERODataset(DGLDataset):
    _urls = {
        "ACM4HERO": "https://huggingface.co/datasets/jiajia101/Openhgnn/resolve/main/ACM.zip",
        "Aminer4HERO": "https://huggingface.co/datasets/jiajia101/Openhgnn/resolve/main/Aminer.zip",
        "DBLP4HERO": "https://huggingface.co/datasets/jiajia101/Openhgnn/resolve/main/DBLP.zip",
        "Yelp4HERO": "https://huggingface.co/datasets/jiajia101/Openhgnn/resolve/main/Yelp.zip"
    }

    def __init__(
        self,name,raw_dir=None,force_reload=False,verbose=True,seed=0,edge_rate=None,feat_norm=True,save_feature_distance=True):
        assert name in ["ACM4HERO", "Aminer4HERO", "DBLP4HERO", "Yelp4HERO"], \
            "name should be one of ['ACM4HERO', 'Aminer4HERO', 'DBLP4HERO', 'Yelp4HERO']"

        self.dataset_name = name
        self.seed = seed
        self.edge_rate=edge_rate
        self.feat_norm = feat_norm
        self.save_feature_distance = save_feature_distance
        # .../openhgnn/dataset
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        # .../openhgnn/dataset/xxx4HERO
        self.dataset_dir = os.path.join(self.base_dir, name)
        if name == "ACM4HERO":
            zip_name = "ACM.zip"
        elif name == "Aminer4HERO":
            zip_name = "Aminer.zip"
        elif name == "Yelp4HERO":
            zip_name = "Yelp.zip"
        else:
            zip_name = "DBLP.zip"
        self.data_path = os.path.join(self.dataset_dir, zip_name)
        self.raw_data_dir = os.path.join(self.dataset_dir, "raw")
        self.g_path = os.path.join(self.dataset_dir, "graph.bin")
        self.info_path = os.path.join(self.dataset_dir, "info.pkl")
        url = self._urls[name]

        if raw_dir is None:
            raw_dir = self.base_dir
        super(HERODataset, self).__init__(name=name,url=url,raw_dir=raw_dir,force_reload=force_reload,verbose=verbose)
    
    def download(self):
        os.makedirs(self.dataset_dir, exist_ok=True)
        os.makedirs(self.raw_data_dir, exist_ok=True)
        if self._raw_files_exist():
            if self.verbose:
                print(f"[{self.dataset_name}] raw files already exist, skip download.")
            return
        if not os.path.exists(self.data_path):
            if self.verbose:
                print(f"[{self.dataset_name}] downloading from {self.url}")
                print(f"[{self.dataset_name}] saving zip to {self.data_path}")
            urllib.request.urlretrieve(self.url, self.data_path)
        if self.verbose:
            print(f"[{self.dataset_name}] extracting {self.data_path} to {self.raw_data_dir}")
        if os.path.exists(self.raw_data_dir):
            shutil.rmtree(self.raw_data_dir)
        os.makedirs(self.raw_data_dir, exist_ok=True)

        with zipfile.ZipFile(self.data_path, "r") as zf:
            zf.extractall(self.raw_data_dir)
        self._flatten_raw_dir_if_needed()
        if not self._raw_files_exist():
            raise FileNotFoundError(
                f"[{self.dataset_name}] raw files are incomplete after extraction in {self.raw_data_dir}")
    
    def process(self):
        setup_seed(self.seed)
        if self.dataset_name == "ACM4HERO":
            self._process_acm()
        elif self.dataset_name == "Aminer4HERO":
            self._process_aminer()
        elif self.dataset_name == "DBLP4HERO":
            self._process_dblp()
        elif self.dataset_name == "Yelp4HERO":
            self._process_yelp()
        else:
            raise ValueError(f"Unsupported dataset name: {self.dataset_name}")

    def _build_feature_distance(self, target_feat, do_normalize=True):
        dist = pairwise_distance(target_feat)
        if do_normalize:
            dist = F.normalize(dist, dim=1)
        total = dist.numel()
        k = int(total * self.edge_rate)
        if k <= 0:
            return dist
        kthvalue = th.kthvalue(dist.reshape(-1), k).values
        # mask = (dist > kthvalue).float()
        mask = (dist > kthvalue).detach().float()
        dist = dist * mask
        return dist

    def _process_aminer(self):
        def _maybe_to_local(ids, offset, num_nodes, name):
            ids = np.asarray(ids, dtype=np.int64)
            if ids.min() >= 0 and ids.max() < num_nodes:
                return ids
            if ids.min() >= offset and ids.max() < offset + num_nodes:
                return ids - offset
            raise ValueError(
                f"{name} ids are out of expected range: "
                f"min={ids.min()}, max={ids.max()}, "
                f"expected local [0,{num_nodes-1}] or global [{offset},{offset+num_nodes-1}]"
            )

        raw_dir = self.raw_data_dir
        feature_list = []
        for i in range(3):
            x = np.load(os.path.join(raw_dir, f"features_{i}.npy"))
            feature_list.append(x)
        features = np.concatenate(feature_list, axis=0)

        features = preprocess_features(features, norm=True)
        labels = np.load(os.path.join(raw_dir, "labels.npy")).astype(np.int64)

        pa = np.loadtxt(os.path.join(raw_dir, "pa.txt")).astype(np.int64).T
        pr = np.loadtxt(os.path.join(raw_dir, "pr.txt")).astype(np.int64).T

        paper_num = 6564
        author_num = 19893 - 6564      # 13329
        reference_num = 55783 - 19893  # 35890

        pa_src = _maybe_to_local(pa[0], 0, paper_num, "pa paper")
        pa_dst = _maybe_to_local(pa[1], 6564, author_num, "pa author")

        pr_src = _maybe_to_local(pr[0], 0, paper_num, "pr paper")
        pr_dst = _maybe_to_local(pr[1], 19893, reference_num, "pr reference")

        data_dict = {
            ("paper", "paper-author", "author"): (
                th.tensor(pa_src, dtype=th.int64),
                th.tensor(pa_dst, dtype=th.int64),
            ),
            ("author", "author-paper", "paper"): (
                th.tensor(pa_dst, dtype=th.int64),
                th.tensor(pa_src, dtype=th.int64),
            ),
            ("paper", "paper-ref", "reference"): (
                th.tensor(pr_src, dtype=th.int64),
                th.tensor(pr_dst, dtype=th.int64),
            ),
            ("reference", "ref-paper", "paper"): (
                th.tensor(pr_dst, dtype=th.int64),
                th.tensor(pr_src, dtype=th.int64),
            ),
        }
        idx = {"paper": np.arange(0, 6564),"author": np.arange(6564, 19893),"reference": np.arange(19893, 55783),}
        g = dgl.heterograph(data_dict,num_nodes_dict={"paper": 6564,"author": 19893 - 6564, "reference": 55783 - 19893,})

        g.nodes["paper"].data["feat"] = features[idx["paper"]]
        g.nodes["author"].data["feat"] = features[idx["author"]]
        g.nodes["reference"].data["feat"] = features[idx["reference"]]
        g.nodes["paper"].data["label"] = th.tensor(labels, dtype=th.long)

        ratio = [20, 40, 60]
        train = [np.load(os.path.join(raw_dir, f"train_{i}.npy")).astype(np.int64) for i in ratio]
        val = [np.load(os.path.join(raw_dir, f"val_{i}.npy")).astype(np.int64) for i in ratio]
        test = [np.load(os.path.join(raw_dir, f"test_{i}.npy")).astype(np.int64) for i in ratio]

        train_idx = train[2]
        val_idx = val[2]
        test_idx = test[2]

        num_paper = g.num_nodes("paper")

        train_mask = np.zeros(num_paper, dtype=bool)
        val_mask = np.zeros(num_paper, dtype=bool)
        test_mask = np.zeros(num_paper, dtype=bool)

        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True

        g.nodes["paper"].data["train_mask"] = th.tensor(train_mask, dtype=th.bool)
        g.nodes["paper"].data["val_mask"] = th.tensor(val_mask, dtype=th.bool)
        g.nodes["paper"].data["test_mask"] = th.tensor(test_mask, dtype=th.bool)

        # 使用 pairwise_distance，为目标节点 paper 构造特征距离矩阵
        if self.save_feature_distance:
            paper_feat = features[:paper_num]
            paper_feat_dist = self._build_feature_distance(paper_feat, do_normalize=False)  
            g.nodes["paper"].data["feature_distance"] = paper_feat_dist

        num_classes = len(np.unique(labels))

        self._g = g
        self._info = {
            "dataset": "Aminer4HERO",
            "num_classes": num_classes,
            "target_ntype": "paper",
            "edge_rate": self.edge_rate,
            "feat_norm": self.feat_norm,
            "node_order": ["paper", "author", "reference"],
            "node_dict": {
                "paper": [0, 6564],
                "author": [6564, 19893],
                "reference": [19893, 55783],
            },
            "canonical_etypes": [
                ("paper", "paper-author", "author"),
                ("author", "author-paper", "paper"),
                ("paper", "paper-ref", "reference"),
                ("reference", "ref-paper", "paper"),
            ],
        }


    def _process_acm(self):
        feat_path = os.path.join(self.raw_data_dir, "node_features.pkl")
        edge_path = os.path.join(self.raw_data_dir, "edges.pkl")
        label_path = os.path.join(self.raw_data_dir, "labels.pkl")

        with open(feat_path, "rb") as f:
            features = pickle.load(f)
        with open(edge_path, "rb") as f:
            edges = pickle.load(f)
        with open(label_path, "rb") as f:
            labels = pickle.load(f)

        # features = preprocess_features(features, norm=self.feat_norm)
        features = preprocess_features(features, norm=True)

        node_types = {
            "paper": np.arange(0, 3025),
            "author": np.arange(3025, 8937),
            "subject": np.arange(8937, 8994),}

        canonical_etypes = [
            ("paper", "paper-author", "author"),
            ("author", "author-paper", "paper"),
            ("paper", "paper-subject", "subject"),
            ("subject", "subject-paper", "paper"),
        ]

        local_id_map = {
            ntype: {gid: i for i, gid in enumerate(gids)}
            for ntype, gids in node_types.items()}

        data_dict = {}
        for i, etype in enumerate(canonical_etypes):
            src_type, _, dst_type = etype
            src_global, dst_global = edges[i].nonzero()
            src_local = [local_id_map[src_type][s] for s in src_global]
            dst_local = [local_id_map[dst_type][d] for d in dst_global]
            data_dict[etype] = (
                th.tensor(src_local, dtype=th.int64),
                th.tensor(dst_local, dtype=th.int64)
            )

        g = dgl.heterograph(data_dict)

        g.nodes["paper"].data["feat"] = features[node_types["paper"]]
        g.nodes["author"].data["feat"] = features[node_types["author"]]
        g.nodes["subject"].data["feat"] = features[node_types["subject"]]

        num_paper = len(node_types["paper"])
        all_label = np.full(num_paper, -1, dtype=np.int64)

        split_names = ["train", "val", "test"]
        for i, split in enumerate(split_names):
            arr = np.array(labels[i])
            node = arr[:, 0].astype(np.int64)
            label = arr[:, 1].astype(np.int64)

            all_label[node] = label

            mask = np.zeros(num_paper, dtype=bool)
            mask[node] = True

            g.nodes["paper"].data[f"{split}_mask"] = th.tensor(mask, dtype=th.bool)

        g.nodes["paper"].data["label"] = th.tensor(all_label, dtype=th.long)

        # 使用 pairwise_distance，为目标节点 paper 构造特征距离矩阵
        if self.save_feature_distance:
            paper_feat_dist = self._build_feature_distance(g.nodes["paper"].data["feat"],do_normalize=True)
            g.nodes["paper"].data["feature_distance"] = paper_feat_dist

        valid_label = all_label[all_label >= 0]
        num_classes = len(np.unique(valid_label))

        self._g = g
        self._info = {
            "dataset": "ACM4HERO",
            "num_classes": num_classes,
            "target_ntype": "paper",
            "edge_rate": self.edge_rate,
            "feat_norm": self.feat_norm,
            "node_order": ["paper", "author", "subject"],
            "node_dict": {
                "paper": [0, 3025],
                "author": [3025, 8937],
                "subject": [8937, 8994],
            },
            "canonical_etypes": canonical_etypes,
        }

    def _process_dblp(self):
        feat_path = os.path.join(self.raw_data_dir, "node_features.pkl")
        edge_path = os.path.join(self.raw_data_dir, "edges.pkl")
        label_path = os.path.join(self.raw_data_dir, "labels.pkl")

        with open(feat_path, "rb") as f:
            features = pickle.load(f)
        with open(edge_path, "rb") as f:
            edges = pickle.load(f)
        with open(label_path, "rb") as f:
            labels = pickle.load(f)

        features = preprocess_features(features, norm=True)
        node_types = {
            "author": np.arange(0, 4057),
            "paper": np.arange(4057, 18385),
            "conference": np.arange(18385, 18405),
        }

        canonical_etypes = [
            ("paper", "paper-author", "author"),       # p-a
            ("author", "author-paper", "paper"),       # a-p
            ("paper", "paper-conference", "conference"),   # p-c
            ("conference", "conference-paper", "paper"),   # c-p
        ]

        local_id_map = {
            ntype: {gid: i for i, gid in enumerate(gids)}
            for ntype, gids in node_types.items()
        }

        data_dict = {}
        for i, etype in enumerate(canonical_etypes):
            src_type, _, dst_type = etype
            src_global, dst_global = edges[i].nonzero()

            src_local = [local_id_map[src_type][s] for s in src_global]
            dst_local = [local_id_map[dst_type][d] for d in dst_global]

            data_dict[etype] = (
                th.tensor(src_local, dtype=th.int64),
                th.tensor(dst_local, dtype=th.int64)
            )

        g = dgl.heterograph(data_dict)
        g.nodes["author"].data["feat"] = features[node_types["author"]]
        g.nodes["paper"].data["feat"] = features[node_types["paper"]]
        g.nodes["conference"].data["feat"] = features[node_types["conference"]]

        num_author = len(node_types["author"])
        all_label = np.full(num_author, -1, dtype=np.int64)

        split_names = ["train", "val", "test"]
        for i, split in enumerate(split_names):
            arr = np.array(labels[i])
            node = arr[:, 0].astype(np.int64)
            label = arr[:, 1].astype(np.int64)

            all_label[node] = label

            mask = np.zeros(num_author, dtype=bool)
            mask[node] = True

            g.nodes["author"].data[f"{split}_mask"] = th.tensor(mask, dtype=th.bool)

        g.nodes["author"].data["label"] = th.tensor(all_label, dtype=th.long)

        if self.save_feature_distance:
            args_node_num = (
                np.array(labels[0]).shape[0]
                + np.array(labels[1]).shape[0]
                + np.array(labels[2]).shape[0]
            )

            dblp_feat_dist = pairwise_distance(features[:args_node_num])
            dblp_feat_dist = F.normalize(dblp_feat_dist, dim=1)
        if self.save_feature_distance:
            paper_feat = g.nodes["author"].data["feat"]
            dblp_feat_dist = pairwise_distance(paper_feat)
            dblp_feat_dist = F.normalize(dblp_feat_dist, dim=1)
            g.nodes["author"].data["feature_distance"] = dblp_feat_dist
       

        valid_label = all_label[all_label >= 0]
        num_classes = len(np.unique(valid_label))

        self._g = g
        self._info = {
            "dataset": "DBLP4HERO",
            "num_classes": num_classes,
            "target_ntype": "author",
            "edge_rate": self.edge_rate,
            "feat_norm": True,
            "node_order": ["author", "paper", "conference"],
            "node_dict": {
                "author": [0, 4057],
                "paper": [4057, 18385],
                "conference": [18385, 18405],
            },
            "canonical_etypes": canonical_etypes,
        }

    def _process_yelp(self):
        meta_path = os.path.join(self.raw_data_dir, "meta_data.pkl")
        label_path = os.path.join(self.raw_data_dir, "labels.pkl")
        edge_path = os.path.join(self.raw_data_dir, "edges.pkl")
        feat_path = os.path.join(self.raw_data_dir, "node_features.pkl")

        with open(meta_path, "rb") as f:
            data = pickle.load(f)
        with open(label_path, "rb") as f:
            labels = pickle.load(f)
        with open(edge_path, "rb") as f:
            edges = pickle.load(f)
        with open(feat_path, "rb") as f:
            features = pickle.load(f)
        features = preprocess_features(features, norm=True)

        node_types = {}
        for t in data["t_info"].keys():
            gids = sorted([i for p, i in data["node2gid"].items() if p.startswith(t)])
            node_types[t] = np.array(gids, dtype=np.int64)

        args_node_num = (
            np.array(labels[0]).shape[0]
            + np.array(labels[1]).shape[0]
            + np.array(labels[2]).shape[0]
        )

        candidate_targets = [t for t, gids in node_types.items() if len(gids) == args_node_num]
        if len(candidate_targets) != 1:
            raise ValueError(
                f"[Yelp4HERO] Cannot uniquely infer target node type. "
                f"args_node_num={args_node_num}, "
                f"candidate_targets={candidate_targets}, "
                f"node_type_sizes={{k: len(v) for k, v in node_types.items()}}"
            )
        target_ntype = candidate_targets[0]

        target_gids = node_types[target_ntype]
        if not np.array_equal(target_gids, np.arange(args_node_num)):
            raise ValueError(
                f"[Yelp4HERO] Target node type '{target_ntype}' is inferred, but its global ids "
                f"are not [0, ..., {args_node_num-1}]. "
                f"This means original load_Yelp logic (features[:args_node_num]) "
                f"cannot be directly aligned with current graph construction."
            )

        canonical_etypes = []
        for rel in edges.keys():
            s, t = rel.split("-")
            canonical_etypes.append((s, f"{s}-{t}", t))

   
        local_id_map = {
            ntype: {gid: i for i, gid in enumerate(gids)}
            for ntype, gids in node_types.items()
        }

        data_dict = {}
        for src_type, etype, dst_type in canonical_etypes:
            rel_key = f"{src_type}-{dst_type}"
            mat = edges[rel_key]

            src_global, dst_global = mat.nonzero()
            src_local = [local_id_map[src_type][s] for s in src_global]
            dst_local = [local_id_map[dst_type][d] for d in dst_global]

            data_dict[(src_type, etype, dst_type)] = (
                th.tensor(src_local, dtype=th.int64),
                th.tensor(dst_local, dtype=th.int64)
            )

        num_nodes_dict = {ntype: len(gids) for ntype, gids in node_types.items()}
        g = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict)


        for ntype, gids in node_types.items():
            g.nodes[ntype].data["feat"] = features[gids]


        num_target = len(node_types[target_ntype])
        all_label = np.full(num_target, -1, dtype=np.int64)

        split_names = ["train", "val", "test"]
        for i, split in enumerate(split_names):
            arr = np.array(labels[i])
            node = arr[:, 0].astype(np.int64)
            label = arr[:, 1].astype(np.int64)

            all_label[node] = label

            mask = np.zeros(num_target, dtype=bool)
            mask[node] = True
            g.nodes[target_ntype].data[f"{split}_mask"] = th.tensor(mask, dtype=th.bool)

        g.nodes[target_ntype].data["label"] = th.tensor(all_label, dtype=th.long)

     
        if self.save_feature_distance:
            yelp_feat_dist = pairwise_distance(features[:args_node_num])
            yelp_feat_dist = F.normalize(yelp_feat_dist, dim=1)

            total = yelp_feat_dist.numel()
            k = int(total * self.edge_rate)
            if k > 0:
                kthvalue = th.kthvalue(yelp_feat_dist.reshape(-1), k).values
                mask = (yelp_feat_dist > kthvalue).float()
                yelp_feat_dist = yelp_feat_dist * mask

            g.nodes[target_ntype].data["feature_distance"] = yelp_feat_dist

        valid_label = all_label[all_label >= 0]
        num_classes = len(np.unique(valid_label))

        self._g = g
        self._info = {
            "dataset": "Yelp4HERO",
            "num_classes": num_classes,
            "target_ntype": target_ntype,
            "edge_rate": self.edge_rate,
            "feat_norm": True,
            "node_order": list(node_types.keys()),
            "node_dict": {k: [int(v[0]), int(v[-1]) + 1] for k, v in node_types.items()},
            "canonical_etypes": canonical_etypes,
        }

    def save(self):
        os.makedirs(self.dataset_dir, exist_ok=True)
        save_graphs(self.g_path, [self._g])

        with open(self.info_path, "wb") as f:
            pickle.dump(self._info, f)

        if self.verbose:
            print(f"[{self.dataset_name}] graph saved to {self.g_path}")
            print(f"[{self.dataset_name}] info saved to {self.info_path}")
  

    def load(self):
        graphs, _ = load_graphs(self.g_path)
        self._g = graphs[0]

        if os.path.exists(self.info_path):
            with open(self.info_path, "rb") as f:
                self._info = pickle.load(f)
        else:
            label = self._g.nodes["paper"].data["label"]
            valid = label[label >= 0] if (label >= 0).any() else label
            self._info = {
                "dataset": self.dataset_name,
                "num_classes": len(th.unique(valid)),
                "target_ntype": "paper",
            }

        if self.verbose:
            print(f"[{self.dataset_name}] graph loaded from {self.g_path}")

    def has_cache(self):
        return os.path.exists(self.g_path)

    def __getitem__(self, idx):
        assert idx == 0, "This dataset has only one graph"
        return self._g

    def __len__(self):
        return 1

    @property
    def num_classes(self):
        return self._info["num_classes"]

    @property
    def target_ntype(self):
        return self._info["target_ntype"]

    @property
    def graph_path(self):
        return self.g_path

    @property
    def info(self):
        return self._info

    def _raw_files_exist(self):
        if self.dataset_name == "ACM4HERO":
            required = [
                os.path.join(self.raw_data_dir, "node_features.pkl"),
                os.path.join(self.raw_data_dir, "edges.pkl"),
                os.path.join(self.raw_data_dir, "labels.pkl"),
            ]
        elif self.dataset_name == "DBLP4HERO":
            required = [
                os.path.join(self.raw_data_dir, "node_features.pkl"),
                os.path.join(self.raw_data_dir, "edges.pkl"),
                os.path.join(self.raw_data_dir, "labels.pkl"),
            ]

        elif self.dataset_name == "Aminer4HERO":
            required = [
                os.path.join(self.raw_data_dir, "features_0.npy"),
                os.path.join(self.raw_data_dir, "features_1.npy"),
                os.path.join(self.raw_data_dir, "features_2.npy"),
                os.path.join(self.raw_data_dir, "labels.npy"),
                os.path.join(self.raw_data_dir, "pa.txt"),
                os.path.join(self.raw_data_dir, "pr.txt"),
    ]

        elif self.dataset_name == "Yelp4HERO":
            required = [
                os.path.join(self.raw_data_dir, "meta_data.pkl"),
                os.path.join(self.raw_data_dir, "labels.pkl"),
                os.path.join(self.raw_data_dir, "edges.pkl"),
                os.path.join(self.raw_data_dir, "node_features.pkl"),
            ]
        else:
            return False

        return all(os.path.exists(x) for x in required)

    def _flatten_raw_dir_if_needed(self):
        if self._raw_files_exist():
            return

        items = os.listdir(self.raw_data_dir)
        subdirs = [
            os.path.join(self.raw_data_dir, x)
            for x in items
            if os.path.isdir(os.path.join(self.raw_data_dir, x))
        ]

        if len(subdirs) != 1:
            return

        nested_dir = subdirs[0]
        for fname in os.listdir(nested_dir):
            shutil.move(
                os.path.join(nested_dir, fname),
                os.path.join(self.raw_data_dir, fname)
            )
        shutil.rmtree(nested_dir)