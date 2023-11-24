import os

import dgl
import torch as th
from dgl.data import extract_archive, download

from . import BaseDataset, register_dataset


@register_dataset("oag_dataset")
class OAGDataset(BaseDataset):
    def __init__(self, *args, **kwargs):
        super(OAGDataset, self).__init__(*args, **kwargs)
        self.name = "oag_cs"
        self.valid_idx = None
        self.test_idx = None
        self.train_idx = None
        self.dims = None
        self.g = None
        self.category = None
        self.num_classes = None
        self.has_feature = True
        self.data_path = "./openhgnn/dataset/data/oag_cs.tgz"
        self.raw_dir = "./openhgnn/dataset/data"
        self.g_path = "./openhgnn/dataset/data/oag_cs/oag_cs.bin"
        self.url = "https://s3.cn-north-1.amazonaws.com.cn/dgl-data/dataset/oag_cs.tgz"
        if not self.has_cache():
            self.download()
        self.load_graph_from_disk(self.g_path)
        self.meta_paths_dict = {
            "P-A": [("paper", "paper-author", "author")],
            "A-P": [("author", "author-paper", "paper")],
            "V-A": [
                ("venue", "venue-paper", "paper"),
                ("paper", "paper-author", "author"),
            ],
            "A-V": [
                ("author", "author-paper", "paper"),
                ("paper", "paper-venue", "venue"),
            ],
        }

    def load_graph_from_disk(self, file_path):
        glist, dims = dgl.load_graphs(file_path)
        self.g = glist[0]
        self.dims = dims

    def get_labels(self, task_type, node_type):
        assert task_type in ["L1", "L2"]
        return self.g.ndata[task_type].pop(node_type)

    def get_split(self, node_type, device="cpu"):
        train_mask = self.g.nodes[node_type].data["train_mask"]
        test_mask = self.g.nodes[node_type].data["test_mask"]
        train_idx = th.nonzero(train_mask, as_tuple=False).squeeze()
        test_idx = th.nonzero(test_mask, as_tuple=False).squeeze()
        valid_idx = train_idx
        self.train_idx = train_idx.to(device)
        self.test_idx = test_idx.to(device)
        self.valid_idx = valid_idx.to(device)
        return self.train_idx, self.valid_idx, self.test_idx

    def get_feature(
            self,
    ):
        return self.g.ndata.pop("feat")

    def to(self, device):
        self.g = self.g.to(device)
        return self

    def download(self):
        if os.path.exists(self.data_path):
            pass
        else:
            file_path = os.path.join(self.raw_dir)
            print("Downloading", self.url, "to", file_path)
            download(self.url, path=file_path)
        extract_archive(self.data_path, os.path.join(self.raw_dir))

    def has_cache(self):
        return os.path.exists(self.g_path)
