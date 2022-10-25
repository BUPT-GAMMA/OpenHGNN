
import pickle
import os
from openhgnn.dataset import register_dataset, BaseDataset
import numpy as np
import copy
from torch.utils.data import Dataset
from scipy.sparse import vstack as s_vstack
from scipy.sparse import csr_matrix
from dgl.data.utils import download, extract_archive


class HgraphDataset(Dataset):

    _prefix = 'https://s3.cn-north-1.amazonaws.com.cn/dgl-data/'
    _urls = {

    }

    def __init__(self, name, is_train=True, raw_dir = None, force_reload = False, verbose = True):
        assert name in ['GPS','drug','MovieLens','wordnet']

        self.data_path = './openhgnn/dataset/{}4DHNE.zip'.format(name.lower())
        raw_dir = './openhgnn/dataset/'
        url = self._prefix + 'dataset/openhgnn/{}4DHNE.zip'.format(name.lower())

        super().__init__()
        self.name=name
        self.url=url
        self.raw_dir=raw_dir
        self.force_reload=force_reload
        self.verbose=verbose
        self.train_file = 'train_data.npz'
        self.test_file = 'test_data.npz'
        self.num_neg_samples = 1
        self.pair_radio = 0.9
        self.sparse_input = True
        self.train_dir = raw_dir + '{}'.format(self.name)
        self.is_train = is_train
        self.download()
        self.process()


    def download(self):
        # download raw data to local disk
        # path to store the file
        if os.path.exists(self.data_path):  # pragma: no cover
           pass
        else:
            file_path = os.path.join(self.raw_dir)
            # download file
            download(self.url, path=file_path)
        extract_archive(self.data_path, os.path.join(self.raw_dir, self.name))

    def process(self):
        if self.is_train:
            self.process_train()
            if not os.path.exists(f"./embeddings_{self.name}.pkl"):
                self.embeddings = generate_embeddings(
                    self.edges, self.nums_type)
                with open(f"./embeddings_{self.name}.pkl", 'wb') as f:
                    pickle.dump(self.embeddings, f)
            else:
                with open(f"./embeddings_{self.name}.pkl", 'rb') as f:
                    self.embeddings = pickle.load(f)

        else:
            self.process_val()
            with open(f"./embeddings_{self.name}.pkl", 'rb') as f:
                self.embeddings = pickle.load(f)

    def process_val(self):
        data = np.load(os.path.join(self.train_dir, self.test_file), allow_pickle=True)
        self.edges = data['test_data']
        self.nums_type = data['nums_type']
        self.node_cluster = data['node_cluster'] if 'node_cluster' in data else None
        self.edge_set = set(map(tuple, self.edges))

    def process_train(self):
        data = np.load(os.path.join(self.train_dir, self.train_file), allow_pickle=True)
        self.edges = data['train_data']
        self.nums_type = data['nums_type']
        self.labels = data['labels'] if 'labels' in data else None
        self.idx_label = data['idx_label'] if 'idx_label' in data else None
        self.label_set = data['label_name'] if 'label_name' in data else None
        self.edge_set = set(map(tuple, self.edges))

    def __getitem__(self, idx):
        pos_data = [self.edges[idx]]
        neg_data = []
        n_neg = 0
        while(n_neg < self.num_neg_samples):
            # warning !!! we need deepcopy to copy list
            index = copy.deepcopy(self.edges[idx])
            mode = np.random.rand()
            if mode < self.pair_radio:
                type_ = np.random.randint(3)
                # Randomly select a mode
                node = np.random.randint(self.nums_type[type_])
                index[type_] = node
            else:
                # Randomly select two types
                types_ = np.random.choice(3, 2, replace=False)
                node_1 = np.random.randint(self.nums_type[types_[0]])
                node_2 = np.random.randint(self.nums_type[types_[1]])
                index[types_[0]] = node_1
                index[types_[1]] = node_2
            if tuple(index) in self.edge_set:
                continue
            n_neg += 1
            neg_data.append(index)
        data = np.vstack(pos_data+neg_data)
        if len(neg_data) > 0:
            nums = len(data)
            labels = np.zeros(nums)
            labels[0] = 1
        else:
            labels = np.ones(1)
        batch_e = embedding_lookup(self.embeddings, data)
        return batch_e, labels

    def __len__(self):
        return len(self.edges)


def embedding_lookup(embeddings, index, sparse_input=True):
    if sparse_input:
        return [embeddings[i][index[:, i], :].todense() for i in range(3)]
    else:
        return [embeddings[i][index[:, i], :] for i in range(3)]


def generate_H(edge, nums_type):
    nums_examples = len(edge)
    H = [csr_matrix((np.ones(nums_examples), (edge[:, i], range(
        nums_examples))), shape=(nums_type[i], nums_examples)) for i in range(3)]
    return H


def generate_embeddings(edge, nums_type):
    r"""
    Args:
        edge (_type_): Number of edges
        nums_type (_type_): Number of node types
    Returns:
        _type_: _description_
    """
    H = generate_H(edge, nums_type)
    embeddings = [H[i].dot(s_vstack([H[j] for j in range(3) if j != i]).T).astype(
        'float') for i in range(3)]
    # 0-1 scaling
    for i in range(3):
        col_max = np.array(embeddings[i].max(0).todense()).flatten()
        _, col_index = embeddings[i].nonzero()
        embeddings[i].data /= col_max[col_index]

    return embeddings


@register_dataset("hypergraph_dataset")
class HGraphDataset(BaseDataset):
    def get_data(self, name, is_train):
        return HgraphDataset(name, is_train)
