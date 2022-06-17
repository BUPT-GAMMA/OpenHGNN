import torch
import torch as th
import numpy as np
import dgl
import os
from dgl.data import DGLBuiltinDataset
from dgl.data.utils import load_graphs, save_graphs, save_info, load_info
import copy
from dgl import transforms as T

__all__ = ['GATNEDataset', 'Amazon4GATNEDataset', 'Twitter4GATNEDataset', 'Youtube4GATNEDataset']


class GATNEDataset(DGLBuiltinDataset):
    r"""GATNE Dataset.
    The network datasets "Amazon", "Twitter" and "YouTube" from the
    `"Representation Learning for Attributed Multiplex Heterogeneous Network"
    <https://arxiv.org/abs/1905.01669>`_ paper.
    Dataset statistics:
    Dataset  nodes     edges n-types e-types
     Amazon 10,166   148,865       1       2
    Youtube  2,000 1,310,617       1       5
    Twitter 10,000   331,899       1       4
    Parameters
    ----------
    name : str
        Name of the dataset. Supported dataset names are 'amazon4GATNE', 'youtube4GATNE' and 'twitter4GATNE'.
    add_reverse : bool
        Whether to add reverse edges. Default: True
    raw_dir : str
        Specifying the directory that will store the
        downloaded data or the directory that
        already stores the input data.
        Default: ~/.dgl/
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose : bool
        Whether to print out progress information. Default: False
    transform : callable, optional
        A transform that takes in a :class:`~dgl.DGLGraph` object and returns
        a transformed version. The :class:`~dgl.DGLGraph` object will be
        transformed before every access.
    Examples
    --------
    >>> dataset = GATNEDataset(name='amazon4GATNE')
    >>> graph = dataset[0]
    """

    def __init__(self, name, add_reverse=True, raw_dir=None, force_reload=False, verbose=False,
                 transform=None):
        assert name in ['amazon4GATNE', 'twitter4GATNE', 'youtube4GATNE']
        if name == 'amazon4GATNE':
            etypes = ['co-viewing', 'co-purchasing']
            ntype = 'product'
            target_link = [('product', 'co-viewing', 'product'), ('product', 'co-purchasing', 'product')]
            meta_paths_dict = {}
        elif name == 'youtube4GATNE':
            etypes = ['contact', 'shared friends', 'shared subscription', 'shared subscriber',
                      'shared favorite videos']
            ntype = 'user'
            target_link = [('user', 'contact', 'user'), ('user', 'shared friends', 'user'),
                           ('user', 'shared subscription', 'user'), ('user', 'shared subscriber', 'user'),
                           ('user', 'shared favorite videos', 'user')]
            meta_paths_dict = {}
        elif name == 'twitter4GATNE':
            etypes = ['re-tweet', 'reply', 'mention', 'friendship']
            ntype = 'user'
            target_link = [('user', 're-tweet', 'user'), ('user', 'reply', 'user'),
                           ('user', 'mention', 'user'), ('user', 'friendship', 'user')]
            meta_paths_dict = {}
        else:
            raise ValueError('Unsupported dataset name {}'.format(name))

        self._add_reverse = add_reverse
        self._target_link = target_link
        self._etypes = etypes
        self._ntype = ntype
        self._meta_paths_dict = meta_paths_dict

        super(GATNEDataset, self).__init__(
            name,
            url='https://s3.cn-north-1.amazonaws.com.cn/dgl-data/dataset/openhgnn/{}.zip'.format(name),
            raw_dir=raw_dir,
            force_reload=force_reload, verbose=verbose, transform=transform)

    def process(self):
        # 读取数据
        training_data_by_type, training_nodes = self.load_training_data('train.txt')
        valid_true_data_by_type, valid_false_data_by_type, valid_nodes = self.load_testing_data('valid.txt')
        testing_true_data_by_type, testing_false_data_by_type, testing_nodes = self.load_testing_data('test.txt')
        all_nodes = list(set(training_nodes + valid_nodes + testing_nodes))

        # 合并training, valid, testing三个边集
        true_data_by_type = copy.deepcopy(training_data_by_type)
        for etype in valid_true_data_by_type:
            if true_data_by_type.get(etype):
                true_data_by_type[etype].extend(valid_true_data_by_type[etype])  # 原地修改true_data_by_type对象
            else:
                true_data_by_type[etype] = valid_true_data_by_type[etype]

        for etype in testing_true_data_by_type:
            if true_data_by_type.get(etype):
                true_data_by_type[etype].extend(testing_true_data_by_type[etype])
            else:
                true_data_by_type[etype] = testing_true_data_by_type[etype]

        # 给节点编号
        i = 0
        vocab = {}
        for node in all_nodes:
            vocab[node] = i
            i = i + 1

        # 处理负边
        neg_val_edges = {}
        for etype in valid_false_data_by_type:
            tmp_data = valid_false_data_by_type[etype]
            src = []
            dst = []

            for edge in tmp_data:
                src.extend([vocab[edge[0]]])
                dst.extend([vocab[edge[1]]])
            neg_val_edges[(self._ntype, etype, self._ntype)] = (torch.from_numpy(np.array(src)),
                                                                torch.from_numpy(np.array(dst)))
        self._neg_val_edges = neg_val_edges

        neg_test_edges = {}
        for etype in testing_false_data_by_type:
            tmp_data = testing_false_data_by_type[etype]
            src = []
            dst = []

            for edge in tmp_data:
                src.extend([vocab[edge[0]]])
                dst.extend([vocab[edge[1]]])
            neg_test_edges[(self._ntype, etype, self._ntype)] = (torch.from_numpy(np.array(src)),
                                                                 torch.from_numpy(np.array(dst)))
        self._neg_test_edges = neg_test_edges

        # 生成异质图
        g = self.get_graph(true_data_by_type, vocab)

        # 增加训练集，验证集，测试集标记
        for etype in g.etypes:
            num1 = len(true_data_by_type[etype])  # 总边数
            if etype in training_data_by_type:
                num2 = len(training_data_by_type[etype])  # 训练集边数
            else:
                num2 = 0
            if etype in valid_true_data_by_type:
                num3 = len(valid_true_data_by_type[etype])  # 验证集边数
            else:
                num3 = 0

            train_mask = th.zeros(num1)
            val_mask = th.zeros(num1)
            test_mask = th.zeros(num1)

            train_mask[:num2] = 1
            val_mask[num2:num2 + num3] = 1
            test_mask[num2 + num3:] = 1

            g.edges[etype].data['train_mask'] = train_mask
            g.edges[etype].data['val_mask'] = val_mask
            g.edges[etype].data['test_mask'] = test_mask

        if self._add_reverse:
            transform = T.AddReverse()
            g = transform(g)
        self._g = g

    def has_cache(self):
        graph_path = os.path.join(self.save_path, 'graph.bin')
        return os.path.exists(graph_path)

    def save(self):
        graph_path = os.path.join(self.save_path, 'graph.bin')
        info_path = os.path.join(self.save_path, 'info.pkl')
        save_graphs(graph_path, self._g)
        save_info(info_path, {'neg_val_edges': self.neg_val_edges,
                              'neg_test_edges': self.neg_test_edges})

    def load(self):
        graph_path = os.path.join(self.save_path, 'graph.bin')
        info_path = os.path.join(self.save_path, 'info.pkl')
        gs, _ = load_graphs(graph_path)
        self._g = gs[0]
        info = load_info(info_path)
        self._neg_val_edges = info['neg_val_edges']
        self._neg_test_edges = info['neg_test_edges']

    def load_training_data(self, f_name):
        f_path = os.path.join(self.raw_path, f_name)
        if self.verbose:
            print("We are loading data from:", f_path)

        edge_data_by_type = dict()
        all_nodes = list()

        with open(f_path, "r") as f:
            for line in f:
                words = line[:-1].split(" ")  # line[-1] == '\n'
                etype = self._etypes[int(words[0]) - 1]
                if etype not in edge_data_by_type:  # 同一类型的边放在一个列表里
                    edge_data_by_type[etype] = list()
                x, y = words[1], words[2]
                edge_data_by_type[etype].append((x, y))
                # if self.name == 'amazon4GATNE' or self.name == 'youtube4GATNE':  # bidirectional
                #     edge_data_by_type[etype].append((y, x))
                all_nodes.append(x)
                all_nodes.append(y)
        all_nodes = list(set(all_nodes))

        if self.verbose:
            print("Total training nodes: " + str(len(all_nodes)))
        return edge_data_by_type, all_nodes

    def load_testing_data(self, f_name):
        f_path = os.path.join(self.raw_path, f_name)
        if self.verbose:
            print("We are loading data from:", f_path)

        true_edge_data_by_type = dict()
        false_edge_data_by_type = dict()
        all_nodes = list()

        with open(f_path, "r") as f:
            for line in f:
                words = line[:-1].split(" ")
                etype = self._etypes[int(words[0]) - 1]
                x, y = words[1], words[2]
                if int(words[3]) == 1:  # 实边
                    if etype not in true_edge_data_by_type:
                        true_edge_data_by_type[etype] = list()  # 每种类型的边构成一个列表
                    true_edge_data_by_type[etype].append((x, y))  # 每条边由一个二元组（src, des）唯一确定
                    # if self.name == 'amazon4GATNE' or self.name == 'youtube4GATNE':  # bidirectional
                    #     true_edge_data_by_type[etype].append((y, x))
                else:  # 虚边
                    if etype not in false_edge_data_by_type:
                        false_edge_data_by_type[etype] = list()
                    false_edge_data_by_type[etype].append((x, y))
                    # if self.name == 'amazon4GATNE' or self.name == 'youtube4GATNE':  # bidirectional
                    #     false_edge_data_by_type[etype].append((y, x))
                all_nodes.append(x)  # x加入节点集
                all_nodes.append(y)
        all_nodes = list(set(all_nodes))  # 去重

        if self.verbose:
            print("Total {} nodes: ".format(f_name[:-4]) + str(len(all_nodes)))
        return true_edge_data_by_type, false_edge_data_by_type, all_nodes

    def get_graph(self, network_data, vocab):
        """ Build graph, treat all nodes as the same type
        Parameters
        ----------
        network_data: a dict
            keys describing the edge types, values representing edges
        vocab: a dict
            mapping node IDs to node indices

        Output
        ------
        DGLHeteroGraph
            a heterogenous graph, with one node type and different edge types
        """
        node_type = self._ntype  # '_N' can be replaced by an arbitrary name
        data_dict = dict()
        num_nodes_dict = {node_type: len(vocab)}

        for edge_type in network_data:
            tmp_data = network_data[edge_type]
            src = []
            dst = []

            for edge in tmp_data:
                src.extend([vocab[edge[0]]])
                dst.extend([vocab[edge[1]]])
            data_dict[(node_type, edge_type, node_type)] = (src, dst)

        graph = dgl.heterograph(data_dict, num_nodes_dict)
        return graph

    @property
    def meta_paths_dict(self):
        return self._meta_paths_dict

    @property
    def neg_val_edges(self):
        return self._neg_val_edges

    @property
    def neg_test_edges(self):
        return self._neg_test_edges

    @property
    def target_link(self):
        return self._target_link

    @property
    def target_link_r(self):
        return self._target_link_r

    def __getitem__(self, idx):
        assert idx == 0
        return self._g

    def __len__(self):
        return 1


class Amazon4GATNEDataset(GATNEDataset):
    def __init__(self, raw_dir=None, force_reload=False, verbose=False, transform=None):
        name = 'amazon4GATNE'
        super(Amazon4GATNEDataset, self).__init__(name, raw_dir=raw_dir, force_reload=force_reload, verbose=verbose,
                                                  transform=transform)


class Twitter4GATNEDataset(GATNEDataset):
    def __init__(self, raw_dir=None, force_reload=False, verbose=False, transform=None):
        name = 'twitter4GATNE'
        super(Twitter4GATNEDataset, self).__init__(name, raw_dir=raw_dir, force_reload=force_reload, verbose=verbose,
                                                   transform=transform)


class Youtube4GATNEDataset(GATNEDataset):
    def __init__(self, raw_dir=None, force_reload=False, verbose=False, transform=None):
        name = 'youtube4GATNE'
        super(Youtube4GATNEDataset, self).__init__(name, raw_dir=raw_dir, force_reload=force_reload, verbose=verbose,
                                                   transform=transform)
Twitter4GATNEDataset()