import torch
import torch as th
import os
from dgl.data import DGLBuiltinDataset
from dgl.data.utils import load_graphs, save_graphs
import csv
from tqdm import tqdm
import numpy as np
import pickle as pkl
import dgl

__all__ = ['AliRCDDataset', 'AliRCDSmallDataset', 'AliRCDSession1Dataset', 'AliRCDSession2Dataset','AliICDMDataset']


class AliRCDDataset(DGLBuiltinDataset):
    r"""AliRCD(Alibaba Risk Commodity Detection) dataset is extracted from real-world risk scenarios at Alibaba.
    This dataset is for ICDM 2022 competition. Detailed Information is here:
    https://tianchi.aliyun.com/competition/entrance/531976/information. When we create the dgl graph, we rearrange
    the node ids from the original files. The map from original node id to dgl node id can be retrieved from property
    item_map and the map from dgl node id to original node id can be retrieved from property rev_item_map.

    Parameters
    ----------
    session : str
        'small', 'session1' or 'session2'. The small one is only used for debug and helps with understanding
        the data format.
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
    """

    def __init__(self, session, raw_dir=None, force_reload=False,
                 verbose=False, transform=None):
        name = 'AliRCD_{}'.format(session)
        self.session = session
        self.load_labels = not session == 'session2'

        super(AliRCDDataset, self).__init__(
            name,
            url='https://s3.cn-north-1.amazonaws.com.cn/dgl-data/dataset/openhgnn/{}.zip'.format(name),
            raw_dir=raw_dir,
            force_reload=force_reload, verbose=verbose, transform=transform)

    def process(self):
        if self.session == 'session1':
            edge_size = 157814864
            node_size = 13806619
        elif self.session == 'session2':
            edge_size = 120691444
            node_size = 10284026
        else:  # for debug
            edge_size = 100
            node_size = 100

        # load node info
        nodes_info = self._get_node_atts(node_size)

        # item map and reversed item map
        self._item_map = nodes_info['maps']['item']
        self._rev_item_map = {}
        for k, v in self._item_map.items():
            self._rev_item_map[v] = k

        # load edges
        g = self._format_dgl_edges(nodes_info, edge_size)

        # add attrs
        for ntype, embedding_dict in nodes_info['embeds'].items():
            dim = embedding_dict[0].shape[0]
            g.nodes[ntype].data['h'] = torch.rand(g.num_nodes(ntype), dim)
            for nid, embedding in tqdm(embedding_dict.items()):
                g.nodes[ntype].data['h'][nid] = torch.from_numpy(embedding)
        # load label
        num_nodes = g.num_nodes(self.category)
        if self.load_labels:
            labels_path = os.path.join(self.save_path, '{}_train_labels.csv'.format(self.name))
            labels = th.tensor([float('nan')] * g.num_nodes(self.category))
            with open(labels_path, 'r') as f:
                csvreader = csv.reader(f)
                for row in csvreader:
                    orig_id = int(row[0])
                    new_id = self._item_map.get(orig_id)
                    if new_id is not None:
                        labels[new_id] = int(row[1])
            label_mask = ~th.isnan(labels)
            label_idx = th.nonzero(label_mask, as_tuple=False).squeeze()
            g.nodes[self.category].data['label'] = labels.type(th.int64)

            # label_idx = np.random.permutation(np.array(label_idx))  # shuffle the label index
            split_ratio = [0.8, 0.2]
            num_labels = len(label_idx)
            train_mask = th.zeros(num_nodes).bool()
            train_mask[label_idx[0: int(split_ratio[0] * num_labels)]] = True
            val_mask = th.zeros(num_nodes).bool()
            val_mask[
                label_idx[int(split_ratio[0] * num_labels): int((split_ratio[0] + split_ratio[1]) * num_labels)]] = True
            g.nodes[self.category].data['train_mask'] = train_mask
            g.nodes[self.category].data['val_mask'] = val_mask

        # load test_idx
        test_idx_path = os.path.join(self.save_path, '{}_test_ids.csv'.format(self.name))
        test_mask = th.zeros(num_nodes).bool()
        with open(test_idx_path, 'r') as f:
            csvreader = csv.reader(f)
            for row in csvreader:
                orig_id = int(row[0])
                new_id = self._item_map.get(orig_id)
                if new_id is not None:
                    test_mask[new_id] = True
        g.nodes[self.category].data['test_mask'] = test_mask

        if self.session == "ICDM" :
            test_labels_path = os.path.join(self.save_path, '{}_test_labels.csv'.format(self.name))
            with open(test_labels_path, 'r') as f:
                csvreader = csv.reader(f)
                for row in csvreader:
                    line = row[0].split('\t')
                    test_label = int(float(line[1]))
                    test_id = int(line[0])
                    new_id = self._item_map.get(test_id)
                    if new_id is not None:
                        g.nodes[self.category].data['label'][new_id] = test_label

        self._g = g
        if self.verbose:
            print(self._g)
            print('finish loading dataset')

    def _get_node_atts(self, node_size=100):
        node_file = os.path.join(self.save_path, '{}_nodes.csv'.format(self.name))
        node_maps = {}
        node_embeds = {}
        count = 0
        count2 = 0
        node_counts = node_size
        process = tqdm(total=node_counts)
        with open(node_file, 'r') as rf:
            while True:
                line = rf.readline()
                if line is None or len(line) == 0:
                    break
                info = line.strip().split(",")

                node_id = int(info[0])
                node_type = info[1].strip()
                node_maps.setdefault(node_type, {})
                node_id_v2 = len(node_maps[node_type])
                node_maps[node_type][node_id] = node_id_v2
                if node_type == 'item' and len(info[2]) == 0:
                    node_embeds.setdefault(node_type, {})
                    node_embeds[node_type][node_id_v2] = np.zeros(128, dtype=np.float32)
                    count2 += 1
                elif len(info) == 3 and len(info[2]) > 0:
                    node_embeds.setdefault(node_type, {})
                    node_embeds[node_type][node_id_v2] = np.array([x for x in info[2].split(":")], dtype=np.float32)
                count += 1
                if count % 100000 == 0:
                    process.update(100000)
        process.close()

        print('lack of features:', count2, count, len(node_maps['item']))
        print('node_types', node_maps.keys())
        for node_type in node_maps:
            print('node_type', node_type, len(node_maps[node_type]))
        print(len(node_embeds['item']))

        nodes_dict = {'maps': node_maps, 'embeds': node_embeds}
        return nodes_dict

    def _format_dgl_edges(self, node_info, edge_size=100):
        node_maps = node_info['maps']
        edges = {}
        edges_hg = {}

        process = tqdm(total=edge_size)
        count = 0
        edge_file = os.path.join(self.save_path, '{}_edges.csv'.format(self.name))
        with open(edge_file, 'r') as rf:
            while True:
                line = rf.readline()
                if line is None or len(line) == 0:
                    break
                line_info = line.strip().split(",")
                source_id, dest_id, source_type, dest_type, edge_type = line_info
                source_id = node_maps[source_type][int(source_id)]
                dest_id = node_maps[dest_type][int(dest_id)]
                edges.setdefault(edge_type, {})
                edges[edge_type].setdefault('source', []).append(source_id)
                edges[edge_type].setdefault('dest', []).append(dest_id)
                edges[edge_type].setdefault('source_type', source_type)
                edges[edge_type].setdefault('dest_type', dest_type)
                count += 1
                if count % 100000 == 0:
                    process.update(100000)
        process.close()

        for edge_type in edges:
            source_type = edges[edge_type]['source_type']
            dest_type = edges[edge_type]['dest_type']
            source = edges[edge_type]['source']
            dest = edges[edge_type]['dest']
            edges_hg[(source_type, edge_type, dest_type)] = list(zip(source, dest))

        hg = dgl.heterograph(edges_hg)
        return hg

    def has_cache(self):
        graph_path = os.path.join(self.save_path, 'graph.bin')
        return os.path.exists(graph_path)

    def save(self):
        graph_path = os.path.join(self.save_path, 'graph.bin')
        save_graphs(graph_path, self._g)
        pkl.dump(self.item_map, open(os.path.join(self.save_path, 'item_map.pkl'), 'wb'))
        pkl.dump(self.rev_item_map, open(os.path.join(self.save_path, 'rev_item_map.pkl'), 'wb'))

    def load(self):
        graph_path = os.path.join(self.save_path, 'graph.bin')
        gs, _ = load_graphs(graph_path)
        self._item_map = pkl.load(open(os.path.join(self.save_path, 'item_map.pkl'), 'rb'))
        self._rev_item_map = pkl.load(open(os.path.join(self.save_path, 'rev_item_map.pkl'), 'rb'))
        self._g = gs[0]
        if self.verbose:
            print(self._g)

    @property
    def item_map(self):
        return self._item_map

    @property
    def rev_item_map(self):
        return self._rev_item_map

    @property
    def category(self):
        return 'item'

    @property
    def target_ntype(self):
        return 'item'

    @property
    def num_classes(self):
        return 2

    def __getitem__(self, idx):
        assert idx == 0
        return self._g

    def __len__(self):
        return 1

    @property
    def meta_paths_dict(self):
        return {'IBI': [('item', 'A', 'b'),
                        ('b', 'A_1', 'item')],
                'IFAFI': [('item', 'B_1', 'f'),
                          ('f', 'G', 'a'),
                          ('a', 'G_1', 'f'),
                          ('f', 'B', 'item')],
                'IFCFI': [('item', 'B_1', 'f'),
                          ('f', 'D', 'c'),
                          ('c', 'D_1', 'f'),
                          ('f', 'B', 'item')],
                'IFDFI': [('item', 'B_1', 'f'),
                          ('f', 'C', 'd'),
                          ('d', 'C_1', 'f'),
                          ('f', 'B', 'item')],
                'IFEFI': [('item', 'B_1', 'f'),
                          ('f', 'F', 'e'),
                          ('e', 'F_1', 'f'),
                          ('f', 'B', 'item')],
                }


class AliRCDSmallDataset(AliRCDDataset):
    def __init__(self, raw_dir=None, force_reload=False, verbose=False, transform=None):
        session = 'small'
        super(AliRCDSmallDataset, self).__init__(session, raw_dir=raw_dir, force_reload=force_reload, verbose=verbose,
                                                 transform=transform)


class AliRCDSession1Dataset(AliRCDDataset):
    def __init__(self, raw_dir=None, force_reload=False, verbose=False, transform=None):
        session = 'session1'
        super(AliRCDSession1Dataset, self).__init__(session, raw_dir=raw_dir, force_reload=force_reload,
                                                    verbose=verbose,
                                                    transform=transform)


class AliRCDSession2Dataset(AliRCDDataset):
    def __init__(self, raw_dir=None, force_reload=False, verbose=False, transform=None):
        session = 'session2'
        super(AliRCDSession2Dataset, self).__init__(session, raw_dir=raw_dir, force_reload=force_reload,
                                                    verbose=verbose,
                                                    transform=transform)
        
class AliICDMDataset(AliRCDDataset):
    def __init__(self, raw_dir=None, force_reload=False, verbose=False, transform=None):
        session = 'ICDM'
        super(AliICDMDataset, self).__init__(session, raw_dir=raw_dir, force_reload=force_reload,
                                                    verbose=verbose,
                                                    transform=transform)