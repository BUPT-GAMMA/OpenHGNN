import dgl
import dgl.function as fn
from dgl.data import DGLDataset
import torch as th
from . import load_acm, load_acm_raw
from openhgnn.dataset import BaseDataset, register_dataset
from dgl.data.rdf import AIFBDataset, MUTAGDataset, BGSDataset, AMDataset
from dgl.data.utils import load_graphs, save_graphs
from ogb.nodeproppred import DglNodePropPredDataset
from . import AcademicDataset


@register_dataset('node_classification')
class NodeClassificationDataset(BaseDataset):
    """
    metric: Accuracy, multi-label f1 or multi-class f1. Default: `accuracy`
    """

    def __init__(self):
        super(NodeClassificationDataset, self).__init__()
        self.has_feature = False

    def get_labels(self):
        raise NotImplemented

    def get_idx(self, ):
        raise NotImplemented


@register_dataset('rdf_node_classification')
class RDF_NodeCLassification(NodeClassificationDataset):
    def __init__(self, dataset_name):
        super(RDF_NodeCLassification, self).__init__()
        self.g, self.category, self.num_classes = self.load_RDF_dgl(dataset_name)
        self.has_feature = False

    def load_RDF_dgl(self, dataset):
        # load graph data
        if dataset == 'aifb':
            kg_dataset = AIFBDataset()
        elif dataset == 'mutag':
            kg_dataset = MUTAGDataset()
        elif dataset == 'bgs':
            kg_dataset = BGSDataset()
        elif dataset == 'am':
            kg_dataset = AMDataset()
        else:
            raise ValueError()

        # Load from hetero-graph
        kg = kg_dataset[0]
        category = kg_dataset.predict_category
        num_classes = kg_dataset.num_classes
        return kg, category, num_classes

    def get_idx(self, validation=True):
        train_mask = self.g.nodes[self.category].data.pop('train_mask')
        test_mask = self.g.nodes[self.category].data.pop('test_mask')
        train_idx = th.nonzero(train_mask, as_tuple=False).squeeze()
        test_idx = th.nonzero(test_mask, as_tuple=False).squeeze()
        if validation:
            val_idx = train_idx[:len(train_idx) // 10]
            train_idx = train_idx[len(train_idx) // 10:]
        else:
            val_idx = train_idx
            train_idx = train_idx
        return train_idx, val_idx, test_idx

    def get_labels(self):
        if 'labels' in self.g.nodes[self.category].data:
            labels = self.g.nodes[self.category].data.pop('labels')
        else:
            raise ValueError('label in not in the hg.nodes[category].data')
        return labels


@register_dataset('hin_node_classification')
class HIN_NodeCLassification(NodeClassificationDataset):
    def __init__(self, dataset_name):
        super(HIN_NodeCLassification, self).__init__()
        self.g, self.category, self.num_classes = self.load_HIN(dataset_name)

    def load_HIN(self, name_dataset):
        if name_dataset == 'acm4NSHE':
            dataset = AcademicDataset(name='acm4NSHE', raw_dir='')
            category = 'paper'
            g = dataset[0].long()
            num_classes = 3
            self.in_dim = g.ndata['h'][category].shape[1]
        elif name_dataset == 'dblp':
            data_path = './openhgnn/dataset/dblp_graph.bin'
            category = 'author'
            num_classes = 4
            g, _ = load_graphs(data_path)
            g = g[0].long()
            self.in_dim = g.ndata['h'][category].shape[1]
        elif name_dataset == 'dblp4MAGNN':
            dataset = AcademicDataset(name='dblp4MAGNN', raw_dir='')
            category = 'A'
            g = dataset[0].long()
            num_classes = 4
            self.in_dim = g.ndata['feat'][category].shape[1]

        elif name_dataset == 'imdb4MAGNN':
            dataset = AcademicDataset(name='imdb4MAGNN', raw_dir='')
            category = 'M'
            g = dataset[0].long()
            num_classes = 3
            self.in_dim = g.ndata['feat'][category].shape[1]
        elif name_dataset == 'imdb4GTN':
            dataset = AcademicDataset(name='imdb4GTN', raw_dir='')
            category = 'movie'
            g = dataset[0].long()
            num_classes = 3
            self.in_dim = g.ndata['h'][category].shape[1]
        elif name_dataset == 'acm4GTN':
            dataset = AcademicDataset(name='acm4GTN', raw_dir='')
            category = 'paper'
            g = dataset[0].long()
            num_classes = 3
            self.in_dim = g.ndata['h'][category].shape[1]
        elif name_dataset == 'acm4NARS':
            dataset = AcademicDataset(name='acm4NARS', raw_dir='')
            g = dataset[0].long()
            num_classes = 3
            # g, labels, num_classes, train_nid, val_nid, test_nid = load_acm_nars()
            category = 'paper'
        elif name_dataset == 'academic4HetGNN':
            # which is used in HetGNN
            dataset = AcademicDataset(name='academic4HetGNN', raw_dir='')
            category = 'author'
            g = dataset[0].long()
            num_classes = 4
        elif name_dataset in ['acm_han', 'acm_han_raw']:
            if name_dataset == 'acm_han':
                g, category, num_classes = load_acm(True)
            elif name_dataset == 'acm_han_raw':
                g, category, num_classes, self.in_dim = load_acm_raw(False)
            else:
                return NotImplementedError('Unsupported dataset {}'.format(name_dataset))
            return g, category, num_classes
        # g, _ = load_graphs(data_path)
        # g = g[0]
        return g, category, num_classes

    def get_idx(self, validation=True):
        if 'train_mask' not in self.g.nodes[self.category].data:
            num_nodes = self.g.number_of_nodes(self.category)

            n_test = int(num_nodes * 0.2)
            n_train = num_nodes - n_test

            train, test = th.utils.data.random_split(range(num_nodes), [n_train, n_test])
            train_idx = th.tensor(train.indices)
            test_idx = th.tensor(test.indices)
            if validation:
                val_idx = train_idx[:len(train_idx) // 10]
                train_idx = train_idx[len(train_idx) // 10:]
            else:
                val_idx = train_idx
                train_idx = train_idx
        else:
            train_mask = self.g.nodes[self.category].data.pop('train_mask')
            test_mask = self.g.nodes[self.category].data.pop('test_mask')
            train_idx = th.nonzero(train_mask, as_tuple=False).squeeze()
            test_idx = th.nonzero(test_mask, as_tuple=False).squeeze()
            if validation:
                if 'val_mask' in self.g.nodes[self.category].data:
                    val_mask = self.g.nodes[self.category].data.pop('val_mask')
                    val_idx = th.nonzero(val_mask, as_tuple=False).squeeze()
                    pass
                else:
                    val_idx = train_idx[:len(train_idx) // 10]
                    train_idx = train_idx[len(train_idx) // 10:]
            else:
                val_idx = train_idx
                train_idx = train_idx
        return train_idx, val_idx, test_idx

    def get_labels(self):
        if 'labels' in self.g.nodes[self.category].data:
            labels = self.g.nodes[self.category].data.pop('labels').long()
        elif 'label' in self.g.nodes[self.category].data:
            labels = self.g.nodes[self.category].data.pop('label').long()
        else:
            raise ValueError('label in not in the hg.nodes[category].data')
        return labels


@register_dataset('HGBn_node_classification')
class HGB_NodeCLassification(NodeClassificationDataset):
    def __init__(self, dataset_name):
        super(HGB_NodeCLassification, self).__init__()
        if dataset_name == 'HGBn-acm':
            data_path = './openhgnn/dataset/HGBn-acm.bin'
            category = 'paper'
            num_classes = 4
            g, _ = load_graphs(data_path)
            g = g[0].long()
            g.nodes['term'].data['h'] = th.eye(g.number_of_nodes('term'))
            self.in_dim = g.ndata['h'][category].shape[1]
            # graph: dgl graph object, label: torch tensor of shape (num_nodes, num_tasks)
        elif dataset_name == 'HGBn-dblp':
            data_path = './openhgnn/dataset/HGBn-dblp.bin'
            category = 'author'
            num_classes = 4
            g, _ = load_graphs(data_path)
            g = g[0].long()
            g.nodes['venue'].data['h'] = th.eye(g.number_of_nodes('venue'))
            self.in_dim = g.ndata['h'][category].shape[1]
            # graph: dgl graph object, label: torch tensor of shape (num_nodes, num_tasks)
        elif dataset_name == 'HGBn-freebase':
            data_path = './openhgnn/dataset/HGBn-freebase.bin'
            category = 'BOOK'
            num_classes = 4
            g, _ = load_graphs(data_path)
            g = g[0].long()
            g.nodes['venue'].data['h'] = th.eye(g.number_of_nodes('venue'))
            self.in_dim = g.ndata['h'][category].shape[1]
            # graph: dgl graph object, label: torch tensor of shape (num_nodes, num_tasks)
        else:
            raise ValueError
        self.g, self.category, self.num_classes = g, category, num_classes

    def get_idx(self, validation=True):
        if 'train_mask' not in self.g.nodes[self.category].data:
            num_nodes = self.g.number_of_nodes(self.category)

            n_test = int(num_nodes * 0.2)
            n_train = num_nodes - n_test

            train, test = th.utils.data.random_split(range(num_nodes), [n_train, n_test])
            train_idx = th.tensor(train.indices)
            test_idx = th.tensor(test.indices)
            if validation:
                val_idx = train_idx[:len(train_idx) // 10]
                train_idx = train_idx[len(train_idx) // 10:]
            else:
                val_idx = train_idx
                train_idx = train_idx
        else:
            train_mask = self.g.nodes[self.category].data.pop('train_mask')
            test_mask = self.g.nodes[self.category].data.pop('test_mask')
            train_idx = th.nonzero(train_mask, as_tuple=False).squeeze()
            test_idx = th.nonzero(test_mask, as_tuple=False).squeeze()
            if validation:
                if 'val_mask' in self.g.nodes[self.category].data:
                    val_mask = self.g.nodes[self.category].data.pop('val_mask')
                    val_idx = th.nonzero(val_mask, as_tuple=False).squeeze()
                    pass
                else:
                    val_idx = train_idx[:len(train_idx) // 5]
                    train_idx = train_idx[len(train_idx) // 5:]
            else:
                val_idx = train_idx
                train_idx = train_idx
        return train_idx, val_idx, test_idx

    def get_labels(self):
        # RuntimeError: Expected object of scalar type Long but got scalar type Float for argument #2 'target' in call to _thnn_nll_loss_forward
        if 'labels' in self.g.nodes[self.category].data:
            labels = self.g.nodes[self.category].data.pop('labels').long()
        elif 'label' in self.g.nodes[self.category].data:
            labels = self.g.nodes[self.category].data.pop('label').long()
        else:
            raise ValueError('label in not in the hg.nodes[category].data')
        return labels


@register_dataset('ogbn_node_classification')
class OGB_NodeCLassification(NodeClassificationDataset):
    def __init__(self, dataset_name):
        super(OGB_NodeCLassification, self).__init__()
        if dataset_name == 'ogbn-mag':
            dataset = DglNodePropPredDataset(name='ogbn-mag')
            self.category = 'paper'  # graph: dgl graph object, label: torch tensor of shape (num_nodes, num_tasks)
        else:
            raise ValueError

        split_idx = dataset.get_idx_split()
        self.num_classes = dataset.num_classes
        self.train_idx, self.valid_idx, self.test_idx = split_idx["train"][self.category], split_idx["valid"][
            self.category], split_idx["test"][self.category]
        self.g, self.label_dict = dataset[0]
        self.g = self.mag4HGT(self.g)
        self.label = self.label_dict[self.category].squeeze(dim=-1)
        # 2-dim label
        self.in_dim = self.g.ndata['h'][self.category].shape[1]
        self.has_feature = True
        # pass

    def get_idx(self, validation=True):
        return self.train_idx, self.valid_idx, self.test_idx

    def get_labels(self):
        return self.label

    def mag4HGT(self, hg):
        # Add reverse edge types

        edges = {etype: hg.edges(etype=etype) for etype in hg.canonical_etypes}
        edges.update({(v, e + '_inv', u): (dst, src) for (u, e, v), (src, dst) in edges.items()})
        hg2 = dgl.heterograph(edges)
        hg2 = dgl.to_simple(hg2)

        # Initialize year
        hg2.nodes['paper'].data['timestamp'] = hg.nodes['paper'].data['year'].squeeze()
        for ntype in hg.ntypes:
            if ntype != 'paper':
                hg2.nodes[ntype].data['timestamp'] = th.zeros(hg2.num_nodes(ntype), dtype=th.int64)

        # Aggregate bag-of-paper features
        hg2.nodes['paper'].data['h'] = hg.nodes['paper'].data['feat']
        hg2.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'h'), etype='has_topic')  # field_of_study
        hg2.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'h'), etype='writes_inv')  # author
        hg2.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'h'), etype='affiliated_with')  # institution

        # Attach log-degree to feature of each node type
        for ntype in hg2.ntypes:
            hg2.nodes[ntype].data['deg'] = th.zeros(hg2.num_nodes(ntype))
        for utype, etype, vtype in hg2.canonical_etypes:
            hg2.nodes[vtype].data['deg'] += hg2.in_degrees(etype=etype)
        for ntype in hg2.ntypes:
            hg2.nodes[ntype].data['h'] = th.cat([
                hg2.nodes[ntype].data['h'],
                th.log10(hg2.nodes[ntype].data['deg'][:, None])], 1)
            del hg2.nodes[ntype].data['deg']

        return hg2
