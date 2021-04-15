import torch as th
import dgl
import os
import numpy as np
from dgl.sampling.neighbor import select_topk
from dgl.data.utils import load_graphs, save_graphs
from ogb.nodeproppred import DglNodePropPredDataset
from ogb.linkproppred import DglLinkPropPredDataset

def load_link_pred(path_file):
    #path_file = './openhgnn/dataset/a_a_list_train.txt'
    u_list = []
    v_list = []
    label_list = []
    with open(path_file) as f:
        for i in f.readlines():
            u, v, label = i.strip().split(', ')
            u_list.append(int(u))
            v_list.append(int(v))
            label_list.append(int(label))
    return u_list, v_list, label_list

def load_dgl_graph(path_file):
    g, _ = load_graphs(path_file)
    return g[0]


def load_OGB(dataset):
    if dataset == 'mag':
        dataset = DglNodePropPredDataset(name='ogbn-mag')
        return dataset
        # split_idx = dataset.get_idx_split()
        # train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        # graph, label = dataset[0]  # graph: dgl graph object, label: torch tensor of shape (num_nodes, num_tasks)
    elif dataset in ['biokg', 'wikikg']:
        d_name = 'ogbl-' + dataset
        dataset = DglLinkPropPredDataset(name=d_name)

        split_edge = dataset.get_edge_split()
        train_edge, valid_edge, test_edge = split_edge["train"], split_edge["valid"], split_edge["test"]
        graph = dataset[0]  # dgl graph object containing only training edges



def load_HIN(dataset):
    if dataset == 'acm':
        data_path = './openhgnn/dataset/acm_graph.bin'
        category = 'paper'
        num_classes = 3
    elif dataset == 'imdb':
        data_path = './openhgnn/dataset/imdb_graph.bin'
        category = 'movie'
        num_classes = 3
    elif dataset == 'acm1':
        data_path = './openhgnn/dataset/acm_graph1.bin'
        category = 'paper'
        num_classes = 3
    elif dataset == 'academic':
        # which is used in HetGNN
        data_path = './openhgnn/dataset/academic.bin'
        category = 'author'
        num_classes = 4
    g = load_dgl_graph(data_path)
    g = g.long()
    return g, category, num_classes


def load_KG(dataset):
    from dgl.data.rdf import AIFBDataset, MUTAGDataset, BGSDataset, AMDataset
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


def give_one_hot_feats(g, ntype='h'):
    # if the nodes are featureless, the input feature is then the node id.
    num_nodes = g.num_nodes()
    #g.ndata[ntype] = th.arange(num_nodes, dtype=th.float32, device=g.device)
    g.ndata[ntype] = th.eye(num_nodes).to(g.device)
    return g


def normalize_edges(hg, ntype='norm'):
    # calculate norm for each edge type and store in edge
    for canonical_etype in hg.canonical_etypes:
        u, v, eid = hg.all_edges(form='all', etype=canonical_etype)
        _, inverse_index, count = th.unique(v, return_inverse=True, return_counts=True)
        degrees = count[inverse_index]
        norm = th.ones(eid.shape[0]).float().to(degrees.device) / degrees.float()
        norm = norm.unsqueeze(1)
        hg.edges[canonical_etype].data[ntype] = norm
    return hg

def normalize_edges_homo(g, i_norm=-1, ntype='norm'):
    # normalization
    degs = g.in_degrees().float()
    norm = th.pow(degs, i_norm)
    norm[th.isinf(norm)] = 0
    g.ndata[ntype] = norm.unsqueeze(1)
    return g


class coarsened_line_graph():

    def __init__(self, rw_len, batch_size, n_dataset, symmetric=True):
        self.rw_len = rw_len
        self.batch_size = batch_size
        self.n_dataset = n_dataset
        self.symmetric = symmetric # which means the original graph had inverse edges

        return

    def get_cl_graph(self, hg):
        fname = './openhgnn/output/RSHN/{}_cl_graoh.bin'.format(
            self.n_dataset)
        if os.path.exists(fname):
            g, _ = load_graphs(fname)
            return g[0]
        else:
            g = self.build_cl_graph(hg)
            save_graphs(fname, g)
            return g

    def init_cl_graph(self, cl_graph):
        cl_graph = give_one_hot_feats(cl_graph, 'h')

        cl_graph = dgl.remove_self_loop(cl_graph)
        edge_attr = cl_graph.edata['w'].type(th.FloatTensor).to(cl_graph.device)
        row, col = cl_graph.edges()
        for i in range(cl_graph.num_nodes()):
            mask = th.eq(col, i)
            edge_attr[mask] = th.nn.functional.normalize(edge_attr[mask], p=2, dim=0)
        cl_graph.edata['w'] = edge_attr

        # add_self_loop, set 1 as edge feature
        cl_graph = dgl.add_self_loop(cl_graph)
        edge_attr = cl_graph.edata['w']
        mask = th.eq(edge_attr, 0)
        edge_attr[mask] = 1
        cl_graph.edata['w'] = edge_attr
        return cl_graph

    def build_cl_graph(self, hg):
        if not hg.is_homogeneous:
            self.num_edge_type = len(hg.etypes)
            g = dgl.to_homogeneous(hg).to('cpu')

        traces = self.random_walks(g)
        edge_batch = self.rw_map_edge_type(g, traces)
        cl_graph = self.edge2graph(edge_batch)
        return cl_graph

    def random_walks(self, g):
        source_nodes = th.randint(0, g.number_of_nodes(), (self.batch_size,))
        traces, _ = dgl.sampling.random_walk(g, source_nodes, length=self.rw_len-1)
        return traces

    def rw_map_edge_type(self, g, traces):
        edge_type = g.edata[dgl.ETYPE].long()
        edge_batch = []
        first_flag = True
        for t in traces:
            u = t[:-1]
            v = t[1:]
            edge_path = edge_type[g.edge_ids(u, v)].unsqueeze(0)
            if first_flag == True:
                edge_batch = edge_path
                first_flag = False
            else:
                edge_batch = th.cat((edge_batch, edge_path), dim=0)
        return edge_batch

    def edge2graph(self, edge_batch):

        u = edge_batch[:, :-1].reshape(-1)
        v = edge_batch[:, 1:].reshape(-1)
        if self.symmetric:
            tmp = u
            u = th.cat((u, v), dim=0)
            v = th.cat((v,tmp), dim=0)

        g = dgl.graph((u, v))
        sg = dgl.to_simple(g, return_counts='w')
        return sg


def edata_in_out_mask(hg):

    """
    An API for CompGCN which needs identify the edge is IN or OUT.

    :param a heterogeneous graph:
    in_edges_mask means the edge is the original edge.
    out_edges_mask means the edge is the inverse edge.

    :return: hg
    """
    for canonical_etype in hg.canonical_etypes:
        eid = hg.all_edges(form='eid', etype=canonical_etype)
        if canonical_etype[1][:4] == 'rev-':
            hg.edges[canonical_etype].data['in_edges_mask'] = th.zeros(eid.shape[0], device=hg.device).bool()
            hg.edges[canonical_etype].data['out_edges_mask'] = th.ones(eid.shape[0], device=hg.device).bool()
        else:
            hg.edges[canonical_etype].data['out_edges_mask'] = th.zeros(eid.shape[0], device=hg.device).bool()
            hg.edges[canonical_etype].data['in_edges_mask'] = th.ones(eid.shape[0], device=hg.device).bool()

    return hg



class hetgnn_graph():
    def __init__(self, hg, n_dataset):
        self.hg = hg
        self.g = dgl.to_homogeneous(hg).to('cpu')
        self.NID = self.g.ndata[dgl.NID]
        self.NTYPE = self.g.ndata[dgl.NTYPE]
        num_nodes = {}
        for i in range(th.max(self.NTYPE) + 1):
            num_nodes[self.hg.ntypes[i]] = int((self.NTYPE == i).sum())
        self.num_nodes = num_nodes
        self.weight_column = 'w'
        self.n_dataset = n_dataset

    def get_hetgnn_graph(self, length, walks, restart_prob):
        fname = './openhgnn/output/HetGNN/{}.bin'.format(
            self.n_dataset)
        if os.path.exists(fname):
            g, _ = load_graphs(fname)
            return g[0]
        else:
            g = self.build_hetgnn_graph(length, walks, restart_prob)
            save_graphs(fname, g)
            return g

    def build_hetgnn_graph(self, length, walks, restart_prob):
        #edges = [[[[],[]]] * len(self.num_nodes)] * len(self.num_nodes)
        edges = [[[[],[]], [[],[]], [[],[]]],
                 [[[],[]], [[],[]], [[],[]]],
                 [[[],[]], [[],[]], [[],[]]]]

        for i in range(self.g.number_of_nodes()):
            nodes = th.tensor([i]).repeat(walks)
            traces, types = dgl.sampling.random_walk(self.g, nodes, length=length, restart_prob=restart_prob)
            concat_vids, _, _, _ = dgl.sampling.pack_traces(traces, types)
            concat_types = th.index_select(self.NTYPE, 0, concat_vids)
            uid = concat_vids[0]
            utype = concat_types[0]
            for (vid, vtype) in zip(concat_vids, concat_types):
                edges[int(utype)][int(vtype)][0].append(self.NID[uid])
                edges[int(utype)][int(vtype)][1].append(self.NID[vid])

        edge_dict = {}
        k = {}
        num_ntypes = self.NTYPE.max() + 1
        for i in range(num_ntypes):
            for j in range(num_ntypes):
                edge = (self.hg.ntypes[j], self.hg.ntypes[j]+'-'+self.hg.ntypes[i], self.hg.ntypes[i])
                edge_dict[edge] = (th.tensor(edges[i][j][1]), th.tensor(edges[i][j][0]))
                if j == 2:
                    k[edge] = 3
                else:
                    k[edge] = 10

        neighbor_graph = dgl.heterograph(
            edge_dict,
            self.num_nodes
        )

        neighbor_graph = dgl.to_simple(neighbor_graph, return_counts=self.weight_column)
        counts = neighbor_graph.edata[self.weight_column]
        neighbor_graph = select_topk(neighbor_graph, k, self.weight_column)

        return neighbor_graph