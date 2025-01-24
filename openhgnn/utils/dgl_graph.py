import torch as th
from dgl.data.utils import load_graphs
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
