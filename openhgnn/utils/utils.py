import dgl
import copy
from dgl import backend as F
import torch as th
from scipy.sparse import coo_matrix
import numpy as np
import random
from . import load_HIN, load_KG, load_OGB, BEST_CONFIGS


def sum_up_params(model):
    """ Count the model parameters """
    n = []
    n.append(model.u_embeddings.weight.cpu().data.numel() * 2)
    n.append(model.lookup_table.cpu().numel())
    n.append(model.index_emb_posu.cpu().numel() * 2)
    n.append(model.grad_u.cpu().numel() * 2)

    try:
        n.append(model.index_emb_negu.cpu().numel() * 2)
    except:
        pass
    try:
        n.append(model.state_sum_u.cpu().numel() * 2)
    except:
        pass
    try:
        n.append(model.grad_avg.cpu().numel())
    except:
        pass
    try:
        n.append(model.context_weight.cpu().numel())
    except:
        pass

    print("#params " + str(sum(n)))
    exit()


def add_reverse_edges(hg, copy_ndata=True, copy_edata=True, ignore_one_type=True):
    # get node cnt for each ntype

    canonical_etypes = hg.canonical_etypes
    num_nodes_dict = {ntype: hg.number_of_nodes(ntype) for ntype in hg.ntypes}

    edge_dict = {}
    for etype in canonical_etypes:
        u, v = hg.edges(form='uv', order='eid', etype=etype)
        edge_dict[etype] = (u, v)
        edge_dict[(etype[2], etype[1] + '-rev', etype[0])] = (v, u)
    new_hg = dgl.heterograph(edge_dict, num_nodes_dict=num_nodes_dict)

    # handle features
    if copy_ndata:
        node_frames = dgl.utils.extract_node_subframes(hg, None)
        dgl.utils.set_new_frames(new_hg, node_frames=node_frames)

    if copy_edata:
        for etype in canonical_etypes:
            edge_frame = hg.edges[etype].data
            for data_name, value in edge_frame.items():
                new_hg.edges[etype].data[data_name] = value
    return new_hg


def set_best_config(args):
    configs = BEST_CONFIGS.get(args.task)
    if configs is None:
        print('The task do not have a best_config!')
        return args
    if args.model not in configs:
        print('The model is not in the best config.')
        return args
    configs = configs[args.model]
    for key, value in configs["general"].items():
        args.__setattr__(key, value)
    if args.dataset not in configs:
        print('The dataset is not in the best config.')
        return args
    for key, value in configs[args.dataset].items():
        args.__setattr__(key, value)
    print('Use the best config.')
    return args


class EarlyStopping(object):
    def __init__(self, patience=10, save_path=None):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.best_loss = None
        self.early_stop = False
        if save_path is None:
            self.best_model = None
        self.save_path = save_path

    def step(self, loss, score, model):
        if isinstance(score ,tuple):
            score = score[0]
        if self.best_loss is None:
            self.best_score = score
            self.best_loss = loss
            self.save_model(model)
        elif (loss > self.best_loss) and (score < self.best_score):
            self.counter += 1
            #print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if (score >= self.best_score) and (loss <= self.best_loss):
                self.save_model(model)

            self.best_loss = np.min((loss, self.best_loss))
            self.best_score = np.max((score, self.best_score))
            self.counter = 0
        return self.early_stop

    def step_score(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_model(model)
        elif score < self.best_score:
            self.counter += 1
            #print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if score >= self.best_score:
                self.save_model(model)
                
            self.best_score = np.max((score, self.best_score))
            self.counter = 0
        return self.early_stop

    def loss_step(self, loss, model):
        if self.best_loss is None:
            self.best_loss = loss
            self.save_model(model)
        elif loss >= self.best_loss:
            self.counter += 1
            #print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if loss < self.best_loss:
                self.save_model(model)
            self.best_loss = np.min((loss, self.best_loss))
            self.counter = 0
        return self.early_stop

    def save_model(self, model):
        if self.save_path is None:
            self.best_model = copy.deepcopy(model)
        else:
            model.eval()
            th.save(model.state_dict(), self.save_path)

    def load_model(self, model):
        if self.save_path is None:
            return self.best_model
        else:
            model.load_state_dict(th.load(self.save_path))


def get_nodes_dict(hg):
    n_dict = {}
    for n in hg.ntypes:
        n_dict[n] = hg.num_nodes(n)
    return n_dict

def extract_embed(node_embed, input_nodes):
    emb = {}

    for ntype, nid in input_nodes.items():
        nid = input_nodes[ntype]
        emb[ntype] = node_embed[ntype][nid]
    return emb


def build_dataset(model_name, dataset_name):
    # load the graph(HIN or KG)
    if dataset_name in ['mag']:
        dataset = load_OGB(dataset_name)
        return dataset
    if model_name in ['GTN', 'NSHE', 'HetGNN']:
        g, category, num_classes = load_HIN(dataset_name)
    elif model_name in ['RSHN', 'RGCN', 'CompGCN']:
        g, category, num_classes = load_KG(dataset_name)
    return g, category, num_classes


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    dgl.seed(seed)

def com_mult(a, b):
    r1, i1 = a[..., 0], a[..., 1]
    r2, i2 = b[..., 0], b[..., 1]
    return th.stack([r1 * r2 - i1 * i2, r1 * i2 + i1 * r2], dim=-1)


def conj(a):
    a[..., 1] = -a[..., 1]
    return a


def ccorr(a, b):
    """
    Compute circular correlation of two tensors.
    Parameters
    ----------
    a: Tensor, 1D or 2D
    b: Tensor, 1D or 2D
    Notes
    -----
    Input a and b should have the same dimensions. And this operation supports broadcasting.
    Returns
    -------
    Tensor, having the same dimension as the input a.
    """
    import torch.fft as fft
    return th.irfft(com_mult(conj(th.rfft(a, 1)), th.rfft(b, 1)), 1, signal_sizes=(a.shape[-1],))


def transform_relation_graph_list(hg, category, identity=True):
    r"""
        extract subgraph :math:`G_i` from :math:`G` in which
        only edges whose type :math:`R_i` belongs to :math:`\mathcal{R}`

        Parameters
        ----------
            hg : dgl.heterograph
                Input heterogeneous graph
            category : string
                Type of predicted nodes.
            identity : bool
                If True, the identity matrix will be added to relation matrix set.
    """

    # get target category id
    for i, ntype in enumerate(hg.ntypes):
        if ntype == category:
            category_id = i
    g = dgl.to_homogeneous(hg, ndata='h')
    # find out the target node ids in g
    loc = (g.ndata[dgl.NTYPE] == category_id)
    category_idx = th.arange(g.num_nodes())[loc]


    edges = g.edges()
    etype = g.edata[dgl.ETYPE]
    ctx = g.device
    #g.edata['w'] = th.ones(g.num_edges(), device=ctx)
    num_edge_type = th.max(etype).item()

    # norm = EdgeWeightNorm(norm='right')
    # edata = norm(g.add_self_loop(), th.ones(g.num_edges() + g.num_nodes(), device=ctx))
    graph_list = []
    for i in range(num_edge_type + 1):
        e_ids = th.nonzero(etype == i).squeeze(-1)
        sg = dgl.graph((edges[0][e_ids], edges[1][e_ids]), num_nodes=g.num_nodes())
        # sg.edata['w'] = edata[e_ids]
        sg.edata['w'] = th.ones(sg.num_edges(), device=ctx)
        graph_list.append(sg)
    if identity == True:
        x = th.arange(0, g.num_nodes(), device=ctx)
        sg = dgl.graph((x, x))
        # sg.edata['w'] = edata[g.num_edges():]
        sg.edata['w'] = th.ones(g.num_nodes(), device=ctx)
        graph_list.append(sg)
    return graph_list, g.ndata['h'], category_idx


def extract_mtx_with_id_edge(g):
    # input a homogeneous graph
    # return tensor with shape of [2,num_edges]
    edges = g.edges()
    edata = g.edata['_TYPE']
    num_edge_type = th.max(edata).item()
    ctx = F.context(edges[0])
    dtype = F.dtype(edges[0])
    A = []
    num_nodes = g.num_nodes()
    for i in range(num_edge_type + 1):
        index = th.nonzero(edata == i).squeeze()
        e_0 = edges[0][index].to('cpu').numpy()
        e_1 = edges[1][index].to('cpu').numpy()
        values = np.ones(e_0.shape[0])
        m = coo_matrix((values, (e_0, e_1)), shape=(num_nodes, num_nodes))
        m = th.from_numpy(m.todense()).type(th.FloatTensor).unsqueeze(0)
        if 0 == i:
            A = m
        else:
            A = th.cat([A, m], dim=0)
    m = th.eye(num_nodes).unsqueeze(0)
    A = th.cat([A, m], dim=0)
    return A.to(ctx)


def h2dict(h, hdict):
    pre = 0
    for i, value in hdict.items():
        hdict[i] = h[pre:value.shape[0] + pre]
        pre += value.shape[0]
    return hdict


def print_dict(d, end_string='\n\n'):
    for key in d.keys():
        if isinstance(d[key], dict):
            print('\n', end='')
            print_dict(d[key], end_string='')
        elif isinstance(d[key], int):
            print('{}: {:04d}'.format(key, d[key]), end=', ')
        elif isinstance(d[key], float):
            print('{}: {:.4f}'.format(key, d[key]), end=', ')
        else:
            print('{}: {}'.format(key, d[key]), end=', ')
    print(end_string, end='')


def extract_metapaths(category, canonical_etypes, self_loop=False):
    meta_paths = []
    for etype in canonical_etypes:
        if etype[0] == category:
            for dst_e in canonical_etypes:
                if etype[0] == dst_e[2] and etype[2] == dst_e[0]:
                    if self_loop:
                        meta_paths.append((etype, dst_e))
                    else:
                        if etype[0] != etype[2]:
                            meta_paths.append((etype, dst_e))
    return meta_paths
