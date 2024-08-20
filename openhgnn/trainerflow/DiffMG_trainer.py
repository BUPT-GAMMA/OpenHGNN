import dgl
import numpy as np
import torch as th
from tqdm import tqdm
import torch
from openhgnn.models import build_model
from . import BaseFlow, register_flow
from ..utils import EarlyStopping
import scipy.sparse as sp
import torch.nn.functional as F
@register_flow("DiffMG_trainer")
class DiffMG_trainer(BaseFlow):
    """Recommendation flows."""

    def __init__(self, args=None):
        super(DiffMG_trainer, self).__init__(args)
        self.target_link = self.task.dataset.target_link
        self.args.out_node_type = self.task.dataset.out_ntypes
        self.args.out_dim = self.args.hidden_dim
        self.model = build_model(self.model).build_model_from_args(self.args, self.hg)
        self.reg_weight = 0.1
        self.metric = ['recall', 'ndcg']
        self.val_metric = 'recall'
        self.topk = 20
        self.patience = args.patience
        self.max_epoch = args.max_epoch
        self.num_neg = self.task.dataset.num_neg
        self.user_name = self.task.dataset.user_name
        self.item_name = self.task.dataset.item_name
        self.num_user = self.hg.num_nodes(self.user_name)
        self.num_item = self.hg.num_nodes(self.user_name)
        self.train_eid_dict = {
            etype: self.hg.edges(etype=etype, form='eid')
            for etype in self.hg.canonical_etypes}

    def preprocess(self):
        if self.args.model == 'DiffMG':
            self.train_hg = self.task.train_hg
            self.val_hg = self.task.val_hg
            self.test_hg = self.task.test_hg
            train_mask = self.train_hg.edges['ui'].data['train_mask'].squeeze()
            train_index = th.nonzero(train_mask).squeeze()
            train_edge = self.train_hg.find_edges(train_index, 'ui')
            train_graph = dgl.heterograph({('user', 'ui', 'item'): train_edge},
                                          {ntype: self.train_hg.number_of_nodes(ntype) for ntype in ['user', 'item']})
            self.train_hg = train_graph
            process(self)
            return

    def train(self):
        self.preprocess()
        train(self)

def find_pos(edges):
    return (edges.data['rating'] > 3)

def find_neg(edges):
    return (edges.data['rating'] < 4)

def find_pairs(target, g, hg,type,u_num):
    eid = g.edges(etype=type)
    if target == 'pos':
        find = hg.filter_edges(find_pos, edges=(eid[0], eid[1]), etype=type)
        pairs = hg.find_edges(find, etype=type)
    else:
        find = hg.filter_edges(find_neg, edges=(eid[0], eid[1]), etype=type)
        pairs = hg.find_edges(find, etype=type)
    pairs = [pairs[0],pairs[1]]
    pairs[1] += u_num
    return pairs

def preprocess_amazon(self):
    hg = self.hg
    np.random.seed(self.args.Amazon_preprocess_seed)
    offsets = {'i': hg.num_nodes('user'), 'b': hg.num_nodes('item') + hg.num_nodes('user')}
    offsets['c'] = offsets['b'] + hg.num_nodes('brand')
    offsets['v'] = offsets['c'] + hg.num_nodes('category')
    in_dims = [hg.num_nodes('user'), hg.num_nodes('item'), hg.num_nodes('brand'),hg.num_nodes('category'), hg.num_nodes(
        'view')]
    node_types = np.zeros((hg.num_nodes(),), dtype=np.int32)
    node_types[offsets['i']:offsets['b']] = 1
    node_types[offsets['b']:offsets['c']] = 2
    node_types[offsets['c']:offsets['v']] = 3
    node_types[offsets['v']:] = 4
    self.train_pos = find_pairs('pos', self.train_hg, self.hg, 'ui', self.hg.num_nodes('user'))
    self.train_neg = find_pairs('neg', self.train_hg, self.hg, 'ui', self.hg.num_nodes('user'))
    self.test_pos = find_pairs('pos', self.test_hg, self.hg, 'ui', self.hg.num_nodes('user'))
    self.test_neg = find_pairs('neg', self.test_hg, self.hg, 'ui', self.hg.num_nodes('user'))
    self.val_pos = find_pairs('pos', self.val_hg, self.hg, 'ui', self.hg.num_nodes('user'))
    self.val_neg = find_pairs('neg', self.val_hg, self.hg, 'ui', self.hg.num_nodes('user'))
    keep_mask = hg.edges['ui'].data['keep_mask'].squeeze()
    keep_index = th.nonzero(keep_mask).squeeze()
    ui_edge = hg.find_edges(keep_index, 'ui')
    ib_edge = hg.edges(etype = 'ib')
    ic_edge = hg.edges(etype = 'ic')
    iv_edge = hg.edges(etype = 'iv')
    edge_dict = {'i':[ui_edge[0],ui_edge[1]],'b':[ib_edge[0],ib_edge[1]],'c':[ic_edge[0],ic_edge[1]],'v':[iv_edge[0],iv_edge[1]]}
    i = 1
    adjs_offset = {}
    for k,v in edge_dict.items():
        adj_offset = np.zeros((node_types.shape[0], node_types.shape[0]), dtype=np.float32)
        adj_offset[v[0].tolist(), (v[1]+ offsets[k]).tolist() ] = 1
        adjs_offset[str(i)] = sp.coo_matrix(adj_offset)
        i = i+1
    self.node_types = node_types
    self.adjs_offset = adjs_offset
    # self.ndarry_dict = ndarry_dict
    self.in_dims = in_dims
def process(self):
    self.cstr_source = {  #* u-side
    "amazon" : [0, 8],
}
    self.cstr_target = {  #* i-side
    "amazon" : [1, 2, 4, 6, 8],
}
    if self.args.gpu > -1:
        print("use gpu---------------")
        torch.cuda.set_device(self.args.gpu)
        torch.cuda.manual_seed(self.args.Amazon_train_seed)
    else:
        print("use cpu_______________")
    np.random.seed(self.args.seed)
    torch.manual_seed(self.args.seed)

    preprocess_amazon(self)
    num_node_types = self.node_types.max() + 1
    node_types = torch.from_numpy(self.node_types)
    adjs_pt = []
    if '0' in self.adjs_offset:
            adjs_pt.append(sparse_mx_to_torch_sparse_tensor(
                normalize_sym(self.adjs_offset['0'] + sp.eye(self.adjs_offset['0'].shape[0], dtype=np.float32))))
    for i in range(1, int(max(self.adjs_offset.keys())) + 1):
            adjs_pt.append(sparse_mx_to_torch_sparse_tensor(
                normalize_row(self.adjs_offset[str(i)] + sp.eye(self.adjs_offset[str(i)].shape[0], dtype=np.float32))))
            adjs_pt.append(sparse_mx_to_torch_sparse_tensor(
                normalize_row(self.adjs_offset[str(i)].T + sp.eye(self.adjs_offset[str(i)].shape[0], dtype=np.float32))))
    adjs_pt.append(
            sparse_mx_to_torch_sparse_tensor(sp.eye(self.adjs_offset['1'].shape[0], dtype=np.float32).tocoo()))
    adjs_pt.append(torch.sparse.FloatTensor(size=self.adjs_offset['1'].shape))
    print("Loading {} adjs...".format(len(adjs_pt)))
    self.serach_adjs_pt = []
    if '0' in self.adjs_offset:
        self.serach_adjs_pt.append(sparse_mx_to_torch_sparse_tensor(normalize_sym(self.adjs_offset['0'] + sp.eye(self.adjs_offset['0'].shape[0], dtype=np.float32))))
    for i in range(1, int(max(self.adjs_offset.keys())) + 1):
        self.serach_adjs_pt.append(sparse_mx_to_torch_sparse_tensor(normalize_row(self.adjs_offset[str(i)] + sp.eye(self.adjs_offset[str(i)].shape[0], dtype=np.float32))))
        self.serach_adjs_pt.append(sparse_mx_to_torch_sparse_tensor(normalize_row(self.adjs_offset[str(i)].T + sp.eye(self.adjs_offset[str(i)].shape[0], dtype=np.float32))))
    self.serach_adjs_pt.append(sparse_mx_to_torch_sparse_tensor(sp.eye(self.adjs_offset['1'].shape[0], dtype=np.float32).tocoo()))
    self.serach_adjs_pt.append(torch.sparse.FloatTensor(size=self.adjs_offset['1'].shape))
    node_feats = []
    for k in range(num_node_types):
        i = torch.stack((torch.arange(self.in_dims[k], dtype=torch.long), torch.arange(self.in_dims[k], dtype=torch.long)))
        v = torch.ones(self.in_dims[k])
        node_feats.append(torch.sparse.FloatTensor(i, v, torch.Size([self.in_dims[k], self.in_dims[k]])))
    self.node_feats = node_feats
    self.adjs_pt = adjs_pt

    search(self)
    steps_s = [len(meta) for meta in self.archs[self.args.dataset]["source"][0]]
    steps_t = [len(meta) for meta in self.archs[self.args.dataset]["target"][0]]
    model_s = self.model(self.in_dims, self.args.attn_dim, steps_s, dropout=self.args.dropout)
    model_t = self.model(self.in_dims, self.args.attn_dim, steps_t, dropout=self.args.dropout)
    optimizer = torch.optim.Adam(
        list(model_s.parameters()) + list(model_t.parameters()),
        lr=self.args.lr,
        weight_decay=self.args.wd
    )
    self.steps_s = steps_s
    self.steps_t = steps_t
    self.model_s = model_s
    self.model_t = model_t
    self.optimizer = optimizer
def train1(self):
    self.model_s.train()
    self.model_t.train()
    self.optimizer.zero_grad()
    out_s = self.model_s(self.node_feats, self.node_types, self.adjs_pt, self.archs[self.args.dataset]["source"][0],
                    self.archs[self.args.dataset]["source"][1], self.args.gpu)
    out_t = self.model_t(self.node_feats, self.node_types, self.adjs_pt, self.archs[self.args.dataset]["target"][0],
                    self.archs[self.args.dataset]["target"][1], self.args.gpu)
    loss = - torch.mean(F.logsigmoid(torch.mul(out_s[self.train_pos[0].tolist()], out_t[self.train_pos[1].tolist()]).sum(dim=-1)) + \
                        F.logsigmoid(- torch.mul(out_s[self.train_neg[0].tolist()], out_t[self.train_neg[1].tolist()]).sum(dim=-1)))
    loss.backward()
    self.optimizer.step()
    return loss.item()

def infer(self):
    import torch.nn.functional as F
    from sklearn.metrics import roc_auc_score
    self.model_s.eval()
    self.model_t.eval()
    with torch.no_grad():
        out_s = self.model_s(self.node_feats, self.node_types, self.adjs_pt, self.archs[self.args.dataset]["source"][0],
                        self.archs[self.args.dataset]["source"][1], self.args.gpu)
        out_t = self.model_t(self.node_feats, self.node_types, self.adjs_pt, self.archs[self.args.dataset]["target"][0],
                        self.archs[self.args.dataset]["target"][1], self.args.gpu)

    # * validation performance
    pos_val_prod = torch.mul(out_s[self.val_pos[0].tolist()], out_t[self.val_pos[1].tolist()]).sum(dim=-1)
    neg_val_prod = torch.mul(out_s[self.val_neg[0].tolist()], out_t[self.val_neg[1].tolist()]).sum(dim=-1)
    loss = - torch.mean(F.logsigmoid(pos_val_prod) + F.logsigmoid(- neg_val_prod))
    y_true_val = np.zeros((len(self.val_pos[0]) + len(self.val_neg[0])), dtype=np.long)
    y_true_val[:len(self.val_pos[0])] = 1
    y_pred_val = np.concatenate(
        (torch.sigmoid(pos_val_prod).cpu().numpy(), torch.sigmoid(neg_val_prod).cpu().numpy()))
    auc_val = roc_auc_score(y_true_val, y_pred_val)
    # * test performance
    pos_test_prod = torch.mul(out_s[self.test_pos[0].tolist()], out_t[self.test_pos[1].tolist()]).sum(dim=-1)
    neg_test_prod = torch.mul(out_s[self.test_neg[0].tolist()], out_t[self.test_neg[1].tolist()]).sum(dim=-1)

    y_true_test = np.zeros((len(self.test_pos[0]) + len(self.test_neg[0])), dtype=np.long)
    y_true_test[:len(self.test_pos[0])] = 1
    y_pred_test = np.concatenate(
        (torch.sigmoid(pos_test_prod).cpu().numpy(), torch.sigmoid(neg_test_prod).cpu().numpy()))
    auc_test = roc_auc_score(y_true_test, y_pred_test)
    return loss.item(), auc_val, auc_test

def train(self):
    import logging
    best_val = None
    final = None
    anchor = None
    for epoch in range(self.max_epoch):
        train_loss = train1(self)
        val_loss, auc_val, auc_test = infer(self)
        logging.info(
            "Epoch {}; Train err {}; Val err {}; Val auc {}".format(epoch + 1, train_loss, val_loss, auc_val))
        if best_val is None or auc_val > best_val:
            best_val = auc_val
            final = auc_test
            anchor = epoch + 1
    logging.info("Best val auc {} at epoch {}; Test auc {}".format(best_val, anchor, final))

def search(self):
    # torch.cuda.set_device(args.gpu)
    import logging
    np.random.seed(self.args.Amazon_search_seed)
    torch.manual_seed(self.args.Amazon_search_seed)

    model_s = self.args.search_model(self.in_dims, self.args.attn_dim, len( self.serach_adjs_pt), [self.args.search_steps_s], self.cstr_source[self.args.dataset])
    model_t = self.args.search_model(self.in_dims, self.args.attn_dim, len( self.serach_adjs_pt), [self.args.search_steps_t], self.cstr_target[self.args.dataset])
    optimizer_w = torch.optim.Adam(
        list(model_s.parameters()) + list(model_t.parameters()),
        lr=self.args.search_lr,
        weight_decay=self.args.search_wd
    )

    optimizer_a = torch.optim.Adam(
        model_s.alphas() + model_t.alphas(),
        lr=self.args.search_alr
    )

    eps = self.args.search_eps
    for epoch in range(self.args.search_epochs):
        min_error = 100
        train_error, val_error = serach_train(self,model_s, model_t, optimizer_w, optimizer_a, eps)
        logging.info(
            "Epoch {}; Train err {}; Val err {}; Source arch {}; Target arch {}".format(epoch + 1, train_error,
                                                                                        val_error, model_s.parse(),
                                                                                        model_t.parse()))
        if val_error < min_error:
            self.archs = {
                "amazon": {
                    "source": model_s.parse(),
                    "target": model_t.parse()
                },
            }
            min_error = val_error
        eps = eps * self.args.search_decay

def serach_train(self, model_s, model_t, optimizer_w,
          optimizer_a, eps):

    idxes_seq_s, idxes_res_s = model_s.sample(eps)
    idxes_seq_t, idxes_res_t = model_t.sample(eps)

    optimizer_w.zero_grad()
    out_s = model_s(self.node_feats, self.node_types, self.serach_adjs_pt, idxes_seq_s, idxes_res_s)
    out_t = model_t(self.node_feats, self.node_types,  self.serach_adjs_pt, idxes_seq_t, idxes_res_t)
    loss_w = - torch.mean(F.logsigmoid(torch.mul(out_s[self.train_pos[0].tolist()], out_t[self.train_pos[1].tolist()]).sum(dim=-1)) + \
                          F.logsigmoid(- torch.mul(out_s[self.train_neg[0].tolist()], out_t[self.train_neg[1].tolist()]).sum(dim=-1)))
    loss_w.backward()
    optimizer_w.step()

    optimizer_a.zero_grad()
    out_s = model_s(self.node_feats, self.node_types,  self.serach_adjs_pt, idxes_seq_s, idxes_res_s)
    out_t = model_t(self.node_feats, self.node_types,  self.serach_adjs_pt, idxes_seq_t, idxes_res_t)
    loss_a = - torch.mean(F.logsigmoid(torch.mul(out_s[self.val_pos[0].tolist()], out_t[self.val_pos[1].tolist()]).sum(dim=-1)) + \
                          F.logsigmoid(- torch.mul(out_s[self.val_neg[0].tolist()], out_t[self.val_neg[1].tolist()]).sum(dim=-1)))
    loss_a.backward()
    optimizer_a.step()

    return loss_w.item(), loss_a.item()

def normalize_sym(adj):
    """Symmetrically normalize adjacency matrix."""
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
def normalize_row(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx.tocoo()
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    indices = torch.from_numpy(
    np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)