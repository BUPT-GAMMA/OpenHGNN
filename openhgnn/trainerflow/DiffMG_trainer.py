import dgl
import pandas as pd
from tqdm import tqdm
import pickle
import os
import sys
import numpy as np
import scipy.sparse as sp
import logging
import argparse
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import torch as th
from torch import nn
from tqdm import tqdm
import torch
import torch.nn.functional as F
from openhgnn.trainerflow import BaseFlow, register_flow
from openhgnn.models import build_model
from openhgnn.tasks import build_task
##! cstr includes identity but not zero
cstr_source = {  # * u-side
    "Amazon": [0, 8],
}

cstr_target = {  # * i-side
    "Amazon": [1, 2, 4, 6, 8],
}
archs = {
    "amazon" : {
        "source" : ([[4, 3, 2, 0]], [[1, 1, 9, 9, 0, 8]]),
        "target" : ([[5, 4, 2, 1]], [[9, 2, 7, 9, 8, 6]])
    },
}

@register_flow("DiffMG_trainer")
class DiffMG_trainer(BaseFlow):
    def __init__(self, args):
        super(DiffMG_trainer, self).__init__(args)
        self.args = args
        self.lr = self.args.lr
        self.model_name = self.args.model_name
        self.task = build_task(args)
        self.device = self.args.device
        self.model = build_model(self.model).build_model_from_args(self.args)

    def normalize_sym(self,adj):
        """Symmetrically normalize adjacency matrix."""
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

    def normalize_row(self,mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx.tocoo()

    def sparse_mx_to_torch_sparse_tensor(self,sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def read_dgl(self):
        pbar = tqdm(total=5, desc='loading dgl', position=0)
        # with tqdm(total=5,desc='loading dgl') as pbar:
        g = self.hg
        ui_src = (g.edges(etype='ui')[0]).numpy().tolist()
        ui_dst = (g.edges(etype='ui')[1]).numpy().tolist()
        ui_r = (g.edges['ui'].data['rating']).numpy().tolist()
        ui_t = (g.edges['ui'].data['time']).numpy().tolist()
        ui = [ui_src, ui_dst, ui_r, ui_t]
        ui_dataframe = (pd.DataFrame(ui)).transpose()
        ui_dataframe.columns = ['uid', 'iid', 'rating', 'time']
        pbar.update(2)

        ib_src = (g.edges(etype='ib')[0]).numpy().tolist()
        ib_dst = (g.edges(etype='ib')[1]).numpy().tolist()
        ib = [ib_src, ib_dst]
        ib_dataframe = (pd.DataFrame(ib)).transpose()
        ib_dataframe.columns = ['iid', 'bid']
        pbar.update(1)

        ic_src = (g.edges(etype='ic')[0]).numpy().tolist()
        ic_dst = (g.edges(etype='ic')[1]).numpy().tolist()
        ic = [ic_src, ic_dst]
        ic_dataframe = (pd.DataFrame(ic)).transpose()
        ic_dataframe.columns = ['iid', 'cid']
        pbar.update(1)

        iv_src = (g.edges(etype='iv')[0]).numpy().tolist()
        iv_dst = (g.edges(etype='iv')[1]).numpy().tolist()
        iv = [iv_src, iv_dst]
        iv_dataframe = (pd.DataFrame(iv)).transpose()
        iv_dataframe.columns = ['iid', 'vid']
        pbar.update(1)

        return ui_dataframe, ib_dataframe, ic_dataframe, iv_dataframe

    def preprocess_amazon(self):
        # print("processing data:")
        # with tqdm(total=8,desc='processing data') as pbar:
        pbar = tqdm(total=8, desc='processing data', position=0)
        # * indices start from 0
        np.random.seed(self.args.Amazon_preprocess_seed)
        ui, ib, ic, iv = self.read_dgl()
        pbar.update(3)
        u_num = ui['uid'].unique().shape[0]
        i_num = ui['iid'].unique().shape[0]
        print(u_num, i_num)
        # ! unconnected pairs
        # 升序排列
        ui = ui.sort_values(by=['uid', 'iid'], ascending=[True, True]).reset_index(drop=True)
        unconnected_pairs_offset = []
        # unconnected_pairs_offset = np.load("unconnected_pairs_offset.npy")
        pbar.update(1)
        count = 0
        print("ok")
        unconnected = tqdm(total=u_num, desc='get unconnected pairs', position=0)
        for u in range(u_num):
            for i in range(i_num):
                if count < ui.shape[0]:
                    if i == ui.iloc[count]['iid'] and u == ui.iloc[count]['uid']:
                        count += 1
                    else:
                        unconnected_pairs_offset.append([u, i + u_num])
                else:
                    unconnected_pairs_offset.append([u, i + u_num])
            unconnected.update(1)
        assert (count == ui.shape[0])
        assert (count + len(unconnected_pairs_offset) == u_num * i_num)
        unconnected_pairs_offset = np.array(unconnected_pairs_offset)
        # np.save("unconnected_pairs_offset", np.array(unconnected_pairs_offset))

        offsets = {'i': u_num, 'b': u_num + i_num}
        offsets['c'] = offsets['b'] + ib['bid'].max() + 1
        offsets['v'] = offsets['c'] + ic['cid'].max() + 1

        # * node types
        node_types = np.zeros((offsets['v'] + iv['vid'].max() + 1,), dtype=np.int32)
        node_types[offsets['i']:offsets['b']] = 1
        node_types[offsets['b']:offsets['c']] = 2
        node_types[offsets['c']:offsets['v']] = 3
        node_types[offsets['v']:] = 4
        pbar.update(1)
        # if not os.path.exists("./preprocessed/Amazon/node_types.npy"):
        #     np.save("./preprocessed/Amazon/node_types", node_types)

        # * positive pairs
        ui_pos = ui[ui['rating'] > 3].to_numpy()[:, :2]

        # ! negative rating
        neg_ratings = ui[ui['rating'] < 4].to_numpy()[:, :2]
        assert (ui_pos.shape[0] + neg_ratings.shape[0] == ui.shape[0])
        neg_ratings[:, 1] += offsets['i']
        pbar.update(1)
        # np.save("./preprocessed/Amazon/neg_ratings_offset", neg_ratings)

        indices = np.arange(ui_pos.shape[0])
        np.random.shuffle(indices)
        keep, mask = np.array_split(indices, 2)
        np.random.shuffle(mask)
        train, val, test = np.array_split(mask, [int(len(mask) * 0.6), int(len(mask) * 0.8)])

        ui_pos_train = ui_pos[train]
        ui_pos_val = ui_pos[val]
        ui_pos_test = ui_pos[test]

        ui_pos_train[:, 1] += offsets['i']
        ui_pos_val[:, 1] += offsets['i']
        ui_pos_test[:, 1] += offsets['i']
        pbar.update(1)
        # np.savez("./preprocessed/Amazon/pos_pairs_offset", train=ui_pos_train, val=ui_pos_val, test=ui_pos_test)

        neg_train, neg_val, neg_test = self.gen_neg(ui_pos_train, ui_pos_val, ui_pos_test, unconnected_pairs_offset,
                                               neg_ratings, self.args.Amazon_gen_neg_seed)
        # * adjs with offset
        adjs_offset = {}

        ## ui
        ui_pos_keep = ui_pos[keep]
        adj_offset = np.zeros((node_types.shape[0], node_types.shape[0]), dtype=np.float32)
        adj_offset[ui_pos_keep[:, 0], ui_pos_keep[:, 1] + offsets['i']] = 1
        adjs_offset['1'] = sp.coo_matrix(adj_offset)

        ## ib
        ib_npy = ib.to_numpy()[:, :2]
        adj_offset = np.zeros((node_types.shape[0], node_types.shape[0]), dtype=np.float32)
        adj_offset[ib_npy[:, 0] + offsets['i'], ib_npy[:, 1] + offsets['b']] = 1
        adjs_offset['2'] = sp.coo_matrix(adj_offset)

        ## ic
        ic_npy = ic.to_numpy()[:, :2]
        adj_offset = np.zeros((node_types.shape[0], node_types.shape[0]), dtype=np.float32)
        adj_offset[ic_npy[:, 0] + offsets['i'], ic_npy[:, 1] + offsets['c']] = 1
        adjs_offset['3'] = sp.coo_matrix(adj_offset)

        ## iv
        iv_npy = iv.to_numpy()[:, :2]
        adj_offset = np.zeros((node_types.shape[0], node_types.shape[0]), dtype=np.float32)
        adj_offset[iv_npy[:, 0] + offsets['i'], iv_npy[:, 1] + offsets['v']] = 1
        adjs_offset['4'] = sp.coo_matrix(adj_offset)
        pbar.update(1)

        # f2 = open("./preprocessed/Amazon/adjs_offset.pkl", "wb")
        # pickle.dump(adjs_offset, f2)
        # f2.close()
        return node_types, neg_ratings, ui_pos_train, ui_pos_val, ui_pos_test, adjs_offset, neg_train, neg_val, neg_test

    def gen_neg(self,pos_pairs_offset_train, pos_pairs_offset_val, pos_pairs_offset_test, unconnected_pairs_offset,
                neg_ratings_offset, neg_seed):

        # pos_pairs_offset = np.load(os.path.join(prefix, "pos_pairs_offset.npz"))
        # unconnected_pairs_offset = np.load(os.path.join(prefix, "unconnected_pairs_offset.npy"))
        # neg_ratings_offset = np.load(os.path.join(prefix, "neg_ratings_offset.npy"))
        np.random.seed(neg_seed)
        train_len = pos_pairs_offset_train.shape[0]
        val_len = pos_pairs_offset_val.shape[0]
        test_len = pos_pairs_offset_test.shape[0]
        pos_len = train_len + val_len + test_len

        if pos_len > neg_ratings_offset.shape[0]:
            indices = np.arange(unconnected_pairs_offset.shape[0])
            assert (indices.shape[0] > pos_len)
            np.random.shuffle(indices)
            makeup = indices[:pos_len - neg_ratings_offset.shape[0]]
            neg_ratings_offset = np.concatenate((neg_ratings_offset, unconnected_pairs_offset[makeup]), axis=0)
            assert (pos_len == neg_ratings_offset.shape[0])
        indices = np.arange(neg_ratings_offset.shape[0])
        np.random.shuffle(indices)
        train = neg_ratings_offset[indices[:train_len]],
        val = neg_ratings_offset[indices[train_len:train_len + val_len]],
        test = neg_ratings_offset[indices[train_len + val_len:pos_len]]
        return train, val, test
        # np.savez(os.path.join(prefix, "neg_pairs_offset"), train=neg_ratings_offset[indices[:train_len]],
        #                                                     val=neg_ratings_offset[indices[train_len:train_len + val_len]],
        #                                                     test=neg_ratings_offset[indices[train_len + val_len:pos_len]])

        # ! Yelp 2
        # ! Amazon 4
        # ! Douban_Movie 6

    def process(self):
        if self.args.gpu > -1:
            print("use gpu---------------")
            torch.cuda.set_device(self.args.gpu)
            torch.cuda.manual_seed(self.args.Amazon_train_seed)
        else:
            print("use cpu_______________")
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        steps_s = [len(meta) for meta in archs[self.args.dataset]["source"][0]]
        steps_t = [len(meta) for meta in archs[self.args.dataset]["target"][0]]
        # print(steps_s, steps_t)

        datadir = "preprocessed"
        prefix = os.path.join(datadir, self.args.dataset)
        
        if self.args.dataset == 'amazon':
            node_types, neg_ratings, pos_train, pos_val, pos_test, adjs_offset, neg_train, neg_val, neg_test = self.preprocess_amazon()
        # * load data
        # node_types = np.load(os.path.join(prefix, "node_types.npy"))
        num_node_types = node_types.max() + 1
        if self.args.gpu > -1:
            node_types = torch.from_numpy(node_types).cuda()
        else:
            node_types = torch.from_numpy(node_types)
        # adjs_offset = pickle.load(open(os.path.join(prefix, "adjs_offset.pkl"), "rb"))

        adjs_pt = []
        if '0' in adjs_offset:
            if self.args.gpu > -1:
                adjs_pt.append(self.sparse_mx_to_torch_sparse_tensor(
                    self.normalize_sym(adjs_offset['0'] + sp.eye(adjs_offset['0'].shape[0], dtype=np.float32))).cuda())
            else:
                adjs_pt.append(self.sparse_mx_to_torch_sparse_tensor(
                    self.normalize_sym(adjs_offset['0'] + sp.eye(adjs_offset['0'].shape[0], dtype=np.float32))))
        for i in range(1, int(max(adjs_offset.keys())) + 1):
            if self.args.gpu > -1:
                adjs_pt.append(self.sparse_mx_to_torch_sparse_tensor(
                    self.normalize_row(adjs_offset[str(i)] + sp.eye(adjs_offset[str(i)].shape[0], dtype=np.float32))).cuda())
                adjs_pt.append(self.sparse_mx_to_torch_sparse_tensor(
                    self.normalize_row(
                        adjs_offset[str(i)].T + sp.eye(adjs_offset[str(i)].shape[0], dtype=np.float32))).cuda())
            else:
                adjs_pt.append(self.sparse_mx_to_torch_sparse_tensor(
                    self.normalize_row(adjs_offset[str(i)] + sp.eye(adjs_offset[str(i)].shape[0], dtype=np.float32))))
                adjs_pt.append(self.sparse_mx_to_torch_sparse_tensor(
                    self.normalize_row(adjs_offset[str(i)].T + sp.eye(adjs_offset[str(i)].shape[0], dtype=np.float32))))
        if self.args.gpu > -1:
            adjs_pt.append(
                self.sparse_mx_to_torch_sparse_tensor(sp.eye(adjs_offset['1'].shape[0], dtype=np.float32).tocoo()).cuda())
            adjs_pt.append(torch.sparse.FloatTensor(size=adjs_offset['1'].shape).cuda())
        else:
            adjs_pt.append(
                self.sparse_mx_to_torch_sparse_tensor(sp.eye(adjs_offset['1'].shape[0], dtype=np.float32).tocoo()))
            adjs_pt.append(torch.sparse.FloatTensor(size=adjs_offset['1'].shape))
        print("Loading {} adjs...".format(len(adjs_pt)))

        neg_train = np.array(neg_train)
        neg_train = neg_train[0]
        neg_val = np.array(neg_val)
        neg_val = neg_val[0]
        # * one-hot IDs as input features
        in_dims = []
        node_feats = []
        for k in range(num_node_types):
            in_dims.append((node_types == k).sum().item())
            i = torch.stack((torch.arange(in_dims[-1], dtype=torch.long), torch.arange(in_dims[-1], dtype=torch.long)))
            v = torch.ones(in_dims[-1])
            if self.args.gpu > -1:
                node_feats.append(torch.sparse.FloatTensor(i, v, torch.Size([in_dims[-1], in_dims[-1]])).cuda())
                assert (len(in_dims) == len(node_feats))

                model_s = self.model(in_dims, self.args.n_hid, steps_s, dropout=self.args.dropout).cuda()
                model_t = self.model(in_dims, self.args.n_hid, steps_t, dropout=self.args.dropout).cuda()
            else:
                node_feats.append(torch.sparse.FloatTensor(i, v, torch.Size([in_dims[-1], in_dims[-1]])))
                assert (len(in_dims) == len(node_feats))

                model_s = self.model(in_dims, self.args.n_hid, steps_s, dropout=self.args.dropout)
                model_t = self.model(in_dims, self.args.n_hid, steps_t, dropout=self.args.dropout)

        optimizer = torch.optim.Adam(
            list(model_s.parameters()) + list(model_t.parameters()),
            lr=self.args.lr,
            weight_decay=self.args.wd
        )
        self.node_feats = node_feats
        self.node_types = node_types
        self.adjs_pt = adjs_pt
        self.pos_train = pos_train
        self.neg_train = neg_train
        self.pos_val = pos_val
        self.neg_val = neg_val
        self.pos_test = pos_test
        self.neg_test = neg_test
        self.model_s = model_s
        self.model_t = model_t
        self.optimizer = optimizer


    def train1(self,node_feats, node_types, adjs, pos_train, neg_train, model_s, model_t, optimizer, gpu):
        model_s.train()
        model_t.train()
        optimizer.zero_grad()
        out_s = model_s(node_feats, node_types, adjs, archs[self.args.dataset]["source"][0],
                        archs[self.args.dataset]["source"][1], gpu)
        out_t = model_t(node_feats, node_types, adjs, archs[self.args.dataset]["target"][0],
                        archs[self.args.dataset]["target"][1], gpu)
        loss = - torch.mean(F.logsigmoid(torch.mul(out_s[pos_train[:, 0]], out_t[pos_train[:, 1]]).sum(dim=-1)) + \
                            F.logsigmoid(- torch.mul(out_s[neg_train[:, 0]], out_t[neg_train[:, 1]]).sum(dim=-1)))
        loss.backward()
        optimizer.step()
        return loss.item()

    def infer(self,node_feats, node_types, adjs, pos_val, neg_val, pos_test, neg_test, model_s, model_t, gpu):
        model_s.eval()
        model_t.eval()
        with torch.no_grad():
            out_s = model_s(node_feats, node_types, adjs, archs[self.args.dataset]["source"][0],
                            archs[self.args.dataset]["source"][1], gpu)
            out_t = model_t(node_feats, node_types, adjs, archs[self.args.dataset]["target"][0],
                            archs[self.args.dataset]["target"][1], gpu)

        # * validation performance
        pos_val_prod = torch.mul(out_s[pos_val[:, 0]], out_t[pos_val[:, 1]]).sum(dim=-1)
        neg_val_prod = torch.mul(out_s[neg_val[:, 0]], out_t[neg_val[:, 1]]).sum(dim=-1)
        loss = - torch.mean(F.logsigmoid(pos_val_prod) + F.logsigmoid(- neg_val_prod))

        y_true_val = np.zeros((pos_val.shape[0] + neg_val.shape[0]), dtype=np.long)
        y_true_val[:pos_val.shape[0]] = 1
        y_pred_val = np.concatenate(
            (torch.sigmoid(pos_val_prod).cpu().numpy(), torch.sigmoid(neg_val_prod).cpu().numpy()))
        auc_val = roc_auc_score(y_true_val, y_pred_val)

        # * test performance
        pos_test_prod = torch.mul(out_s[pos_test[:, 0]], out_t[pos_test[:, 1]]).sum(dim=-1)
        neg_test_prod = torch.mul(out_s[neg_test[:, 0]], out_t[neg_test[:, 1]]).sum(dim=-1)

        y_true_test = np.zeros((pos_test.shape[0] + neg_test.shape[0]), dtype=np.long)
        y_true_test[:pos_test.shape[0]] = 1
        y_pred_test = np.concatenate(
            (torch.sigmoid(pos_test_prod).cpu().numpy(), torch.sigmoid(neg_test_prod).cpu().numpy()))
        auc_test = roc_auc_score(y_true_test, y_pred_test)

        return loss.item(), auc_val, auc_test

    def train(self):
        self.process()
        best_val = None
        final = None
        anchor = None
        for epoch in range(self.max_epoch):
            train_loss = self.train1(self.node_feats, self.node_types, self.adjs_pt, self.pos_train, self.neg_train, self.model_s, self.model_t, self.optimizer,
                                    self.args.gpu)
            val_loss, auc_val, auc_test = self.infer(self.node_feats, self.node_types, self.adjs_pt, self.pos_val, self.neg_val, self.pos_test,
                                                     self.neg_test,
                                                     self.model_s, self.model_t, self.args.gpu)
            logging.info(
                "Epoch {}; Train err {}; Val err {}; Val auc {}".format(epoch + 1, train_loss, val_loss, auc_val))
            if best_val is None or auc_val > best_val:
                best_val = auc_val
                final = auc_test
                anchor = epoch + 1
        logging.info("Best val auc {} at epoch {}; Test auc {}".format(best_val, anchor, final))




