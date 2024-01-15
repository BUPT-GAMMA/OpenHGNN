import torch
import dgl
from sklearn.metrics import accuracy_score, roc_auc_score
import random as rd
import numpy as np
import collections
from time import time
import scipy.sparse as sp
import torch.multiprocessing
from openhgnn.trainerflow.base_flow import BaseFlow
from openhgnn.trainerflow import register_flow
from openhgnn.models import build_model
from ..tasks import build_task

torch.multiprocessing.set_sharing_strategy('file_system')
torch.manual_seed(2023)
np.random.seed(2023)
@register_flow('kacltrainer')
class KACLtrainer(BaseFlow):

    def __init__(self, args):
        super(KACLtrainer, self).__init__(args)
        self.args = args
        self.task = build_task(args)

        self.dataset = self.task.dataset
        self.g = self.dataset.g.to(self.device)
        self.kg = self.dataset.kg.to(self.device)
        self._g = self.dataset._g
        self._kg = self.dataset._kg

        self.model = build_model(self.model).build_model_from_args(args=self.args, hg=self.dataset).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        self.optimizer2 = torch.optim.Adam(self.model.parameters(), lr=args.kg_lr)
        self.optimizer3 = torch.optim.Adam(self.model.parameters(), lr=args.cl_lr)
        self.dropout_rate = args.drop_rate

        self.n_train = self._g.num_edges('interact_train')
        self.n_triples = 0
        for etypes in self._kg.canonical_etypes:
            self.n_triples += self._kg.num_edges(etypes)
        self.n_triples = self.n_triples / 2
        self.n_item = self._g.number_of_nodes('item')
        self.n_user = self._g.number_of_nodes('user')
        self.n_entity = self._kg.number_of_nodes('entity')

        self.train_user_dict, self.train_item_dict, self.train_data = {}, {}, []
        edges_train = self._g.edges(etype='interact_train')
        for src, dst in zip(edges_train[0].numpy(), edges_train[1].numpy()):
            self.train_data.append([src, dst])
            if src not in self.train_user_dict:
                self.train_user_dict[src] = []
            self.train_user_dict[src].append(dst)
            if dst not in self.train_item_dict:
                self.train_item_dict[dst] = []
            self.train_item_dict[dst].append(src)
        self.train_data = np.array(self.train_data)
        self.exist_user = list(self.train_user_dict.keys())
        self.exist_item = list(self.train_item_dict.keys())

        edges_test = self._g.edges(etype='interact_test')
        self.test_user_dict, self.test_item_dict = {}, {}
        for src, dst in zip(edges_test[0].numpy(), edges_test[1].numpy()):
            if src not in self.test_user_dict:
                self.test_user_dict[src] = []
            self.test_user_dict[src].append(dst)
            if dst not in self.test_item_dict:
                self.test_item_dict[dst] = []
            self.test_item_dict[dst].append(src)
        self.users_to_test = list(self.test_user_dict.keys())

        self.all_kg_dict = collections.defaultdict(list)
        self.relation_dict = collections.defaultdict(list)
        for etype in self._kg.canonical_etypes:
            edges_kg = self._kg.edges(etype=etype)
            for src_kg, dst_kg in zip(edges_kg[0].numpy(), edges_kg[1].numpy()):
                self.all_kg_dict[src_kg].append((dst_kg, int(etype[1])))
                self.relation_dict[etype].append((src_kg, dst_kg))

    def _mini_train_step(self):
        loss, base_loss, kge_loss, reg_loss, cl_loss = 0., 0., 0., 0., 0.
        cf_drop, kg_drop = 0., 0.

        sub_cf_adjM = self._get_cf_adj_list(is_subgraph=True, dropout_rate=self.args.drop_rate)
        sub_cf_g = dgl.DGLGraph(sub_cf_adjM)
        sub_cf_g = dgl.add_self_loop(sub_cf_g)
        sub_cf_g = sub_cf_g.to(self.device)

        sub_kg_adjM = sum(self._get_kg_adj_list(is_subgraph=True, dropout_rate=self.args.drop_rate))
        sub_kg = dgl.DGLGraph(sub_kg_adjM)
        sub_kg = dgl.remove_self_loop(sub_kg)
        sub_kg = dgl.add_self_loop(sub_kg)
        sub_kg = sub_kg.to(self.device)

        n_batch = int(self.n_train // self.args.batch_size + 1)
        n_kg_batch = int(self.n_triples // self.args.batch_size_kg + 1)
        n_cl_batch = int(self.n_item // self.args.batch_size_cl + 1)

        for idx in range(n_batch):
            self.model.train()
            batch_data = self.generate_train_batch()
            loss, cf_drop, kg_drop = self.model("cf", self.g, sub_cf_g, sub_kg, batch_data['users'], [x + self.n_user for x in batch_data['pos_items']],
                                           [x + self.n_user for x in batch_data['neg_items']])
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        for idx in range(n_kg_batch):
            self.model.train()
            batch_data = self.generate_train_kg_batch()
            kge_loss = self.model("kg", sub_kg, batch_data['heads'], batch_data['relations'], batch_data['pos_tails'],
                             batch_data['neg_tails'])

            self.optimizer2.zero_grad()
            kge_loss.backward()
            self.optimizer2.step()

        for idx in range(n_cl_batch):
            self.model.train()
            batch_data = self.generate_train_cl_batch()
            cl_loss = self.model("cl", sub_cf_g, sub_kg, batch_data['items'])

            self.optimizer3.zero_grad()
            cl_loss.backward()
            self.optimizer3.step()
        del sub_kg, sub_cf_g
        return loss, kge_loss, cl_loss, cf_drop, kg_drop

    def train(self):
        t0 = time()
        cur_best_pre_0 = 0
        loss_loger,  ndcg_loger, hit_loger, auc_loger, acc_loger = [], [], [], [], []
        stopping_step = 0
        should_stop = False

        for epoch in range(self.args.epoch):
            # train
            t1 = time()
            loss, kge_loss, cl_loss, cf_drop, kg_drop = self._mini_train_step()
            show_step = self.args.show_step
            if (epoch + 1) % show_step != 0:
                if self.args.verbose > 0 and epoch % self.args.verbose == 0:
                    perf_str = 'Epoch %d [%.1fs]: train==[%.5f + %.5f + %.5f] drop==[%.2f + %.2f]' % (
                        epoch, time() - t1, float(loss), float(kge_loss), float(cl_loss), float(cf_drop),
                        float(kg_drop))
                    print(perf_str)
                continue

            # test
            t2 = time()
            ret = self.test()
            t3 = time()
            loss_loger.append(float(loss))
            ndcg_loger.append(ret['ndcg'])
            # hit_loger.append(ret['hit_ratio'])
            # auc_loger.append(ret['auc'])
            acc_loger.append(ret['acc'])

            if self.args.verbose > 0:
                perf_str = 'Epoch %d [%.1fs + %.1fs]: train==[%.5f + %.5f], ' \
                           ' , ndcg=[%.5f],  acc[%.5f]' % \
                           (epoch, t2 - t1, t3 - t2, float(loss), float(kge_loss),
                             ret['ndcg'],  ret['acc'])
                print(perf_str)

            cur_best_pre_0, stopping_step, should_stop = self.early_stopping(ret['acc'], cur_best_pre_0,
                                                                        stopping_step, expected_order='acc',
                                                                        flag_step=10)

            # *********************************************************
            # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
            if should_stop == True:
                break


        # ndcgs = np.array(ndcg_loger)
        hit = np.array(hit_loger)
        # auc = np.array(auc_loger)
        acc = np.array(acc_loger)

        # best = np.max(acc)
        idx = int (np.argmax(acc))

        final_perf = "Best Iter=[%d]@[%.1f]\thit=[%s],  acc=[%s] " % \
                     (idx, time() - t0, '\t'.join(['%.5f' % hit[idx]]),
                      '\t'.join(['%.5f' % acc[idx]]))
        print(final_perf)
    def early_stopping(self, log_value, best_value, stopping_step, expected_order='acc', flag_step=100):
        # early stopping strategy:
        assert expected_order in ['acc', 'dec']

        if (expected_order == 'acc' and log_value >= best_value) or (
                expected_order == 'dec' and log_value <= best_value):
            stopping_step = 0
            best_value = log_value
        else:
            stopping_step += 1

        if stopping_step >= flag_step:
            print("Early stopping is trigger at step: {} log:{}".format(flag_step, log_value))
            should_stop = True
        else:
            should_stop = False
        return best_value, stopping_step, should_stop
    def get_auc(self, item_score, user_pos_test):
        item_score = sorted(item_score.items(), key=lambda kv: kv[1])
        item_score.reverse()
        item_sort = [x[0] for x in item_score]
        posterior = [x[1] for x in item_score]

        r = []
        for i in item_sort:
            if i in user_pos_test:
                r.append(1)
            else:
                r.append(0)
        try:
            res = roc_auc_score(y_true=r, y_score=posterior)
        except Exception:
            res = 0
        return res

    def ranklist_by_sorted(self, user_pos_test, test_items, rating):
        item_score = {}
        for i in test_items:
            item_score[i] = rating[i]

        # K_max_item_score = heapq.nlargest(K, item_score.items(), key=lambda item : item[1])
        K_max_item_score = sorted(item_score.items(), key=lambda item: item[1], reverse=True)
        r = []
        for i, _ in K_max_item_score[:self.args.K]:
            if i in user_pos_test:
                r.append(1)
            else:
                r.append(0)

        auc = self.get_auc(item_score, user_pos_test)
        return r, auc

    def ndcg_at_k(self, r, k, ground_truth, method=1):

        def dcg_at_k(r, k, method=1):
            r = np.asfarray(r)[:k]
            if r.size:
                if method == 0:
                    return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
                elif method == 1:
                    return np.sum(r / np.log2(np.arange(2, r.size + 2)))
                else:
                    raise ValueError('method must be 0 or 1.')
            return 0.
        GT = set(ground_truth)
        if len(GT) > k:
            sent_list = [1.0] * k
        else:
            sent_list = [1.0] * len(GT) + [0.0] * (k - len(GT))
        dcg_max = dcg_at_k(sent_list, k, method)
        if not dcg_max:
            return 0.
        return dcg_at_k(r, k, method) / dcg_max
    def get_performance(self, user_pos_test, r, K, auc, acc):

        def hit_at_k(r, k):
            r = np.array(r)[:k]
            if np.sum(r) > 0:
                return 1.
            else:
                return 0.

        ndcg = self.ndcg_at_k(r, K, user_pos_test)
        hit_ratio = hit_at_k(r, K)

        return {'ndcg': ndcg, 'hit_ratio': hit_ratio, 'auc': auc, 'acc': acc}
    def test_one_user(self, x):
        # user u's ratings for user u
        rating = x[0]
        # uid
        u = x[1]
        # user u's items in the training set
        try:
            training_items = self.train_user_dict[u]
        except Exception:
            training_items = []
        # user u's items in the test set
        user_pos_test = self.test_user_dict[u]
        # user_pos_test = data_generator.train_user_dict[u]
        all_items = set(range(self.n_item))
        test_items = list(all_items - set(training_items))
        # test_items = list(range(self.n_item))
        r, auc = self.ranklist_by_sorted(user_pos_test, test_items, rating)

        real_lable = [1 if i in user_pos_test else 0 for i in range(max(test_items))]
        pre_lable = [1 if rating[i] > self.args.threshold else 0 for i in range(max(test_items))]
        acc = accuracy_score(real_lable, pre_lable)

        return self.get_performance(user_pos_test, r, self.args.K, auc, acc)

    def test(self):
        # cores = torch.multiprocessing.cpu_count() // 2
        self.model.eval()
        # result = { 'ndcg': 0., 'hit_ratio': 0., 'auc': 0., 'acc': 0.}
        result = { 'ndcg': 0., 'acc': 0.}
        # pool = torch.multiprocessing.Pool(cores)
        u_batch_size = self.args.batch_size

        test_users = self.users_to_test
        n_test_users = len(test_users)
        n_user_batchs = n_test_users // u_batch_size + 1

        count = 0
        for u_batch_id in range(n_user_batchs):
            start = u_batch_id * u_batch_size
            end = (u_batch_id + 1) * u_batch_size

            user_batch = test_users[start: end]
            item_batch = range(self.n_item)
            with torch.no_grad():
                embedding = self.model("test", self.g, self.kg)
                user = embedding[user_batch]
                item_batch = [x + self.n_user for x in item_batch]
                item = embedding[item_batch]
                rate_batch = torch.mm(user, torch.transpose(item, 0, 1)).detach().cpu().numpy()

            rate_sig = torch.sigmoid(torch.tensor(rate_batch))
            rate_batch = rate_sig.numpy()
            user_batch_rating_uid = zip(rate_batch, user_batch)
            # batch_result = pool.map(self.test_one_user, user_batch_rating_uid)
            batch_result = []
            for data in user_batch_rating_uid:
                result = self.test_one_user(data)
                batch_result.append(result)
            count += len(batch_result)

            for re in batch_result:
                result['ndcg'] += re['ndcg'] / n_test_users
                # result['hit_ratio'] += re['hit_ratio'] / n_test_users
                # result['auc'] += re['auc'] / n_test_users
                result['acc'] += re['acc'] / n_test_users

        assert count == n_test_users
        pool.close()
        return result


    def _full_train_setp(self):
        pass

    def _test_step(self, split=None, logits=None):
        pass

    def generate_train_batch(self):
        batch_data = {}

        if self.args.batch_size <= self.n_user:
            users = rd.sample(self.exist_user, self.args.batch_size)
        else:
            users_list = list(self.exist_user)
            users = [rd.choice(users_list) for _ in range(self.args.batch_size)]

        def sample_pos_items_for_u(u, num):
            pos_items = self.train_user_dict[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            neg_items = []
            while True:
                if len(neg_items) == num: break
                neg_i_id = np.random.randint(low=0, high=self.n_item, size=1)[0]

                if neg_i_id not in self.train_user_dict[u] and neg_i_id not in neg_items:
                    neg_items.append(neg_i_id)
            return neg_items

        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)
        batch_data['users'] = users
        batch_data['pos_items'] = pos_items
        batch_data['neg_items'] = neg_items
        return batch_data

    def generate_train_kg_batch(self):
        batch_data = {}
        exist_heads = list(self.all_kg_dict.keys())
        if self.args.batch_size_kg <= len(exist_heads):
            heads = rd.sample(exist_heads, self.args.batch_size_kg)
        else:
            heads = [rd.choice(exist_heads) for _ in range(self.args.batch_size_kg)]

        def sample_pos_triples_for_h(h, num):
            pos_triples = self.all_kg_dict[h]
            n_pos_triples = len(pos_triples)

            pos_rs, pos_ts = [], []
            while True:
                if len(pos_rs) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_triples, size=1)[0]
                t = pos_triples[pos_id][0]
                r = pos_triples[pos_id][1]

                if r not in pos_rs and t not in pos_ts:
                    pos_rs.append(r)
                    pos_ts.append(t)
            return pos_rs, pos_ts

        def sample_neg_triples_for_h(h, r, num):
            neg_ts = []
            while True:
                if len(neg_ts) == num: break

                t = np.random.randint(low=0, high=self.n_entity, size=1)[0]
                if (t, r) not in self.all_kg_dict[h] and t not in neg_ts:
                    neg_ts.append(t)
            return neg_ts

        pos_r_batch, pos_t_batch, neg_t_batch = [], [], []

        for h in heads:
            pos_rs, pos_ts = sample_pos_triples_for_h(h, 1)
            pos_r_batch += pos_rs
            pos_t_batch += pos_ts

            neg_ts = sample_neg_triples_for_h(h, pos_rs[0], 1)
            neg_t_batch += neg_ts

        batch_data['heads'] = heads
        batch_data['relations'] = pos_r_batch
        batch_data['pos_tails'] = pos_t_batch
        batch_data['neg_tails'] = neg_t_batch

        return batch_data

    def generate_train_cl_batch(self):
        if self.args.batch_size_cl <= len(self.exist_item):
            items = rd.sample(self.exist_item, self.args.batch_size_cl)
        else:
            items_list = list(self.exist_item)
            items = [rd.choice(items_list) for _ in range(self.args.batch_size_cl)]
        batch_data = {}
        batch_data['items'] = items
        return batch_data

    def _get_cf_adj_list(self, is_subgraph=False, dropout_rate=None):
        def _np_mat2sp_adj(np_mat, row_pre, col_pre):
            n_all = self.n_user + self.n_item
            # single-direction
            np_mat = np.array(np_mat)
            a_rows = np_mat[:, 0] + row_pre
            a_cols = np_mat[:, 1] + col_pre
            if is_subgraph is True:
                subgraph_idx = np.arange(len(a_rows))
                subgraph_id = np.random.choice(subgraph_idx, size=int(dropout_rate * len(a_rows)), replace=False)
                a_rows = a_rows[subgraph_id]
                a_cols = a_cols[subgraph_id]

            vals = [1.] * len(a_rows) * 2
            rows = np.concatenate((a_rows, a_cols))
            cols = np.concatenate((a_cols, a_rows))
            adj = sp.coo_matrix((vals, (rows, cols)), shape=(n_all, n_all))
            return adj

        R = _np_mat2sp_adj(self.train_data, row_pre=0, col_pre=self.n_user)
        lap_list = self._bi_norm_lap(R)
        return lap_list

    def _get_kg_adj_list(self, is_subgraph=False, dropout_rate=None):
        adj_mat_list = []

        def _np_mat2sp_adj(np_mat):
            n_all = self.n_entity
            # single-direction
            a_rows = np_mat[:, 0]
            a_cols = np_mat[:, 1]
            if is_subgraph is True:
                subgraph_idx = np.arange(len(a_rows))
                subgraph_id = np.random.choice(subgraph_idx, size=int(dropout_rate * len(a_rows)), replace=False)
                # print(subgraph_id[:10])
                a_rows = a_rows[subgraph_id]
                a_cols = a_cols[subgraph_id]
            a_vals = [1.] * len(a_rows)

            b_rows = a_cols
            b_cols = a_rows
            b_vals = [1.] * len(b_rows)

            a_adj = sp.coo_matrix((a_vals, (a_rows, a_cols)), shape=(n_all, n_all))
            b_adj = sp.coo_matrix((b_vals, (b_rows, b_cols)), shape=(n_all, n_all))

            return a_adj, b_adj

        for r_id in self.relation_dict.keys():
            # print(r_id)
            K, K_inv = _np_mat2sp_adj(np.array(self.relation_dict[r_id]))
            adj_mat_list.append(K)
            adj_mat_list.append(K_inv)

        lap_list = [self._bi_norm_lap(adj) for adj in adj_mat_list]

        return lap_list

    def _bi_norm_lap(self, adj):
        rowsum = np.array(adj.sum(1))

        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

        bi_lap = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
        return bi_lap.tocoo()

