import os
import torch
import numpy as np
from scipy.sparse import csr_matrix
from collections import defaultdict
import requests
import zipfile
import io

class AdapropIDataLoader:
    def __init__(self, args):
        self.args = args
        self.dir = './data'
        name1=self.args.dataset_name
        name2=name1+'_ind'
        path_ckp1 = os.path.join(self.dir, name1)
        path_ckp2 = os.path.join(self.dir, name2)
        self.dir = os.path.join(self.dir, name1)
        task_dir=self.dir
        print(path_ckp1)
        folder = os.path.exists(path_ckp1)
        if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
            os.makedirs(path_ckp1)  # makedirs 创建文件时如果路径不存在会创建这个路径
            # 下载数据
            if name1=='fb237_v1':
                url = "https://s3.cn-north-1.amazonaws.com.cn/dgl-data/dataset/openhgnn/fb237_v1.zip"
            elif name1=='fb237_v2':
                url = "https://s3.cn-north-1.amazonaws.com.cn/dgl-data/dataset/openhgnn/fb237_v2.zip"
            elif name1=='fb237_v3':
                url = "https://s3.cn-north-1.amazonaws.com.cn/dgl-data/dataset/openhgnn/fb237_v3.zip"
            elif name1=='fb237_v4':
                url = "https://s3.cn-north-1.amazonaws.com.cn/dgl-data/dataset/openhgnn/fb237_v4.zip"
            response = requests.get(url)
            with zipfile.ZipFile(io.BytesIO(response.content)) as myzip:
                myzip.extractall(path_ckp1)
            print("---  download data  ---")

        else:
            print("---  There is data!  ---")

        print(path_ckp2)
        folder = os.path.exists(path_ckp2)
        if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
            os.makedirs(path_ckp2)  # makedirs 创建文件时如果路径不存在会创建这个路径
            # 下载数据
            if name1=='fb237_v1':
                url = "https://s3.cn-north-1.amazonaws.com.cn/dgl-data/dataset/openhgnn/fb237_v1_ind.zip"
            elif name1=='fb237_v2':
                url = "https://s3.cn-north-1.amazonaws.com.cn/dgl-data/dataset/openhgnn/fb237_v2_ind.zip"
            elif name1=='fb237_v3':
                url = "https://s3.cn-north-1.amazonaws.com.cn/dgl-data/dataset/openhgnn/fb237_v3_ind.zip"
            elif name1=='fb237_v4':
                url = "https://s3.cn-north-1.amazonaws.com.cn/dgl-data/dataset/openhgnn/fb237_v4_ind.zip"
            response = requests.get(url)
            with zipfile.ZipFile(io.BytesIO(response.content)) as myzip:
                myzip.extractall(path_ckp2)
            print("---  download data  ---")

        else:
            print("---  There is data!  ---")
        self.task_dir=self.dir
        task_dir=self.task_dir
        n_batch=args.n_batch
        self.trans_dir = task_dir
        self.n_batch = n_batch
        self.ind_dir = task_dir + '_ind'

        with open(os.path.join(task_dir, 'entities.txt')) as f:
            self.entity2id = dict()
            for line in f:
                entity, eid = line.strip().split()
                self.entity2id[entity] = int(eid)

        with open(os.path.join(task_dir, 'relations.txt')) as f:
            self.relation2id = dict()
            id2relation = []
            for line in f:
                relation, rid = line.strip().split()
                self.relation2id[relation] = int(rid)
                id2relation.append(relation)

        with open(os.path.join(self.ind_dir, 'entities.txt')) as f:
            self.entity2id_ind = dict()
            for line in f:
                entity, eid = line.strip().split()
                self.entity2id_ind[entity] = int(eid)

        for i in range(len(self.relation2id)):
            id2relation.append(id2relation[i] + '_inv')
        id2relation.append('idd')
        self.id2relation = id2relation

        self.n_ent = len(self.entity2id)
        self.n_rel = len(self.relation2id)
        self.n_ent_ind = len(self.entity2id_ind)

        self.tra_train = self.read_triples(self.trans_dir, 'train.txt')
        self.tra_valid = self.read_triples(self.trans_dir, 'valid.txt')
        self.tra_test = self.read_triples(self.trans_dir, 'test.txt')
        self.ind_train = self.read_triples(self.ind_dir, 'train.txt', 'inductive')
        self.ind_valid = self.read_triples(self.ind_dir, 'valid.txt', 'inductive')
        self.ind_test = self.read_triples(self.ind_dir, 'test.txt', 'inductive')

        self.val_filters = self.get_filter('valid')
        self.tst_filters = self.get_filter('test')

        for filt in self.val_filters:
            self.val_filters[filt] = list(self.val_filters[filt])
        for filt in self.tst_filters:
            self.tst_filters[filt] = list(self.tst_filters[filt])

        self.tra_KG, self.tra_sub = self.load_graph(self.tra_train)
        self.ind_KG, self.ind_sub = self.load_graph(self.ind_train, 'inductive')

        self.tra_train = np.array(self.tra_valid)
        self.tra_val_qry, self.tra_val_ans = self.load_query(self.tra_test)
        self.ind_val_qry, self.ind_val_ans = self.load_query(self.ind_valid)
        self.ind_tst_qry, self.ind_tst_ans = self.load_query(self.ind_test)
        self.valid_q, self.valid_a = self.tra_val_qry, self.tra_val_ans
        self.test_q, self.test_a = self.ind_val_qry + self.ind_tst_qry, self.ind_val_ans + self.ind_tst_ans

        self.n_train = len(self.tra_train)
        self.n_valid = len(self.valid_q)
        self.n_test = len(self.test_q)

        print('n_train:', self.n_train, 'n_valid:', self.n_valid, 'n_test:', self.n_test)

    def read_triples(self, directory, filename, mode='transductive'):
        triples = []
        with open(os.path.join(directory, filename)) as f:
            for line in f:
                h, r, t = line.strip().split()
                if mode == 'transductive':
                    h, r, t = self.entity2id[h], self.relation2id[r], self.entity2id[t]
                else:
                    h, r, t = self.entity2id_ind[h], self.relation2id[r], self.entity2id_ind[t]

                triples.append([h, r, t])
                triples.append([t, r + self.n_rel, h])
        return triples

    def load_graph(self, triples, mode='transductive'):
        n_ent = self.n_ent if mode == 'transductive' else self.n_ent_ind

        KG = np.array(triples)
        idd = np.concatenate([np.expand_dims(np.arange(n_ent), 1), 2 * self.n_rel * np.ones((n_ent, 1)),
                              np.expand_dims(np.arange(n_ent), 1)], 1)
        KG = np.concatenate([KG, idd], 0)

        n_fact = KG.shape[0]

        M_sub = csr_matrix((np.ones((n_fact,)), (np.arange(n_fact), KG[:, 0])), shape=(n_fact, n_ent))
        return KG, M_sub

    def load_query(self, triples):
        triples.sort(key=lambda x: (x[0], x[1]))
        trip_hr = defaultdict(lambda: list())

        for trip in triples:
            h, r, t = trip
            trip_hr[(h, r)].append(t)

        queries = []
        answers = []
        for key in trip_hr:
            queries.append(key)
            answers.append(np.array(trip_hr[key]))
        return queries, answers

    def get_neighbors(self, nodes, mode='transductive'):
        # nodes: n_node x 2 with (batch_idx, node_idx)
        if mode == 'transductive':
            KG = self.tra_KG
            M_sub = self.tra_sub
            n_ent = self.n_ent
        else:
            KG = self.ind_KG
            M_sub = self.ind_sub
            n_ent = self.n_ent_ind

        node_1hot = csr_matrix((np.ones(len(nodes)), (nodes[:, 1], nodes[:, 0])), shape=(n_ent, nodes.shape[0]))
        edge_1hot = M_sub.dot(node_1hot)
        edges = np.nonzero(edge_1hot)
        selected_edges = np.concatenate([np.expand_dims(edges[1], 1), KG[edges[0]]],
                                        axis=1)  # (batch_idx, head, rela, tail)
        selected_edges = torch.LongTensor(selected_edges).cuda()

        # index to nodes
        head_nodes, head_index = torch.unique(selected_edges[:, [0, 1]], dim=0, sorted=True, return_inverse=True)
        tail_nodes, tail_index = torch.unique(selected_edges[:, [0, 3]], dim=0, sorted=True, return_inverse=True)
        mask = selected_edges[:, 2] == (self.n_rel * 2)
        _, old_idx = head_index[mask].sort()
        old_nodes_new_idx = tail_index[mask][old_idx]
        selected_edges = torch.cat([selected_edges, head_index.unsqueeze(1), tail_index.unsqueeze(1)], 1)

        return tail_nodes, selected_edges, old_nodes_new_idx

    def get_batch(self, batch_idx, steps=2, data='train'):
        if data == 'train':
            return self.tra_train[batch_idx]
        if data == 'valid':
            query, answer = np.array(self.valid_q), self.valid_a
            n_ent = self.n_ent
        if data == 'test':
            query, answer = np.array(self.test_q), self.test_a
            n_ent = self.n_ent_ind

        subs = []
        rels = []
        objs = []

        subs = query[batch_idx, 0]
        rels = query[batch_idx, 1]
        objs = np.zeros((len(batch_idx), n_ent))
        for i in range(len(batch_idx)):
            objs[i][answer[batch_idx[i]]] = 1
        return subs, rels, objs

    def shuffle_train(self, ):
        rand_idx = np.random.permutation(self.n_train)
        self.tra_train = self.tra_train[rand_idx]

    def get_filter(self, data='valid'):
        filters = defaultdict(lambda: set())
        if data == 'valid':
            for triple in self.tra_train:
                h, r, t = triple
                filters[(h, r)].add(t)
            for triple in self.tra_valid:
                h, r, t = triple
                filters[(h, r)].add(t)
            for triple in self.tra_test:
                h, r, t = triple
                filters[(h, r)].add(t)
        else:
            for triple in self.ind_train:
                h, r, t = triple
                filters[(h, r)].add(t)
            for triple in self.ind_valid:
                h, r, t = triple
                filters[(h, r)].add(t)
            for triple in self.ind_test:
                h, r, t = triple
                filters[(h, r)].add(t)
        return filters
