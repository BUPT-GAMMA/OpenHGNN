import numpy as np
import random
import torch
import dgl
import time
import os
import igraph
import requests
import zipfile
import io


def remove_duplicate(x):
    return list(dict.fromkeys(x))


# 定义一个并查集类
class UnionFind:
    # 初始化
    def __init__(self, n):
        # n表示节点的个数
        self.n = n
        # parent表示每个节点的父节点
        self.parent = list(range(n))

    # 查找x所属的集合，即根节点
    def find(self, x):
        # 如果x不是根节点，就递归查找，并进行路径压缩
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    # 合并x和y所属的集合
    def union(self, x, y):
        # 把x所属的集合的根节点作为y所属的集合的根节点的子节点
        self.parent[self.find(x)] = self.find(y)

    # 判断x和y是否属于同一个集合
    def connected(self, x, y):
        return self.find(x) == self.find(y)


def kruskal(g):
    # 初始化一个并查集，节点个数为图的节点个数
    uf = UnionFind(g.num_nodes())
    # 初始化一个空列表，用于保存最小生成树的边
    mst_edges = []
    # 初始化一个空列表，用于保存最小生成树的边的权重
    mst_weights = []
    # 初始化一个空列表，用于保存边的id
    edge_index = []
    # 获取图的所有边和对应的权重
    edges, weights = g.edges(), g.edata['w']
    # 按照权重从小到大对边进行排序，返回排序后的索引
    indices = torch.argsort(weights)
    # 遍历排序后的索引
    for i in indices:
        # 获取当前索引对应的边和权重
        u, v = edges[0][i], edges[1][i]
        # 判断当前边的两个端点是否已经连通
        if not uf.connected(u, v):
            # 如果没有连通，就把当前边加入到最小生成树中
            mst_edges.append((u, v))
            edge_index.append(int(i))
            # 同时更新并查集
            uf.union(u, v)
            # 如果最小生成树的边的数量等于节点数量减一，就提前结束循环
            if len(mst_edges) == g.num_nodes() - 1:
                break
    mst_g = dgl.graph(mst_edges)
    # 将edge_index转换成tensor
    edge_index = torch.tensor(edge_index)
    return mst_g, edge_index


class Ingram_KG_TrainData():
    def __init__(self, path, dataset_name, *args, **kwargs):
        super(Ingram_KG_TrainData, self).__init__(*args, **kwargs)
        # 上线的时候要更改
        self.path = 'openhgnn/data/' + dataset_name + '/'
        self.rel_info = {}  # (h,t):[r1,r2,...]
        self.pair_info = {}  # r:[(h,t),(h,t),...]
        self.spanning = []  # [(h,t),(h,t),...]，代表着spanning tree的边
        self.remaining = []  # [(h,t),(h,t),...]，代表着剩余的边
        self.ent2id = None  # ent2id
        self.rel2id = None  # rel2id
        self.id2ent, self.id2rel, self.triplets = self.read_triplet(self.path + 'train.txt')
        self.num_triplets = len(self.triplets)
        self.num_ent, self.num_rel = len(self.id2ent), len(self.id2rel)
        self.dataset_name = dataset_name

    def read_triplet(self, path):


        path_ckp = self.path
        print(path_ckp)
        folder = os.path.exists(path_ckp)
        if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
            os.makedirs(path_ckp)  # makedirs 创建文件时如果路径不存在会创建这个路径
            # 下载数据
            url = "https://s3.cn-north-1.amazonaws.com.cn/dgl-data/dataset/openhgnn/NL-100.zip"
            response = requests.get(url)
            with zipfile.ZipFile(io.BytesIO(response.content)) as myzip:
                myzip.extractall(self.path)
            print("---  download data  ---")

        else:
            print("---  There is data!  ---")


        id2ent, id2rel, triplets = [], [], []
        with open(path, 'r') as f:
            for line in f.readlines():
                h, r, t = line.strip().split('\t')
                id2ent.append(h)
                id2ent.append(t)
                id2rel.append(r)
                triplets.append((h, r, t))
        id2ent = remove_duplicate(id2ent)
        id2rel = remove_duplicate(id2rel)
        self.ent2id = {ent: idx for idx, ent in enumerate(id2ent)}
        self.rel2id = {rel: idx for idx, rel in enumerate(id2rel)}
        triplets = [(self.ent2id[h], self.rel2id[r], self.ent2id[t]) for h, r, t in triplets]
        for (h, r, t) in triplets:
            if (h, t) in self.rel_info:
                self.rel_info[(h, t)].append(r)
            else:
                self.rel_info[(h, t)] = [r]
            if r in self.pair_info:
                self.pair_info[r].append((h, t))
            else:
                self.pair_info[r] = [(h, t)]
        G = igraph.Graph.TupleList(np.array(triplets)[:, 0::2])
        G_ent = igraph.Graph.TupleList(np.array(triplets)[:, 0::2], directed=True)
        spanning = G_ent.spanning_tree()
        G_ent.delete_edges(spanning.get_edgelist())
        print(spanning.es)
        for e in spanning.es:
            e1, e2 = e.tuple
            e1 = spanning.vs[e1]["name"]
            e2 = spanning.vs[e2]["name"]
            self.spanning.append((e1, e2))

        spanning_set = set(self.spanning)

        print("-----Train Data Statistics-----")
        print(f"{len(self.ent2id)} entities, {len(self.rel2id)} relations")
        print(f"{len(triplets)} triplets")
        self.triplet2idx = {triplet: idx for idx, triplet in enumerate(triplets)}
        self.triplets_with_inv = np.array([(t, r + len(id2rel), h) for h, r, t in triplets] + triplets)
        return id2ent, id2rel, triplets

    def split_transductive(self, p):
        msg, sup = [], []  # 训练姐和测试集
        rels_encountered = np.zeros(self.num_rel)

        remaining_triplet_indexes = np.ones(self.num_triplets)

        for h, t in self.spanning:
            r = random.choice(self.rel_info[(h, t)])
            msg.append((h, r, t))
            remaining_triplet_indexes[self.triplet2idx[(h, r, t)]] = 0
            rels_encountered[r] = 1
        for r in (1 - rels_encountered).nonzero()[0].tolist():
            h, t = random.choice(self.pair_info[int(r)])
            msg.append((h, r, t))
            remaining_triplet_indexes[self.triplet2idx[(h, r, t)]] = 0

        start = time.time()
        sup = [self.triplets[idx] for idx, tf in enumerate(remaining_triplet_indexes) if tf]
        msg = np.array(msg)
        random.shuffle(sup)
        sup = np.array(sup)
        add_num = max(int(self.num_triplets * p) - len(msg), 0)
        msg = np.concatenate([msg, sup[:add_num]])
        sup = sup[add_num:]

        msg_inv = np.fliplr(msg).copy()
        msg_inv[:, 1] += self.num_rel
        msg = np.concatenate([msg, msg_inv])

        return msg, sup


class Ingram_KG_TestData():
    def __init__(self, path, dataset_name, data_type="valid"):
        self.path = 'openhgnn/data/' + dataset_name + '/'
        self.data_type = data_type
        self.ent2id = None
        self.rel2id = None
        self.id2ent, self.id2rel, self.msg_triplets, self.sup_triplets, self.filter_dict = self.read_triplet()
        self.num_ent, self.num_rel = len(self.id2ent), len(self.id2rel)

    def read_triplet(self):
        id2ent, id2rel, msg_triplets, sup_triplets = [], [], [], []
        total_triplets = []


        with open(self.path + "msg.txt", 'r') as f:
            for line in f.readlines():
                h, r, t = line.strip().split('\t')
                id2ent.append(h)
                id2ent.append(t)
                id2rel.append(r)
                msg_triplets.append((h, r, t))
                total_triplets.append((h, r, t))

        id2ent = remove_duplicate(id2ent)
        id2rel = remove_duplicate(id2rel)
        self.ent2id = {ent: idx for idx, ent in enumerate(id2ent)}
        self.rel2id = {rel: idx for idx, rel in enumerate(id2rel)}
        num_rel = len(self.rel2id)
        msg_triplets = [(self.ent2id[h], self.rel2id[r], self.ent2id[t]) for h, r, t in msg_triplets]
        msg_inv_triplets = [(t, r + num_rel, h) for h, r, t in msg_triplets]

        with open(self.path + self.data_type + ".txt", 'r') as f:
            for line in f.readlines():
                h, r, t = line.strip().split('\t')
                sup_triplets.append((self.ent2id[h], self.rel2id[r], self.ent2id[t]))
                assert (self.ent2id[h], self.rel2id[r], self.ent2id[t]) not in msg_triplets, \
                    (self.ent2id[h], self.rel2id[r], self.ent2id[t])
                total_triplets.append((h, r, t))
        for data_type in ['valid', 'test']:
            if data_type == self.data_type:
                continue
            with open(self.path + data_type + ".txt", 'r') as f:
                for line in f.readlines():
                    h, r, t = line.strip().split('\t')
                    assert (self.ent2id[h], self.rel2id[r], self.ent2id[t]) not in msg_triplets, \
                        (self.ent2id[h], self.rel2id[r], self.ent2id[t])
                    total_triplets.append((h, r, t))

        filter_dict = {}
        for triplet in total_triplets:
            h, r, t = triplet
            if ('_', self.rel2id[r], self.ent2id[t]) not in filter_dict:
                filter_dict[('_', self.rel2id[r], self.ent2id[t])] = [self.ent2id[h]]
            else:
                filter_dict[('_', self.rel2id[r], self.ent2id[t])].append(self.ent2id[h])

            if (self.ent2id[h], '_', self.ent2id[t]) not in filter_dict:
                filter_dict[(self.ent2id[h], '_', self.ent2id[t])] = [self.rel2id[r]]
            else:
                filter_dict[(self.ent2id[h], '_', self.ent2id[t])].append(self.rel2id[r])

            if (self.ent2id[h], self.rel2id[r], '_') not in filter_dict:
                filter_dict[(self.ent2id[h], self.rel2id[r], '_')] = [self.ent2id[t]]
            else:
                filter_dict[(self.ent2id[h], self.rel2id[r], '_')].append(self.ent2id[t])

        print(f"-----{self.data_type.capitalize()} Data Statistics-----")
        print(f"Message set has {len(msg_triplets)} triplets")
        print(f"Supervision set has {len(sup_triplets)} triplets")
        print(f"{len(self.ent2id)} entities, " + \
              f"{len(self.rel2id)} relations, " + \
              f"{len(total_triplets)} triplets")

        msg_triplets = msg_triplets + msg_inv_triplets

        return id2ent, id2rel, np.array(msg_triplets), np.array(sup_triplets), filter_dict
