import networkx as nx
import random
import torch
from tqdm import tqdm
import numpy as np
from itertools import combinations


class HDESampler:
    """
    Sampler for HDE model, the function of this sampler is perform data preprocess
    and compute HDE for each node. Please notice that for different target node set,
    the HDE for each node can be different.
    For more details, please refer to the original paper: http://www.shichuan.org/doc/116.pdf
    """
    def __init__(self, that):
        self.mini_batch = []
        self.type2idx = that.type2idx
        self.node_type = that.node_type
        self.target_link = that.target_link
        self.num_fea = that.num_fea
        self.sample_size = that.sample_size
        self.hg = that.hg
        self.device = that.device

    def preprocess(self):
        self.preprocess_feature()

    def dgl2nx(self):
        """
        Convert the dgl graph data to networkx data structure.
        :return:
        """
        test_ratio = 0.1
        nx_g = nx.Graph()
        sp = 1 - test_ratio * 2

        edges = self.hg.edges(etype=self.target_link[0][1])
        a_list = edges[0]
        b_list = edges[1]
        edge_list = []
        for i in range(self.hg.num_edges(self.target_link[0][1])):
            edge_list.append(['A' + str(int(a_list[i])), 'B' + str(int(b_list[i]))])

        num_edge = len(edge_list)
        sp1 = int(num_edge * sp)
        sp2 = int(num_edge * test_ratio)
        self.g_train = nx.Graph()
        self.g_val = nx.Graph()
        self.g_test = nx.Graph()

        self.g_train.add_edges_from(edge_list[:sp1])
        self.g_val.add_edges_from(edge_list[sp1:sp1 + sp2])
        self.g_test.add_edges_from(edge_list[sp1 + sp2:])

    def compute_hde(self, args):
        """
        Compute hde for training set, validation set and test set.
        :param args: arguments
        :return:
        """
        print("Computing HDE for training set...")
        self.data_A_train, self.data_B_train, self.data_y_train = self.batch_data(self.g_train, args)
        print("Computing HDE for validation set...")
        self.data_A_val, self.data_B_val, self.data_y_val = self.batch_data(self.g_val, args)
        print("Computing HDE for test set...")
        self.data_A_test, self.data_B_test, self.data_y_test = self.batch_data(self.g_test, args)

        val_batch_A_fea = self.data_A_val.reshape(-1, self.sample_size)
        val_batch_B_fea = self.data_B_val.reshape(-1, self.sample_size)
        val_batch_y = self.data_y_val.reshape(-1)
        self.val_batch_A_fea = torch.FloatTensor(val_batch_A_fea).to(self.device)
        self.val_batch_B_fea = torch.FloatTensor(val_batch_B_fea).to(self.device)
        self.val_batch_y = torch.LongTensor(val_batch_y).to(self.device)
        test_batch_A_fea = self.data_A_test.reshape(-1, self.sample_size)
        test_batch_B_fea = self.data_B_test.reshape(-1, self.sample_size)
        test_batch_y = self.data_y_test.reshape(-1)
        self.test_batch_A_fea = torch.FloatTensor(test_batch_A_fea).to(self.device)
        self.test_batch_B_fea = torch.FloatTensor(test_batch_B_fea).to(self.device)
        self.test_batch_y = torch.LongTensor(test_batch_y).to(self.device)

        return self.data_A_train, self.data_B_train, self.data_y_train, \
               self.val_batch_A_fea, self.val_batch_B_fea, self.val_batch_y, \
               self.test_batch_A_fea, self.test_batch_B_fea, self.test_batch_y

    def batch_data(self, g, args):
        """
        Generate batch data.
        :param g: graph data
        :param args: arguments
        :return:
        """
        edge = list(g.edges)
        nodes = list(g.nodes)
        num_batch = int(len(edge) * 2 / args.batch_size)
        random.shuffle(edge)
        data = []
        edge = edge[0: num_batch * args.batch_size // 2]
        for bx in tqdm(edge):
            posA, posB = self.subgraph_sampling_with_DE_node_pair(g, bx, args)
            data.append([posA, posB, 1])

            neg_tmpB_id = random.choice(nodes)
            negA, negB = self.subgraph_sampling_with_DE_node_pair(g,
                                                                 [bx[0], neg_tmpB_id],
                                                                 args)
            data.append([negA, negB, 0])

        random.shuffle(data)
        data = np.array(data)
        data = data.reshape(num_batch, args.batch_size, 3)
        data_A = data[:, :, 0].tolist()
        data_B = data[:, :, 1].tolist()
        data_y = data[:, :, 2].tolist()
        for i in range(len(data_A)):
            for j in range(len(data[0])):
                data_A[i][j] = data_A[i][j].tolist()
                data_B[i][j] = data_B[i][j].tolist()
        data_A = np.squeeze(np.array(data_A))
        data_B = np.squeeze(np.array(data_B))
        data_y = np.squeeze(np.array(data_y))
        return data_A, data_B, data_y

    def subgraph_sampling_with_DE_node_pair(self,
                                            G,
                                            node_pair,
                                            args):
        """
        compute distance encoding given a target node set
        :param G: graph data
        :param node_pair: target node set
        :param args: arguments
        :return:
        """
        [A, B] = node_pair
        A_ego = nx.ego_graph(G, A, radius=args.k_hop)
        B_ego = nx.ego_graph(G, B, radius=args.k_hop)
        sub_G_for_AB = nx.compose(A_ego, B_ego)
        sub_G_for_AB.remove_edges_from(combinations(node_pair, 2))

        sub_G_nodes = sub_G_for_AB.nodes
        SPD_based_on_node_pair = {}
        for node in sub_G_nodes:
            tmpA = self.dist_encoder(A, node, sub_G_for_AB, args)
            tmpB = self.dist_encoder(B, node, sub_G_for_AB, args)
            SPD_based_on_node_pair[node] = np.concatenate([tmpA, tmpB], axis=0)

        A_fea_batch = self.gen_fea_batch(sub_G_for_AB,
                                            A,
                                            SPD_based_on_node_pair,
                                          args)
        B_fea_batch = self.gen_fea_batch(sub_G_for_AB,
                                            B,
                                            SPD_based_on_node_pair,
                                            args)
        return A_fea_batch, B_fea_batch

    def gen_fea_batch(self, G, root, fea_dict, args):
        """
        Neighbor sampling for each node. The Neighbor feature will be concatenated and
        will be separated in the model.
        :param G: graph data
        :param root: root node
        :param fea_dict: node features
        :param args: arguments
        :return:
        """
        fea_batch = []
        self.mini_batch.append([root])
        a = [0] * (self.num_fea - self.node_type) + self.type_encoder(root)
        fea_batch.append(np.asarray(a,
                                    dtype=np.float32
                                    ).reshape(-1, self.num_fea)
                         )

        ns_1 = [list(np.random.choice(list(G.neighbors(node)) + [node],
                                      args.num_neighbor,
                                      replace=True))
                for node in self.mini_batch[-1]]
        self.mini_batch.append(ns_1[0])
        de_1 = [
            np.concatenate([fea_dict[dest], np.asarray(self.type_encoder(dest))], axis=0)
            for dest in ns_1[0]
        ]

        fea_batch.append(np.asarray(de_1,
                                    dtype=np.float32).reshape(1, -1)
                         )
        # 2-order
        ns_2 = [list(np.random.choice(list(G.neighbors(node)) + [node],
                                      args.num_neighbor,
                                      replace=True))
                for node in self.mini_batch[-1]]
        de_2 = []
        for i in range(len(ns_2)):
            tmp = []
            for j in range(len(ns_2[0])):
                tmp.append(
                    np.concatenate(
                        [fea_dict[ns_2[i][j]], np.asarray(self.type_encoder(ns_2[i][j]))],
                        axis=0)
                )
            de_2.append(tmp)

        fea_batch.append(np.asarray(de_2,
                                    dtype=np.float32).reshape(1, -1)
                         )

        return np.concatenate(fea_batch, axis=1)

    def dist_encoder(self, src, dest, G, args):
        """
        compute H_SPD for a node pair
        :param src: source node
        :param dest: target node
        :param G: graph data
        :param args: arguments
        :return:
        """
        paths = list(nx.all_simple_paths(G, src, dest, cutoff=args.max_dist + 1))
        cnt = [args.max_dist] * self.node_type  # truncation SPD at max_spd
        for path in paths:
            res = [0] * self.node_type
            for i in range(1, len(path)):
                tmp = path[i][0]
                res[self.type2idx[tmp]] += 1
            # print(path, res)
            for k in range(self.node_type):
                cnt[k] = min(cnt[k], res[k])
        one_hot_list = [np.eye(args.max_dist + 1, dtype=np.float64)[cnt[i]]
                            for i in range(self.node_type)]
        return np.concatenate(one_hot_list)

    def type_encoder(self, node):
        """
        perform one-hot encoding based on the node type.
        :param node:
        :return:
        """
        res = [0] * self.node_type
        res[self.type2idx[node[0]]] = 1.0
        return res

