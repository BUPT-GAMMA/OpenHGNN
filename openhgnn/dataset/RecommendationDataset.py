import os
import dgl
import torch as th
import numpy as np
from . import BaseDataset, register_dataset
from dgl.data.utils import load_graphs
from scipy.sparse import csr_matrix
import scipy.sparse as sp
from .multigraph import MultiGraphDataset
from ..sampler.negative_sampler import Uniform_exclusive
from . import AcademicDataset


@register_dataset('recommendation')
class RecommendationDataset(BaseDataset):
    """

    """
    def __init__(self,*args, **kwargs):
        super(RecommendationDataset, self).__init__(*args, **kwargs)
        self.meta_paths_dict = None


@register_dataset('kgcn_recommendation')
class KGCN_Recommendation(RecommendationDataset):
    r"""
    Which is used in KGCN.
    """
    def __init__(self, dataset_name, *args, **kwargs):
            super(RecommendationDataset, self).__init__(*args, **kwargs)
            dataset = MultiGraphDataset(name=dataset_name, raw_dir='')
            self.g = dataset[0].long()
            self.g_1 = dataset[1].long()

    def get_split(self, validation=True):
        ratingsGraph = self.g_1
        n_edges = ratingsGraph.num_edges()
        random_int = th.randperm(n_edges)
        train_idx = random_int[:int(n_edges*0.6)]
        val_idx = random_int[int(n_edges*0.6):int(n_edges*0.8)]
        test_idx = random_int[int(n_edges*0.6):int(n_edges*0.8)]

        return train_idx, val_idx, test_idx
    
    def get_train_data(self):
        pass

    def get_labels(self):
        return self.label


@register_dataset('lightGCN_recommendation')
class lightGCN_Recommendation(RecommendationDataset):

    def __init__(self, dataset_name, *args, **kwargs):
        super(RecommendationDataset, self).__init__(*args, **kwargs)

        # train and test data
        self.mode_dict = {'train': 0, "test": 1}
        self.mode = self.mode_dict['train']
        self.n_user = 0
        self.m_item = 0
        path = './openhgnn/dataset/' + dataset_name
        train_file = path + '/train.txt'
        test_file = path + '/test.txt'
        self.path = path
        trainUniqueUsers, trainItem, trainUser = [], [], []
        testUniqueUsers, testItem, testUser = [], [], []
        self.traindataSize = 0
        self.testDataSize = 0

        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    trainUniqueUsers.append(uid)
                    trainUser.extend([uid] * len(items))
                    trainItem.extend(items)

                    self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self.traindataSize += len(items)
        self.trainUniqueUsers = np.array(trainUniqueUsers)
        self.trainUser = np.array(trainUser)
        self.trainItem = np.array(trainItem)

        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    testUniqueUsers.append(uid)
                    testUser.extend([uid] * len(items))
                    testItem.extend(items)
                    self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self.testDataSize += len(items)
        self.m_item += 1
        self.n_user += 1
        self.testUniqueUsers = np.array(testUniqueUsers)
        self.testUser = np.array(testUser)
        self.testItem = np.array(testItem)

        self.Graph = None

        # (users,items), bipartite graph
        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
                                      shape=(self.n_user, self.m_item))
        self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()
        self.users_D[self.users_D == 0.] = 1
        self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        self.items_D[self.items_D == 0.] = 1.
        # pre-calculate
        self.allPos = self.getUserPosItems(list(range(self.n_user)))
        self.testDict = self.__build_test()

        self.g = self.getSparseGraph()

    def get_split(self):
        return self.g, [], []

    def __build_test(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = th.Tensor(coo.row).long()
        col = th.Tensor(coo.col).long()
        index = th.stack([row, col])
        data = th.FloatTensor(coo.data)
        return th.sparse.FloatTensor(index, data, th.Size(coo.shape))

    def getSparseGraph(self):
        print("loading adjacency matrix")
        if self.Graph is None:
            try:
                pre_adj_mat = sp.load_npz(self.path + '/s_pre_adj_mat.npz')
                print("successfully loaded...")
                norm_adj = pre_adj_mat
            except:
                print("generating adjacency matrix")
                # s = time()
                adj_mat = sp.dok_matrix((self.n_user + self.m_item, self.n_user + self.m_item), dtype=np.float32)
                adj_mat = adj_mat.tolil()
                R = self.UserItemNet.tolil()
                adj_mat[:self.n_user, self.n_user:] = R
                adj_mat[self.n_user:, :self.n_user] = R.T
                adj_mat = adj_mat.todok()
                # adj_mat = adj_mat + sp.eye(adj_mat.shape[0])

                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)

                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                # end = time()
                # print(f"costing {end - s}s, saved norm_mat...")
                sp.save_npz(self.path + '/s_pre_adj_mat.npz', norm_adj)

            # if self.split == True:
            #     self.Graph = self._split_A_hat(norm_adj)
            #     print("done split matrix")
            # else:
            self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
            # self.Graph = self.Graph.coalesce().to(self.device)
            self.Graph = self.Graph.coalesce()
            print("don't split the matrix")
        return self.Graph


@register_dataset('hin_recommendation')
class HINRecommendation(RecommendationDataset):
    def __init__(self, dataset_name, *args, **kwargs):
        super(HINRecommendation, self).__init__(*args, **kwargs)
        self.dataset_name = dataset_name
        self.num_neg = 20
        #self.neg_dir = os.path.join(self.raw_dir, dataset_name, 'neg_{}.bin'.format(self.num_neg))
        if dataset_name == 'yelp4rec':
            dataset = AcademicDataset(name='yelp4rec', raw_dir='')
            self.g = dataset[0].long()
            self.target_link = 'user-item'
            self.target_link_r = 'item-user'
            self.user_name = 'user'
            self.item_name = 'item'

        elif dataset_name == 'yelp4HeGAN':
            dataset = AcademicDataset(name='yelp4HeGAN', raw_dir='')
            self.g = dataset[0].long()
            self.target_link = 'usb'
            self.target_link_r = 'bus'
            self.user_name = 'user'
            self.item_name = 'business'

        elif dataset_name == 'DoubanMovie':
            dataset = AcademicDataset(name='DoubanMovie', raw_dir='')
            self.g = dataset[0].long()
            self.target_link = 'user-item'
            self.target_link_r = 'item-user'
            self.user_name = 'user'
            self.item_name = 'item'

        elif dataset_name == 'amazon':
            data_path = './openhgnn/dataset/amazon_graph.bin'
            dataset = AcademicDataset(name='amazon', raw_dir='')
            self.g = dataset[0].long()
            self.target_link = 'ui'
            self.target_link_r = 'iu'
            self.user_name = 'user'
            self.item_name = 'item'

        self.out_ntypes = [self.user_name, self.item_name]
        # self.process()
        # self.neg_g = self.construct_negative_graph(self.g)

    def load_HIN(self, dataset_name):
        g, _ = dgl.load_graphs(dataset_name)
        return g[0]

    # def process(self, g):
    #     # sub 1 for every node
    #     new = {}
    #     for etype in g.canonical_etypes:
    #         edges = g.edges(etype=etype)
    #         new[etype] = (edges[0]-1, edges[1]-1)
    #     hg = dgl.heterograph(new)
    #     hg.edata['val_mask'] = g.edata['val_mask']
    #     hg.edata['test_mask'] = g.edata['test_mask']
    #     hg.edata['train_mask'] = g.edata['train_mask']
    #     from dgl.data.utils import save_graphs
    #     save_graphs(f"./openhgnn/dataset/{self.dataset_name}.bin", hg)

    def get_split(self, validation=True):
        test_mask = self.g.edges[self.target_link].data['test_mask'].squeeze()
        test_index = th.nonzero(test_mask).squeeze()
        test_edge = self.g.find_edges(test_index, self.target_link)

        test_graph = dgl.heterograph({(self.user_name, self.target_link, self.item_name): test_edge},
                                     {ntype: self.g.number_of_nodes(ntype) for ntype in self.out_ntypes})
        if validation:
            val_mask = self.g.edges[self.target_link].data['val_mask'].squeeze()
            val_index = th.nonzero(val_mask).squeeze()
            val_edge = self.g.find_edges(val_index, self.target_link)

            val_graph = dgl.heterograph({(self.user_name, self.target_link, self.item_name): val_edge},
                                             {ntype: self.g.number_of_nodes(ntype) for ntype in self.out_ntypes})

            train_graph = dgl.remove_edges(self.g, th.cat((val_index, test_index)), self.target_link)
            train_graph = dgl.remove_edges(train_graph, th.cat((val_index, test_index)), self.target_link_r)
        else:
            train_graph = dgl.remove_edges(self.g, test_index, self.target_link)
            train_graph = dgl.remove_edges(train_graph, test_index, self.target_link_r)
            val_graph = train_graph

        return train_graph, val_graph, test_graph

    def construct_negative_graph(self, train_g):
        fname = f'./openhgnn/dataset/{self.dataset_name}/neg_graph_{self.num_neg}.bin'
        if os.path.exists(fname):
            g, _ = load_graphs(fname)
            return g[0]
        else:
            k = self.num_neg
            negative_sampler = Uniform_exclusive(k)
            negative_edges = negative_sampler(train_g.to('cpu'), {
                self.target_link: th.arange(train_g.num_edges(self.target_link))})
            # negative_edges = negative_sampler(train_g.to('cpu'), {
            #     self.target_link: th.arange(10)})
            neg_g = dgl.heterograph(negative_edges,
                                    {ntype: self.g.number_of_nodes(ntype) for ntype in self.out_ntypes})
            dgl.save_graphs(fname, neg_g)
            return neg_g


@register_dataset('test_link_prediction')
class Test_Recommendation(RecommendationDataset):
    def __init__(self, dataset_name):
        super(RecommendationDataset, self).__init__()
        self.g = self.load_HIN('./openhgnn/debug/data.bin')
        self.target_link = 'user-item'
        self.has_feature = False
        self.preprocess()
        #self.generate_negative()

    def load_HIN(self, dataset_name):
        g, _ = load_graphs(dataset_name)
        return g[0]

    def preprocess(self):
        test_mask = self.g.edges[self.target_link].data['test_mask']
        index = th.nonzero(test_mask).squeeze()
        self.test_edge = self.g.find_edges(index, self.target_link)
        self.pos_test_graph = dgl.heterograph({('user', 'user-item', 'item'): self.test_edge}, {ntype: self.g.number_of_nodes(ntype) for ntype in ['user', 'item']})
        self.g.remove_edges(index, self.target_link)
        self.g.remove_edges(index, 'item-user')
        self.neg_test_graph, _ = dgl.load_graphs('./openhgnn/debug/neg.bin')
        self.neg_test_graph = self.neg_test_graph[0]
        return

        negative_sampler = Uniform_exclusive(99)
        self.negative_g = negative_sampler(self.hg.to('cpu'), {self.target_link: th.arange(self.hg.num_edges(self.target_link))})
    def generate_negative(self):
        k = 99
        e = self.pos_test_graph.edges()
        neg_src = []
        neg_dst = []
        for i in range(self.pos_test_graph.number_of_edges()):
            src = e[0][i]
            exp = self.pos_test_graph.successors(src)
            dst = th.randint(high=self.g.number_of_nodes('item'), size=(k,))
            for d in range(len(dst)):
                while dst[d] in exp:
                    dst[d] = th.randint(high=self.g.number_of_nodes('item'), size=(1,))
            src = src.repeat_interleave(k)
            neg_src.append(src)
            neg_dst.append(dst)
        neg_edge = (th.cat(neg_src), th.cat(neg_dst))
        neg_graph = dgl.heterograph({('user', 'user-item', 'item'): neg_edge}, {ntype: self.g.number_of_nodes(ntype) for ntype in ['user', 'item']})
        dgl.save_graphs('./openhgnn/debug/neg.bin', neg_graph)
