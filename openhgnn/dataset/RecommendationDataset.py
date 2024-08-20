import os
import dgl
import torch as th
import numpy as np
from . import BaseDataset, register_dataset
from dgl.data.utils import download,load_graphs,save_graphs,save_info,load_info
from scipy.sparse import csr_matrix
import scipy.sparse as sp
from .multigraph import MultiGraphDataset
from ..sampler.negative_sampler import Uniform_exclusive
from . import AcademicDataset
from .HGCLDataset import HGCLDataset
from collections import defaultdict

#add more lib for KGAT
import time
import collections
import zipfile
import numpy as np
import pandas as pd

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

@register_dataset('hgcl_recommendation')
class HGCLRecommendation(RecommendationDataset):
    def __init__(self, dataset_name, *args, **kwargs):
        super(RecommendationDataset, self).__init__(*args, **kwargs)
        dataset = HGCLDataset(name=dataset_name, raw_dir='')
        self.g = dataset[0].long()

    def get_split(self, validation=True):
        ratingsGraph = self.g
        n_edges = ratingsGraph.num_edges()
        random_int = th.randperm(n_edges)
        train_idx = random_int[:int(n_edges * 0.6)]
        val_idx = random_int[int(n_edges * 0.6):int(n_edges * 0.8)]
        test_idx = random_int[int(n_edges * 0.6):int(n_edges * 0.8)]

        return train_idx, val_idx, test_idx

    def get_train_data(self):
        pass

    def get_labels(self):
        return self.label
        
@register_dataset('lightGCN_recommendation')
class lightGCN_Recommendation(RecommendationDataset):

    def __init__(self, dataset_name, *args, **kwargs):
        super(RecommendationDataset, self).__init__(*args, **kwargs)

        if dataset_name not in ['gowalla','yelp2018','amazon-book']:
            raise KeyError('Dataset {} is not supported!'.format(dataset_name))
        self.dataset_name=dataset_name

        self.data_path=f'openhgnn/dataset/{self.dataset_name}'

        if not os.path.exists(f"{self.data_path}/train.txt"):
            self.download()

        # test
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
    
    def download(self):
        prefix = 'https://raw.githubusercontent.com/gusye1234/LightGCN-PyTorch/master/data'

        required_file = ['train.txt', 'test.txt']

        for filename in required_file:
            url = f"{prefix}/{self.dataset_name}/{filename}"
            file_path = f"{self.data_path}/{filename}"
            if not os.path.exists(file_path):
                try:
                    download(url, file_path)

                except BaseException as e:
                    print("\n",e)
                    print("\nNote!   --- If you want to download the file, vpn is required ---")
                    print("If you don't have a vpn, please download the dataset from here: https://github.com/gusye1234/LightGCN-PyTorch")
                    print("\nAfter downloading the dataset, you need to store the files in the following path: ")
                    print(f"{os.getcwd()}\openhgnn\dataset\{self.dataset_name}\\train.txt")
                    print(f"{os.getcwd()}\openhgnn\dataset\{self.dataset_name}\\test.txt")
                    exit()


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

@register_dataset('kgat_recommendation')
class KGAT_recommendation(RecommendationDataset):

    def __init__(self,dataset_name,*args, **kwargs):
        super(KGAT_recommendation, self).__init__(*args, **kwargs)
        if dataset_name not in ['yelp2018','amazon-book','last-fm']:
            raise KeyError('Dataset {} is not supported!'.format(dataset_name))
        self.dataset_name=dataset_name

        self.data_path=f'openhgnn/dataset/{self.dataset_name}'
        if self.dataset_name=='yelp2018':
            self.processed_data_path = 'openhgnn/dataset/KGAT_yelp2018'
        elif self.dataset_name=='amazon-book':
            self.processed_data_path = 'openhgnn/dataset/KGAT_amazon-book'
        else:
            self.processed_data_path = 'openhgnn/dataset/KGAT_last-fm'

        if os.path.exists(f"{self.processed_data_path}/graph.bin"):
            if os.path.exists(f"{self.processed_data_path}/other_info.pkl"):
                self.load()
        else:
            self.download()
            self.preprocess()
            self.save()

    def download(self):
        prefix = 'https://raw.githubusercontent.com/xiangwang1223/knowledge_graph_attention_network/master/Data'
        if self.dataset_name =='last-fm':
            required_file = ['train.txt', 'test.txt', 'kg_final.txt']
        else:
            required_file = ['train.txt', 'test.txt', 'kg_final.txt.zip']

        for filename in required_file:
            url = f"{prefix}/{self.dataset_name}/{filename}"
            file_path = f"{self.data_path}/{filename}"
            if not os.path.exists(file_path):
                try:
                    download(url,file_path)
                    if filename == 'kg_final.txt.zip':
                        zip_file = zipfile.ZipFile(file_path)
                        zip_file.extractall(f"{self.data_path}")
                        zip_file.close()
                except BaseException as e:
                    print("\n",e)
                    print("\nNote!   --- If you want to download the file, vpn is required ---")
                    print("If you don't have a vpn, please download the dataset from here: https://github.com/xiangwang1223/knowledge_graph_attention_network/")
                    print("\nAfter downloading the dataset, you need to store the files in the following path: ")
                    print(f"{os.getcwd()}\openhgnn\dataset\{self.dataset_name}\\train.txt")
                    print(f"{os.getcwd()}\openhgnn\dataset\{self.dataset_name}\\test.txt")
                    print(f"{os.getcwd()}\openhgnn\dataset\{self.dataset_name}\kg_final.txt")
                    print(f"{os.getcwd()}\openhgnn\dataset\{self.dataset_name}\pretrain\mf.npz")
                    exit()

        #load pretrain file
        prefix=("https://raw.githubusercontent.com/xiangwang1223/knowledge_graph_attention_network/master/Model/pretrain")
        pretrain_url=f"{prefix}/{self.dataset_name}/mf.npz"
        self.pretrain_embedding_dir=f"{self.data_path}/pretrain/mf.npz"
        if not os.path.exists(self.pretrain_embedding_dir):
            download(pretrain_url,self.pretrain_embedding_dir)
        print(time.strftime("%Y-%m-%d %H:%M:%S ", time.localtime()), '--- All files have been downloaded! ---')

    def preprocess(self):

        print(time.strftime("%Y-%m-%d %H:%M:%S ", time.localtime()), '--- Start Processing! ---')

        # get data_file_path
        train_file = f"{self.data_path}/train.txt"
        test_file = f"{self.data_path}/test.txt"
        kg_file = f"{self.data_path}/kg_final.txt"

        self.cf_train_data, self.train_user_dict = self.load_cf(train_file)
        self.cf_test_data, self.test_user_dict = self.load_cf(test_file)
        self.statistic_cf()
        print(time.strftime("%Y-%m-%d %H:%M:%S ", time.localtime()), '--- cf data finish ---')

        kg_data = self.load_kg(kg_file)
        print(time.strftime("%Y-%m-%d %H:%M:%S ", time.localtime()), '--- kg data load ---')

        self.construct_data(kg_data)
        self.g = self.create_graph(self.kg_train_data, self.n_users_entities)
        print(time.strftime("%Y-%m-%d %H:%M:%S ", time.localtime()), '--- kg data finish ---')
        print(time.strftime("%Y-%m-%d %H:%M:%S ", time.localtime()), '--- Done! ---')

    def load_cf(self, filename):
        '''
        Create user-item bipartite graph
        '''
        user = []
        item = []
        user_dict = dict()
        lines = open(filename, 'r').readlines()
        for l in lines:
            tmp = l.strip()
            inter = [int(i) for i in tmp.split()]

            if len(inter) > 1:
                user_id, item_ids = inter[0], inter[1:]
                item_ids = list(set(item_ids))
                for item_id in item_ids:
                    user.append(user_id)
                    item.append(item_id)
                user_dict[user_id] = item_ids

        user = np.array(user, dtype=np.int32)
        item = np.array(item, dtype=np.int32)
        return (user, item), user_dict

    def statistic_cf(self):
        '''
        Some statistic information of the user-item bipartite graph
        '''
        self.n_users = max(max(self.cf_train_data[0]), max(self.cf_test_data[0])) + 1
        self.n_items = max(max(self.cf_train_data[1]), max(self.cf_test_data[1])) + 1
        self.n_cf_train = len(self.cf_train_data[0])
        self.n_cf_test = len(self.cf_test_data[0])

    def load_kg(self, filename):
        '''
        Load knowledge graph data
        '''
        kg_data = pd.read_csv(filename, sep=' ', names=['h', 'r', 't'], engine='python')
        kg_data = kg_data.drop_duplicates()
        return kg_data

    def construct_data(self, kg_data):
        '''
        Alignment of knowledge graph and the user-item bipartite graph
        '''
        n_relations = max(kg_data['r']) + 1
        reverse_kg_data = kg_data.copy()
        reverse_kg_data = reverse_kg_data.rename({'h': 't', 't': 'h'}, axis='columns')
        reverse_kg_data['r'] += n_relations
        kg_data = pd.concat([kg_data, reverse_kg_data], axis=0, ignore_index=True, sort=False)
        kg_data['r'] += 2
        self.n_relations = max(kg_data['r']) + 1
        self.n_entities = max(max(kg_data['h']), max(kg_data['t'])) +1
        self.n_users_entities = self.n_users + self.n_entities

        self.cf_train_data = \
            (np.array(list(map(lambda d: d + self.n_entities, self.cf_train_data[0]))).astype(np.int32), self.cf_train_data[1].astype(np.int32))
        self.cf_test_data = \
            (np.array(list(map(lambda d: d + self.n_entities, self.cf_test_data[0]))).astype(np.int32), self.cf_test_data[1].astype(np.int32))

        self.train_user_dict = {k + self.n_entities: np.unique(v).astype(np.int32) for k, v in self.train_user_dict.items()}
        self.test_user_dict = {k + self.n_entities: np.unique(v).astype(np.int32) for k, v in self.test_user_dict.items()}

        cf2kg_train_data = pd.DataFrame(np.zeros((self.n_cf_train, 3), dtype=np.int32), columns=['h', 'r', 't'])
        cf2kg_train_data['h'] = self.cf_train_data[0]
        cf2kg_train_data['t'] = self.cf_train_data[1]

        reverse_cf2kg_train_data = pd.DataFrame(np.ones((self.n_cf_train, 3), dtype=np.int32), columns=['h', 'r', 't'])
        reverse_cf2kg_train_data['h'] = self.cf_train_data[1]
        reverse_cf2kg_train_data['t'] = self.cf_train_data[0]

        self.kg_train_data = pd.concat([kg_data, cf2kg_train_data, reverse_cf2kg_train_data], ignore_index=True)

        self.n_kg_train = len(self.kg_train_data)

        # construct kg dict
        self.train_kg_dict = collections.defaultdict(list)
        for row in self.kg_train_data.iterrows():
            h, r, t = row[1]
            self.train_kg_dict[h].append((t, r))

    def create_graph(self, kg_data, n_nodes):
        '''
        Create DGLgraph
        '''
        g = dgl.graph((kg_data['t'], kg_data['h']))
        g.ndata['id'] = th.arange(n_nodes, dtype=th.long)
        g.edata['type'] = th.LongTensor(kg_data['r'])
        return g

    def get_split(self):
        '''
        Make it compatible to recommendation task
        '''
        return None,None,None

    def get_labels(self):
        '''
        Make it compatible to recommendation task
        '''
        return None

    def save(self):
        '''
        Store all the information for training
        '''
        #save a dictionary
        print(f'storing graph and other information in {self.processed_data_path}')
        info_dict={'n_users':self.n_users,'n_entities':self.n_entities,'n_relations':self.n_relations,'n_items':self.n_items,
                   'train_user_dict':self.train_user_dict,'test_user_dict':self.test_user_dict,'train_kg_dict':self.train_kg_dict,
                   'n_cf_train':self.n_cf_train,'n_kg_train':self.n_kg_train}
        os.makedirs(os.path.dirname(f"{self.processed_data_path}/other_info.pkl"),exist_ok=True)
        save_info(f"{self.processed_data_path}/other_info.pkl",info_dict)
        save_graphs(f"{self.processed_data_path}/graph.bin",[self.g])

    def load(self):
        '''
        Store all the information for training
        '''
        #load information from a dictionary
        print(f'loading data from {self.processed_data_path}')
        new_dict=load_info(f"{self.processed_data_path}/other_info.pkl")

        self.train_user_dict=new_dict['train_user_dict']
        self.test_user_dict=new_dict['test_user_dict']
        self.train_kg_dict = new_dict['train_kg_dict']

        self.n_users=new_dict['n_users']
        self.n_entities=new_dict['n_entities']
        self.n_relations=new_dict['n_relations']
        self.n_items=new_dict['n_items']

        self.n_cf_train=new_dict['n_cf_train']
        self.n_kg_train =new_dict['n_kg_train']

        self.g=load_graphs(f"{self.processed_data_path}/graph.bin")
        self.g=self.g[0][0]

        self.pretrain_embedding_dir = f"{self.data_path}/pretrain/mf.npz"

        print(time.strftime("%Y-%m-%d %H:%M:%S ", time.localtime()), '--- Start Training! ---')

@register_dataset("sian_recommendation")
class SIAN_recommendation(RecommendationDataset):
    def __init__(self, *args, **kwargs):
        super(SIAN_recommendation, self).__init__(*args, **kwargs)
        self.data_path = f'openhgnn/dataset/SIAN_recommendation_dataset/'
        self.u_items = defaultdict(set)
        self.i_users = defaultdict(set)
        self.data = list() # train, test, val

        if not os.path.exists(self.data_path):
            self.download()

        self.load_data()
        self.load_profile()
        self.load_social_data()

    def get_split(self):
        '''
        Make it compatible to recommendation task
        '''
        return None,None,None

    def load_data(self):
        need = ['yelp.train.rating.712', 'yelp.test.rating.712', 'yelp.val.rating.712']
        for file in need:
            with open(self.data_path + file) as f:
                data_size = 0
                data = defaultdict(int)
                for line in f:
                    token = line.split('\t')  # user_id \t item_id \t label \t act_list
                    user = int(token[0])
                    item = int(token[1])
                    label = int(token[2])
                    act = token[3].strip()
                    data[(user, item, label, act)] = 1
                    if label == 1:  # positive
                        self.u_items[user].add(item)
                        self.i_users[item].add(user)
                    data_size += 1

            idx2user = np.zeros((data_size,), dtype=np.int32)
            idx2item = np.zeros((data_size,), dtype=np.int32)
            idx2label = np.zeros((data_size,), dtype=np.int32)
            idx2act = np.zeros((data_size,), dtype=object)

            for idx, (u, i, l, a) in enumerate(data):
                idx2user[idx] = u
                idx2item[idx] = i
                idx2label[idx] = l
                idx2act[idx] = list(map(lambda x: int(x), a.split(' ')))

            self.data.append(np.array([idx2user, idx2item, idx2label, idx2act]))

    def load_profile(self):
        self.user_profile = th.from_numpy(np.load(self.data_path + 'user_profile.npy')).float()
        self.item_profile = th.from_numpy(np.load(self.data_path + 'item_profile.npy')).float()

    def load_social_data(self):
        social_graph_filename = self.data_path + 'yelp.social.graph'
        self.social_relation = defaultdict(list)
        with open(social_graph_filename) as f:
            for line in f:
                token = line.split('\t')  # user_id \t user_id
                self.social_relation[int(token[0])].append(int(token[1]))
                self.social_relation[int(token[1])].append(int(token[0]))


    def download(self):
        prefix = 'https://raw.githubusercontent.com/rootlu/SIAN/master/data/yelp'
        required_file = ['item_profile.npy', 'user_profile.npy', 'yelp.test.rating.712', 'yelp.train.rating.712', 'yelp.val.rating.712', 'yelp.social.graph']

        for filename in required_file:
            url = f"{prefix}/{filename}"
            file_path = self.data_path + filename
            if not os.path.exists(file_path):
                try:
                    download(url,file_path)
                except BaseException as e:
                    print("\n",e)
                    print("\nNote!   --- If you want to download the file, vpn is required ---")
                    print("If you don't have a vpn, please download the dataset from here: https://github.com/rootlu/SIAN/tree/master/data/yelp")
                    exit()
