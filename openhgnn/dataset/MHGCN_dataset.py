from collections import defaultdict
import os

import numpy as np
import scipy.sparse as sp
import torch
import dgl
from dgl.data import download, extract_archive

from . import BaseDataset, register_dataset

class MHGCN_Base_Dataset(BaseDataset):
    def __init__(self, *args, **kwargs):
        # train_percent = kwargs.pop('train_percent', 0.1)
        super(MHGCN_Base_Dataset, self).__init__(*args, **kwargs)
        _dataset_list = ['dblp4mhgcn','imdb4mhgcn','aminer4mhgcn','alibaba4mhgcn','amazon4mhgcn']
        self.data_path = ""
        self.name = args[0]
        if not self.name in _dataset_list:
            raise ValueError("Unsupported dataset name {}".format(self.name))
        self.data_path = 'openhgnn/dataset/data/{}'.format(self.name)
        
        self.url = 'https://raw.githubusercontent.com/AckerlyLau/raw_openhgnn_test/main/{}.zip'.format(self.name)
        self.process()
        self.multi_label = False

    def process(self):
        if not os.path.exists(self.data_path) or not os.path.exists(os.path.join(self.data_path,self.name+".zip")):
            self.download()
        # self.download()
        self.load_data()

    def _multiplex_relation_aggregation(self,coo_list,etype_count):
        '''
        Multiplex relation aggregation (1)
        In order to decrease the consumption of time , we divide multiplex relation aggregation into two parts.
        The first part is to aggregate all relations into one relation , which will be run only once.
        The second part is to update the weight of the relations , which will be run in every epochs.
        Codes below are the first part.
        '''
        adj_list = []
        for i in range(etype_count):
            adj = sp.coo_matrix(coo_list[i]).todense()
            adj_list.append(torch.tensor(adj,dtype=torch.float32))
        weight = torch.linspace(0,etype_count - 1,etype_count)
        weight = [2**i for i in weight]
        weight = torch.tensor(weight,dtype=torch.float32)
        adj = torch.stack(adj_list,dim=2)
        adj = adj + adj.transpose(0,1)
        aggr_adj = torch.matmul(adj,weight)
        sparse_aggr_adj = sp.coo_matrix(aggr_adj.detach().cpu())

        self.g = dgl.from_scipy(sparse_aggr_adj)
        self.etype_num = etype_count
        etag = torch.zeros(sparse_aggr_adj.data.shape[0],etype_count,dtype=torch.float32)
        for i in range(sparse_aggr_adj.data.shape[0]):
            etag[i] = adj[sparse_aggr_adj.row[i],sparse_aggr_adj.col[i]]
        self.g.edata['tag'] = etag.clone()
    def _multiplex_relation_aggregation_old(self,coo_list,etype_count):
        '''
        Multiplex relation aggregation (1)
        In order to decrease the consumption of time , we divide multiplex relation aggregation into two parts.
        The first part is to aggregate all relations into one relation , which will be run only once.
        The second part is to update the weight of the relations , which will be run in every epochs.
        Codes below are the first part.
        '''
        adj_list = []
        for i in range(etype_count):
            adj = sp.coo_matrix(coo_list[i]).todense()
            adj_list.append(torch.tensor(adj,dtype=torch.float32))
        weight = torch.linspace(0,etype_count - 1,etype_count)
        weight = [2**i for i in weight]
        weight = torch.tensor(weight,dtype=torch.float32)
        adj = torch.stack(adj_list,dim=2)
        # adj = adj + adj.transpose(0,1)
        aggr_adj = torch.matmul(adj,weight)
        aggr_adj = aggr_adj + aggr_adj.transpose(0,1)
        sparse_aggr_adj = sp.coo_matrix(aggr_adj.detach().cpu())
        data_val = sparse_aggr_adj.data
        # data_val = [1 for i in data_val]
        data_val = torch.tensor(data_val,dtype=torch.int32)
        self.g = dgl.from_scipy(sparse_aggr_adj)
        # self.g = dgl.to_bidirected(self.g)
        self.etype_num = etype_count
        etag = torch.zeros(data_val.shape[0],etype_count,dtype=torch.float32)
        for index,val in enumerate(data_val):
            for i in range(etype_count):
                if val & 2**i != 0:
                    etag[index][i] = 1
        self.g.edata['tag'] = etag

    def load_data(self):
        if self.name == 'dblp4mhgcn':
            self.labels = np.load(os.path.join(self.data_path,"labels_mat.npy"))
            self.edges_paper_author = sp.load_npz(os.path.join(self.data_path,'edge_paper_author.npz'))
            self.edges_paper_term = sp.load_npz(os.path.join(self.data_path,'edge_paper_term.npz'))
            self.edges_paper_venue = sp.load_npz(os.path.join(self.data_path,'edge_paper_venue.npz'))
            self.features = np.load(os.path.join(self.data_path,'features.npy'))
            self.train_val_test_idx = np.load(os.path.join(self.data_path,'train_val_test_idx.npz'))
            self.train_idx = self.train_val_test_idx['test_idx']
            self.val_idx = self.train_val_test_idx['val_idx']
            self.test_idx = self.train_val_test_idx['train_idx']
            self._multiplex_relation_aggregation([self.edges_paper_author,self.edges_paper_term,self.edges_paper_venue],3)
            self.etype_num = 3
            self.num_classes = self.labels.shape[1]

        elif self.name == 'imdb4mhgcn':
            self.labels = np.load(os.path.join(self.data_path,"labels.npy"))
            self.edges_A = sp.load_npz(os.path.join(self.data_path,'edges_A.npz'))
            self.edges_B = sp.load_npz(os.path.join(self.data_path,'edges_B.npz'))
            self.features = np.load(os.path.join(self.data_path,'features.npy'))
            self.train_val_test_idx = np.load(os.path.join(self.data_path,'train_val_test_idx.npz'))
            self.train_idx = self.train_val_test_idx['train_idx']
            self.val_idx = self.train_val_test_idx['val_idx']
            self.test_idx = self.train_val_test_idx['test_idx']
            self._multiplex_relation_aggregation([self.edges_A,self.edges_B],2)
            self.etype_num = 2
            self.num_classes = self.labels.shape[1]

        elif self.name == 'alibaba4mhgcn':
            self.labels = np.load(os.path.join(self.data_path,"labels.npy"))
            self.edges_A = sp.load_npz(os.path.join(self.data_path,'edges_A.npz'))
            self.edges_B = sp.load_npz(os.path.join(self.data_path,'edges_B.npz'))
            self.edges_C = sp.load_npz(os.path.join(self.data_path,'edges_C.npz'))
            self.edges_D = sp.load_npz(os.path.join(self.data_path,'edges_D.npz'))
            self.features = np.load(os.path.join(self.data_path,'features.npy'))
            self.train_val_test_idx = np.load(os.path.join(self.data_path,'train_val_test_idx.npz'))
            self.train_idx = self.train_val_test_idx['train_idx']
            self.val_idx = self.train_val_test_idx['val_idx']
            self.test_idx = self.train_val_test_idx['test_idx']
            self._multiplex_relation_aggregation([self.edges_A,self.edges_B,self.edges_C,self.edges_D],4)
            self.etype_num = 4
            self.num_classes = self.labels.shape[1]
            
            # model comparison
            self.A = [self.edges_A,self.edges_B,self.edges_C,self.edges_D]
        # The original author did not provide the following datasets
        # elif self.name == 'amazon4mhgcn':
        #     pass
        # elif self.name == 'aminer4mhgcn':
        #     pass 

    def download(self):
        # download raw data to local disk
        # path to store the file
        if os.path.exists(self.data_path):  # pragma: no cover
           pass
        else:
            file_path = os.path.join(self.data_path,self.name+".zip")
            # download file
            download(self.url, path=file_path)
        extract_archive(os.path.join(self.data_path, self.name+".zip"),self.data_path)

@register_dataset('mhgcn_node_classification')
class MHGCN_NC_Dataset(MHGCN_Base_Dataset):
    def __init__(self, *args, **kwargs):
        # train_percent = kwargs.pop('train_percent', 0.1)
        super(MHGCN_NC_Dataset, self).__init__(*args, **kwargs)

    def get_split(self):
        return self.train_idx, self.val_idx, self.test_idx
    def get_labels(self):
        return torch.argmax(torch.tensor(self.labels),dim=1)
  


@register_dataset('mhgcn_link_prediction')
class MHGCN_LP_Dataset(MHGCN_Base_Dataset):
    def __init__(self, *args, **kwargs):
        # train_percent = kwargs.pop('train_percent', 0.1)
        super(MHGCN_LP_Dataset, self).__init__(*args, **kwargs)
      


    def load_training_data_g(self):
        f_name = os.path.join(self.data_path,'train.txt')
        edges_src = list()
        edges_dst = list()
        edges_type = list()
        with open(f_name, 'r') as f:
            for line in f:
                # words = line[:-1].split('\t')
                words = line[:-1].split()
                x = int(words[1])
                y = int(words[2])
                type = int(words[0])
                edges_src.append(x)
                edges_dst.append(y)
                edges_type.append(type)
        g = dgl.graph((edges_src,edges_dst),num_nodes=self.num_nodes)
        g.edata['type'] = torch.tensor(edges_type)
        return g
    

    def load_testing_data_g(self,is_val:bool):
        f_name = os.path.join(self.data_path,'valid.txt') if is_val else os.path.join(self.data_path,'test.txt')

        true_edges_src = list()
        true_edges_dst = list()
        true_edges_type = list()
        
        false_edges_src = list()
        false_edges_dst = list()
        false_edges_type = list()


        with open(f_name, 'r') as f:
            for line in f:
                # words = line[:-1].split('\t')
                words = line[:-1].split()
                x = int(words[1])
                y = int(words[2])
                type = int(words[0])
                if int(words[3]) == 1:
                    true_edges_src.append(x)
                    true_edges_dst.append(y)
                    true_edges_type.append(type)
                else:
                    false_edges_src.append(x)
                    false_edges_dst.append(y)
                    false_edges_type.append(type)
        true_g = dgl.graph((true_edges_src,true_edges_dst),num_nodes=self.num_nodes)
        true_g.edata['type'] = torch.tensor(true_edges_type)
        false_g = dgl.graph((false_edges_src,false_edges_dst),num_nodes=self.num_nodes)
        false_g.edata['type'] = torch.tensor(false_edges_type)
        return true_g,false_g
    
    def load_training_data(self,f_name):
        edge_data_by_type = dict()
        all_edges = list()
        all_nodes = list()
        with open(f_name, 'r') as f:
            for line in f:
                # words = line[:-1].split('\t')
                words = line[:-1].split()
                # print(words)
                if words[0] not in edge_data_by_type:
                    edge_data_by_type[words[0]] = list()
                x, y = words[1], words[2]
                edge_data_by_type[words[0]].append((x, y))
                all_edges.append((x, y))
                all_nodes.append(x)
                all_nodes.append(y)
        all_nodes = list(set(all_nodes))
        all_edges = list(set(all_edges))
        edge_data_by_type['Base'] = all_edges
        print('total training nodes: ' + str(len(all_nodes)))
        # print('Finish loading training data')
        return edge_data_by_type


    def load_testing_data(self,f_name):
        true_edge_data_by_type = dict()
        false_edge_data_by_type = dict()
        all_edges = list()
        all_nodes = list()
        with open(f_name, 'r') as f:
            for line in f:
                # words = line[:-1].split('\t')
                words = line[:-1].split()
                x, y = words[1], words[2]
                if int(words[3]) == 1:
                    if words[0] not in true_edge_data_by_type:
                        true_edge_data_by_type[words[0]] = list()
                    true_edge_data_by_type[words[0]].append((x, y))
                else:
                    if words[0] not in false_edge_data_by_type:
                        false_edge_data_by_type[words[0]] = list()
                    false_edge_data_by_type[words[0]].append((x, y))
                all_nodes.append(x)
                all_nodes.append(y)
        all_nodes = list(set(all_nodes))
        # print('Finish loading testing data')
        return true_edge_data_by_type, false_edge_data_by_type



    def load_data(self):
        super().load_data()
        # link_prediction
        self.num_nodes = self.g.num_nodes()

        self.train_g = self.load_training_data_g()
        self.val_g,self.val_neg_g = self.load_testing_data_g(is_val=True)
        self.test_g,self.test_neg_g = self.load_testing_data_g(is_val=False)
        self.train_edge_data_by_type = self.load_training_data(os.path.join(self.data_path,'train.txt'))
        self.val_edge_data_by_type, self.val_neg_edge_data_by_type = self.load_testing_data(os.path.join(self.data_path,'valid.txt'))
        self.test_edge_data_by_type, self.test_neg_edge_data_by_type = self.load_testing_data(os.path.join(self.data_path,'test.txt'))
        

    def get_split(self):
        return self.train_g, self.val_g, self.test_g, self.val_neg_g, self.test_neg_g
    def get_labels(self):
        return torch.argmax(torch.tensor(self.labels),dim=1)
    
        

