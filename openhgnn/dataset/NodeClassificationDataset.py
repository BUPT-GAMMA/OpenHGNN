import os
import dgl
import dgl.function as fn
import torch as th
import numpy as np
from dgl.data.rdf import AIFBDataset, MUTAGDataset, BGSDataset, AMDataset
from dgl.data.utils import load_graphs, save_graphs
from dgl.data import CoraGraphDataset,CiteseerGraphDataset,PubmedGraphDataset
from dgl import sparse as dglsp
# from dgl.data import TexasDataset,CornellDataset
import scipy.sparse as sp
from ogb.nodeproppred import DglNodePropPredDataset
from . import load_acm_raw
from . import BaseDataset, register_dataset

from . import AcademicDataset, HGBDataset, OHGBDataset,IMDB4MAGNN_Dataset
from .utils import sparse_mx_to_torch_sparse_tensor, to_symmetric, row_norm
from ..utils import add_reverse_edges
import os
from dgl.data.utils import download, extract_archive
from abc import ABC

from collections import defaultdict

import torch



########################        add dataset here


@register_dataset('common_dataset')
class Common_Dataset(BaseDataset):
    def __init__(self, dataset_name, *args, **kwargs):
        super(Common_Dataset, self).__init__(*args, **kwargs)
        assert dataset_name in ['BPHGNN_dataset','acm4HGMAE','hgprompt_acm_dblp','acm4FedHGNN']

        if dataset_name == 'acm4HGMAE':
            # 这是从云盘上下载下来的   本地zip文件
            self.zip_file = f'./openhgnn/dataset/Common_Dataset/{dataset_name}.zip'
            #本地base_dir文件夹.
            self.base_dir = './openhgnn/dataset/Common_Dataset/' + dataset_name + '_dir'
            #   云端的zip文件
            self.url = f'https://s3.cn-north-1.amazonaws.com.cn/dgl-data/dataset/openhgnn/{dataset_name}.zip'
            if os.path.exists(self.zip_file):  
                pass
            else:
                os.makedirs(    os.path.join('./openhgnn/dataset/Common_Dataset/')  ,exist_ok= True)
                download(self.url, 
                        path=os.path.join('./openhgnn/dataset/Common_Dataset/')     
                        )     
            if os.path.exists( self.base_dir ):
                pass
            else:
                os.makedirs( os.path.join( self.base_dir )  ,exist_ok= True       )
                extract_archive(self.zip_file, self.base_dir)  # 把graph.bin 解压到 base_dir文件夹中


            self.g = dgl.load_graphs(   os.path.join(self.base_dir,f'{dataset_name}.bin')    )[0][0]
            self.category = 'paper'
            self.num_classes = 3
            
            self.meta_paths_dict = {}   #   元路径
            self.has_feature = True    #   是否有初始特征

######  add dataset here
        elif dataset_name == 'BPHGNN_dataset':
            # 这是从云盘上下载下来的   本地zip文件
            self.zip_file = f'./openhgnn/dataset/Common_Dataset/{dataset_name}.zip'
            #本地base_dir文件夹.
            self.base_dir = './openhgnn/dataset/Common_Dataset/' + dataset_name + '_dir'
            #   云端的zip文件
            self.url = f'https://s3.cn-north-1.amazonaws.com.cn/dgl-data/dataset/openhgnn/{dataset_name}.zip'
            if os.path.exists(self.zip_file):  
                pass
            else:
                os.makedirs(    os.path.join('./openhgnn/dataset/Common_Dataset/')  ,exist_ok= True)
                download(self.url, 
                        path=os.path.join('./openhgnn/dataset/Common_Dataset/')     
                        )     
            if os.path.exists( self.base_dir ):
                pass
            else:
                os.makedirs( os.path.join( self.base_dir )  ,exist_ok= True       )
                extract_archive(self.zip_file, self.base_dir)  



        elif dataset_name == 'hgprompt_acm_dblp':
            self.zip_file = f'./openhgnn/dataset/Common_Dataset/{dataset_name}.zip'
            self.base_dir = './openhgnn/dataset/Common_Dataset/' + dataset_name + '_dir'
            self.url = f'https://s3.cn-north-1.amazonaws.com.cn/dgl-data/dataset/openhgnn/{dataset_name}.zip'
            if os.path.exists(self.zip_file):  
                pass
            else:
                os.makedirs(    os.path.join('./openhgnn/dataset/Common_Dataset/')  ,exist_ok= True)
                download(self.url, 
                        path=os.path.join('./openhgnn/dataset/Common_Dataset/')     
                        )     
            if os.path.exists( self.base_dir ):
                pass
            else:
                os.makedirs( os.path.join( self.base_dir )  ,exist_ok= True       )
                extract_archive(self.zip_file, self.base_dir)  

        elif dataset_name == 'acm4FedHGNN':
            self.zip_file = f'./openhgnn/dataset/Common_Dataset/{dataset_name}.zip'
            self.base_dir = './openhgnn/dataset/Common_Dataset/' + dataset_name + '_dir'
            self.url = f'https://s3.cn-north-1.amazonaws.com.cn/dgl-data/dataset/openhgnn/{dataset_name}.zip'
            if os.path.exists(self.zip_file):  
                pass
            else:
                os.makedirs(    os.path.join('./openhgnn/dataset/Common_Dataset/')  ,exist_ok= True)
                download(self.url, 
                        path=os.path.join('./openhgnn/dataset/Common_Dataset/')     
                        )     
            if os.path.exists( self.base_dir ):
                pass
            else:
                os.makedirs( os.path.join( self.base_dir )  ,exist_ok= True       )
                extract_archive(self.zip_file, self.base_dir)  

            self.acm_mat_file = os.path.join(self.base_dir,'acm4FedHGNN.mat')
            import scipy.io as sio
            self.data = sio.loadmat(self.acm_mat_file)



        elif dataset_name == 'acm4FedHGNN':
            self.zip_file = f'./openhgnn/dataset/Common_Dataset/{dataset_name}.zip'
            self.base_dir = './openhgnn/dataset/Common_Dataset/' + dataset_name + '_dir'
            self.url = f'https://s3.cn-north-1.amazonaws.com.cn/dgl-data/dataset/openhgnn/{dataset_name}.zip'
            if os.path.exists(self.zip_file):  
                pass
            else:
                os.makedirs(    os.path.join('./openhgnn/dataset/Common_Dataset/')  ,exist_ok= True)
                download(self.url, 
                        path=os.path.join('./openhgnn/dataset/Common_Dataset/')     
                        )     
            if os.path.exists( self.base_dir ):
                pass
            else:
                os.makedirs( os.path.join( self.base_dir )  ,exist_ok= True       )
                extract_archive(self.zip_file, self.base_dir)  

            self.acm_mat_file = os.path.join(self.base_dir,'acm4FedHGNN.mat')
            import scipy.io as sio
            self.data = sio.loadmat(self.acm_mat_file)




@register_dataset('GraphBolt_Dataset')
class GraphBolt_Dataset(BaseDataset):
    def __init__(self, dataset_name, *args, **kwargs):

        super(GraphBolt_Dataset, self).__init__(*args, **kwargs)
        assert dataset_name in ['imdb4GTN','HGBl-amazon']
        self.zip_path = f'./openhgnn/dataset/GraphBolt_Dataset/{dataset_name}_base_dir.zip'
        self.base_dir = './openhgnn/dataset/GraphBolt_Dataset/' + dataset_name + '_base_dir' 
        self.url = f'https://s3.cn-north-1.amazonaws.com.cn/dgl-data/dataset/openhgnn/{dataset_name}_base_dir.zip'
        import os
        from dgl.data.utils import download, extract_archive
        if os.path.exists(self.zip_path):  
           pass
        else:
            os.makedirs(    os.path.join('./openhgnn/dataset/GraphBolt_Dataset/')  ,exist_ok= True)

            download(self.url, 
                     path=os.path.join('./openhgnn/dataset/GraphBolt_Dataset/')     
                     )   
        if os.path.exists( self.base_dir ):
            pass
        else:
            extract_archive(self.zip_path, os.path.join('./openhgnn/dataset/GraphBolt_Dataset/'))  



@register_dataset('node_classification')
class NodeClassificationDataset(BaseDataset):
    r"""
    The class *NodeClassificationDataset* is a base class for datasets which can be used in task *node classification*.
    So its subclass should contain attributes such as graph, category, num_classes and so on.
    Besides, it should implement the functions *get_labels()* and *get_split()*.

    Attributes
    -------------
    g : dgl.DGLHeteroGraph
        The heterogeneous graph.
    category : str
        The category(or target) node type need to be predict. In general, we predict only one node type.
    num_classes : int
        The target node  will be classified into num_classes categories.
    has_feature : bool
        Whether the dataset has feature. Default ``False``.
    multi_label : bool
        Whether the node has multi label. Default ``False``. For now, only HGBn-IMDB has multi-label.
    """

    def __init__(self, *args, **kwargs):
        super(NodeClassificationDataset, self).__init__(*args, **kwargs)
        self.g = None
        self.category = None
        self.num_classes = None
        self.has_feature = False
        self.multi_label = False
        self.meta_paths_dict =None
        # self.in_dim = None

    def get_labels(self):
        r"""
        The subclass of dataset should overwrite the function. We can get labels of target nodes through it.

        Notes
        ------
        In general, the labels are th.LongTensor.
        But for multi-label dataset, they should be th.FloatTensor. Or it will raise
        RuntimeError: Expected object of scalar type Long but got scalar type Float for argument #2 target' in call to _thnn_nll_loss_forward
        
        return
        -------
        labels : torch.Tensor
        """
        if 'labels' in self.g.nodes[self.category].data:
            labels = self.g.nodes[self.category].data.pop('labels').long()
        elif 'label' in self.g.nodes[self.category].data:
            labels = self.g.nodes[self.category].data.pop('label').long()
        else:
            raise ValueError('Labels of nodes are not in the hg.nodes[category].data.')
        labels = labels.float() if self.multi_label else labels
        return labels

    def get_split(self, validation=True):
        r"""
        
        Parameters
        ----------
        validation : bool
            Whether to split dataset. Default ``True``. If it is False, val_idx will be same with train_idx.

        We can get idx of train, validation and test through it.

        return
        -------
        train_idx, val_idx, test_idx : torch.Tensor, torch.Tensor, torch.Tensor
        """
        if 'train_mask' not in self.g.nodes[self.category].data:
            self.logger.dataset_info("The dataset has no train mask. "
                  "So split the category nodes randomly. And the ratio of train/test is 8:2.")
            num_nodes = self.g.number_of_nodes(self.category)
            n_test = int(num_nodes * 0.2)
            n_train = num_nodes - n_test
    
            train, test = th.utils.data.random_split(range(num_nodes), [n_train, n_test])
            train_idx = th.tensor(train.indices)
            test_idx = th.tensor(test.indices)
            if validation:
                self.logger.dataset_info("Split train into train/valid with the ratio of 8:2 ")
                random_int = th.randperm(len(train_idx))
                valid_idx = train_idx[random_int[:len(train_idx) // 5]]
                train_idx = train_idx[random_int[len(train_idx) // 5:]]
            else:
                self.logger.dataset_info("Set valid set with train set.")
                valid_idx = train_idx
                train_idx = train_idx
        else:
            train_mask = self.g.nodes[self.category].data.pop('train_mask')
            test_mask = self.g.nodes[self.category].data.pop('test_mask')
            train_idx = th.nonzero(train_mask, as_tuple=False).squeeze()
            test_idx = th.nonzero(test_mask, as_tuple=False).squeeze()
            if validation:
                if 'val_mask' in self.g.nodes[self.category].data:
                    val_mask = self.g.nodes[self.category].data.pop('val_mask')
                    valid_idx = th.nonzero(val_mask, as_tuple=False).squeeze()
                elif 'valid_mask' in self.g.nodes[self.category].data:
                    val_mask = self.g.nodes[self.category].data.pop('valid_mask').squeeze()
                    valid_idx = th.nonzero(val_mask, as_tuple=False).squeeze()
                else:
                    # RDF_NodeClassification has train_mask, no val_mask
                    self.logger.dataset_info("Split train into train/valid with the ratio of 8:2 ")
                    random_int = th.randperm(len(train_idx))
                    valid_idx = train_idx[random_int[:len(train_idx) // 5]]
                    train_idx = train_idx[random_int[len(train_idx) // 5:]]
            else:
                self.logger.dataset_info("Set valid set with train set.")
                valid_idx = train_idx
                train_idx = train_idx
        self.train_idx = train_idx
        self.valid_idx = valid_idx
        self.test_idx = test_idx
        # Here set test_idx as attribute of dataset to save results of HGB
        return self.train_idx, self.valid_idx, self.test_idx



@register_dataset('hga_node_classification')
class HGA_NodeClassification(NodeClassificationDataset):
    def __init__(self, dataset_name, *args, **kwargs):   
        super(HGA_NodeClassification, self).__init__(*args, **kwargs)
        assert dataset_name in ['acm4HGA','dblp4HGA']
        if dataset_name == 'acm4HGA':
            self.zip_file = f'./openhgnn/dataset/Common_Dataset/{dataset_name}.zip'
            self.base_dir = './openhgnn/dataset/Common_Dataset/' + dataset_name + '_dir'
            self.url = f'https://s3.cn-north-1.amazonaws.com.cn/dgl-data/dataset/openhgnn/{dataset_name}.zip'
            if os.path.exists(self.zip_file):  
                pass
            else:
                os.makedirs(    os.path.join('./openhgnn/dataset/Common_Dataset/')  ,exist_ok= True)
                download(self.url, 
                        path=os.path.join('./openhgnn/dataset/Common_Dataset/')     
                        )     
            if os.path.exists( self.base_dir ):
                pass
            else:
                os.makedirs( os.path.join( self.base_dir )  ,exist_ok= True       )
                extract_archive(self.zip_file, self.base_dir) 
            self.g = dgl.load_graphs(   os.path.join(self.base_dir,f'{dataset_name}.bin')    )[0][0].long()
            self.category = '1'
            self.num_classes = 4
            self.in_dim = self.g.ndata['h'][self.category].shape[1]

            self.meta_paths_dict = {
                '131': [('1', '2', '3'), ('3', '2', '1')],
                '121': [('1', '1', '2'), ('2', '1', '1')],
            }
            self.has_feature = True    

        elif dataset_name == 'dblp4HGA':
            self.zip_file = f'./openhgnn/dataset/Common_Dataset/{dataset_name}.zip'
            self.base_dir = './openhgnn/dataset/Common_Dataset/' + dataset_name + '_dir'
            self.url = f'https://s3.cn-north-1.amazonaws.com.cn/dgl-data/dataset/openhgnn/{dataset_name}.zip'
            if os.path.exists(self.zip_file):  
                pass
            else:
                os.makedirs(    os.path.join('./openhgnn/dataset/Common_Dataset/')  ,exist_ok= True)
                download(self.url, 
                        path=os.path.join('./openhgnn/dataset/Common_Dataset/')     
                        )     
            if os.path.exists( self.base_dir ):
                pass
            else:
                os.makedirs( os.path.join( self.base_dir )  ,exist_ok= True       )
                extract_archive(self.zip_file, self.base_dir)  

            self.g = dgl.load_graphs(   os.path.join(self.base_dir,f'{dataset_name}.bin')    )[0][0].long()
            self.category = '1'
            self.num_classes = 4
            self.in_dim = self.g.ndata['h'][self.category].shape[1]
            self.meta_paths_dict = {
                '131': [('1', '2', '3'), ('3', '2', '1')],
                '121': [('1', '1', '2'), ('2', '1', '1')],
            }
            self.has_feature = True    


@register_dataset('rhine_node_classification')
class RHINE_NodeClassification(NodeClassificationDataset):
    def __init__(self, dataset_name, *args, **kwargs):
        super(RHINE_NodeClassification, self).__init__(*args, **kwargs)
        self.dataset_name = dataset_name
        assert self.dataset_name in['dblp4RHINE']


        self.zip_file = f'./openhgnn/dataset/Common_Dataset/{dataset_name}.zip'
        self.base_dir = './openhgnn/dataset/Common_Dataset/' + dataset_name + '_dir'
        self.url = f'https://s3.cn-north-1.amazonaws.com.cn/dgl-data/dataset/openhgnn/{dataset_name}.zip'
        if os.path.exists(self.zip_file):  
            pass
        else:
            os.makedirs(    os.path.join('./openhgnn/dataset/Common_Dataset/')  ,exist_ok= True)
            download(self.url, 
                    path=os.path.join('./openhgnn/dataset/Common_Dataset/')     
                    )     
        if os.path.exists( self.base_dir ):
            pass
        else:
            os.makedirs( os.path.join( self.base_dir )  ,exist_ok= True       )
            extract_archive(self.zip_file, self.base_dir)          
    

        self.graph_file = os.path.join(self.base_dir,'dblp4RHINE.bin')
        self.load_rhine_data()


    def load_rhine_data(self):
        if self.dataset_name == 'dblp4RHINE':
            if os.path.exists(self.graph_file):
                self.g=dgl.load_graphs(self.graph_file)[0][0].long()
            
            self.node_types = {'a': 'author', 'p': 'paper', 't': 'term', 'c': 'conf'}
            self.IRs = ['ap', 'pt', 'apt']
            self.ARs = ['pc', 'apc']
            self.category='paper'
            self.meta_paths_dict={
                'ap':[('author', 'writes', 'paper'),('paper','written_by','author')],
                'pt':[('paper', 'has_term', 'term'),('term','term_of','paper')],
                'apt':[('author', 'writes', 'paper'), ('paper', 'has_term', 'term'),('term','term_of','paper'),('paper','written_by','author')],
                'pc':[('paper', 'published_in', 'conf'), ('conf', 'publish', 'paper')],
                'apc':[('author', 'writes', 'paper'), ('paper', 'published_in', 'conf'), ('conf', 'publish', 'paper'),('paper','written_by','author')]
            }
            test_mask = [i!=-1 for i in self.g.nodes['paper'].data['label']]
            self.g.nodes['paper'].data['label_mask']=th.tensor(test_mask)
            self.train_id = th.tensor(range(self.g.num_nodes('paper')))
            self.pred_id=self.test_id=self.valid_id = th.tensor(range(self.g.num_nodes('paper')))[test_mask]





class MHGCN_Base_Dataset(BaseDataset):
    def __init__(self, *args, **kwargs):
        # train_percent = kwargs.pop('train_percent', 0.1)
        super(MHGCN_Base_Dataset, self).__init__(*args, **kwargs)
        _dataset_list = ['dblp4MHGCN','imdb4MHGCN','alibaba4MHGCN']
        self.data_path = ""
        self.name = args[0]
        if not self.name in _dataset_list:
            raise ValueError("Unsupported dataset name {}".format(self.name))
        self.data_path = 'openhgnn/dataset/data/{}'.format(self.name)
        self.url = 'https://s3.cn-north-1.amazonaws.com.cn/dgl-data/dataset/openhgnn/{}.zip'.format(self.name)
        self.process()
        self.multi_label = False

    def process(self):
        if not os.path.exists(self.data_path) or not os.path.exists(os.path.join(self.data_path,self.name+".zip")):
            self.download()
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
        aggr_adj = torch.matmul(adj,weight)
        aggr_adj = aggr_adj + aggr_adj.transpose(0,1)
        sparse_aggr_adj = sp.coo_matrix(aggr_adj.detach().cpu())
        data_val = sparse_aggr_adj.data
        data_val = torch.tensor(data_val,dtype=torch.int32)
        self.g = dgl.from_scipy(sparse_aggr_adj)
        self.etype_num = etype_count
        etag = torch.zeros(data_val.shape[0],etype_count,dtype=torch.float32)
        for index,val in enumerate(data_val):
            for i in range(etype_count):
                if val & 2**i != 0:
                    etag[index][i] = 1
        self.g.edata['tag'] = etag

    def load_data(self):
        if self.name == 'dblp4MHGCN':
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

        elif self.name == 'imdb4MHGCN':
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

        elif self.name == 'alibaba4MHGCN':
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

    def download(self):
        # download raw data to local disk
        try:
            if os.path.exists(self.data_path):  # pragma: no cover
                pass
            else:
                file_path = os.path.join(self.data_path,self.name+".zip")
                # download file
                download(self.url, path=file_path)
            extract_archive(os.path.join(self.data_path, self.name+".zip"),self.data_path)
        except Exception as e:
            os.removedirs(os.path.join(self.data_path))
            raise e



@register_dataset('mhgcn_node_classification')
class MHGCN_NC_Dataset(MHGCN_Base_Dataset):
    def __init__(self, *args, **kwargs):
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

    def load_training_data(self):
        f_name = os.path.join(self.data_path,'train.txt')
        edges_src = list()
        edges_dst = list()
        edges_type = list()
        with open(f_name, 'r') as f:
            for line in f:
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

    def load_testing_data(self,is_val:bool):
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

    def load_data(self):
        super().load_data()
        self.num_nodes = self.g.num_nodes()
        self.train_g = self.load_training_data()
        self.val_g,self.val_neg_g = self.load_testing_data(is_val=True)
        self.test_g,self.test_neg_g = self.load_testing_data(is_val=False)

    def get_split(self):
        return self.train_g, self.val_g, self.test_g, self.val_neg_g, self.test_neg_g

    def get_labels(self):
        return torch.argmax(torch.tensor(self.labels),dim=1)







####################################


@register_dataset('rdf_node_classification')
class RDF_NodeClassification(NodeClassificationDataset):
    r"""
    The RDF dataset will be used in task *entity classification*.
    Dataset Name : aifb/ mutag/ bgs/ am.
    We download from dgl and process it, refer to
    `RDF datasets <https://docs.dgl.ai/api/python/dgl.data.html#rdf-datasets>`_.

    Notes
    ------
    They are all have no feature.
    """

    def __init__(self, dataset_name, *args, **kwargs):
        super(RDF_NodeClassification, self).__init__(*args, **kwargs)
        self.g, self.category, self.num_classes = self.load_RDF_dgl(dataset_name)
        self.has_feature = False

    def load_RDF_dgl(self, dataset):
        # load graph data
        if dataset == 'aifb':
            kg_dataset = AIFBDataset()
        elif dataset == 'mutag':
            kg_dataset = MUTAGDataset()
        elif dataset == 'bgs':
            kg_dataset = BGSDataset()
        elif dataset == 'am':
            kg_dataset = AMDataset()
        else:
            raise ValueError()

        # Load from hetero-graph
        kg = kg_dataset[0]
        category = kg_dataset.predict_category
        num_classes = kg_dataset.num_classes
        return kg, category, num_classes


@register_dataset('hin_node_classification')
class HIN_NodeClassification(NodeClassificationDataset):
    r"""
    The HIN dataset are all used in different papers. So we preprocess them and store them as form of dgl.DGLHeteroGraph.
    The dataset name combined with paper name through 4(for).

    Dataset Name :
    acm4NSHE/ acm4GTN/ acm4NARS/ acm_han_raw/ academic4HetGNN/ dblp4MAGNN/ imdb4MAGNN/ ...
    """

    def __init__(self, dataset_name, *args, **kwargs):
        super(HIN_NodeClassification, self).__init__(*args, **kwargs)

        if 'args' in kwargs:  
            self.args = kwargs['args']
        else:
            self.args = None

        self.g, self.category, self.num_classes = self.load_HIN(dataset_name)


    def load_HIN(self, name_dataset):
        if name_dataset == 'demo_graph':
            data_path = './openhgnn/dataset/demo_graph.bin'
            category = 'author'
            num_classes = 4
            g, _ = load_graphs(data_path)
            g = g[0].long()
            self.in_dim = g.ndata['h'][category].shape[1]
        elif name_dataset == 'acm4NSHE':
            dataset = AcademicDataset(name='acm4NSHE', raw_dir='')
            category = 'paper'
            g = dataset[0].long()
            num_classes = 3
            self.in_dim = g.ndata['h'][category].shape[1]
        elif name_dataset == 'dblp4MAGNN':
            dataset = AcademicDataset(name='dblp4MAGNN', raw_dir='')
            category = 'A'
            g = dataset[0].long()
            num_classes = 4
            self.meta_paths_dict = {
                'APVPA': [('A', 'A-P', 'P'), ('P', 'P-V', 'V'), ('V', 'V-P', 'P'), ('P', 'P-A', 'A')],
                'APA': [('A', 'A-P', 'P'), ('P', 'P-A', 'A')],
            }
            self.meta_paths = [(('A', 'A-P', 'P'), ('P', 'P-V', 'V'), ('V', 'V-P', 'P'), ('P', 'P-A', 'A')),
                               (('A', 'A-P', 'P'), ('P', 'P-A', 'A'))]
            self.in_dim = g.ndata['h'][category].shape[1]

        elif name_dataset == 'imdb4MAGNN':
            if self.args:
                if self.args.use_database == True:
                    dataset = IMDB4MAGNN_Dataset(name='imdb4MAGNN',args = self.args)
            else:
                dataset = AcademicDataset(name='imdb4MAGNN', raw_dir='')

            category = 'M'
            g = dataset[0].long()
            num_classes = 3
            self.in_dim = g.ndata['h'][category].shape[1]
        elif name_dataset == 'imdb4GTN':
            dataset = AcademicDataset(name='imdb4GTN', raw_dir='')
            category = 'movie'
            g = dataset[0].long()
            num_classes = 3
            self.in_dim = g.ndata['h'][category].shape[1]
        elif name_dataset == 'acm4GTN':
            dataset = AcademicDataset(name='acm4GTN', raw_dir='')
            category = 'paper'
            g = dataset[0].long()
            num_classes = 3
            self.meta_paths_dict = {'PAPSP': [('paper', 'paper-author', 'author'), ('author', 'author-paper', 'paper'),
                                              ('paper', 'paper-subject', 'subject'),
                                              ('subject', 'subject-paper', 'paper')],
                                    'PAP': [('paper', 'paper-author', 'author'), ('author', 'author-paper', 'paper')],
                                    'PSP': [('paper', 'paper-subject', 'subject'),
                                              ('subject', 'subject-paper', 'paper')]
                                    }
            # self.meta_paths = [(('paper', 'paper-author', 'author'), ('author', 'author-paper', 'paper'),
            #                     ('paper', 'paper-subject', 'subject'), ('subject', 'subject-paper', 'paper'))]
            self.in_dim = g.ndata['h'][category].shape[1]
        elif name_dataset == 'acm4NARS':
            dataset = AcademicDataset(name='acm4NARS', raw_dir='')
            g = dataset[0].long()
            num_classes = 3
            # g, labels, num_classes, train_nid, val_nid, test_nid = load_acm_nars()
            category = 'paper'
        elif name_dataset == 'acm4HeCo':
            dataset = AcademicDataset(name='acm4HeCo', raw_dir='')
            pos = sp.load_npz("./openhgnn/dataset/acm4HeCo/pos.npz")
            self.pos = sparse_mx_to_torch_sparse_tensor(pos)
            g = dataset[0].long()
            num_classes = 3
            category = 'paper'
        elif name_dataset == 'academic4HetGNN':
            # which is used in HetGNN
            dataset = AcademicDataset(name='academic4HetGNN', raw_dir='')
            category = 'author'
            g = dataset[0].long()
            num_classes = 4
        elif name_dataset == 'yelp4HeGAN':
            # which is used in HeGAN
            dataset = AcademicDataset(name='yelp4HeGAN', raw_dir='')
            category = 'business'
            g = dataset[0].long()
            num_classes = 3
        elif name_dataset == 'yelp4HGSL':
            # yelp used for HGSL
            dataset = AcademicDataset(name = 'yelp4HGSL', raw_dir='')
            category = 'b'
            g = dataset[0].long()
            num_classes = 4
            self.meta_paths_dict = {'bub': [('b', 'b-u', 'u'), ('u', 'u-b', 'b')],
                                    'bsb': [('b', 'b-s', 's'), ('s', 's-b', 'b')],
                                    'bublb': [('b', 'b-u', 'u'), ('u', 'u-b', 'b'),
                                                ('b', 'b-l', 'l'), ('l', 'l-b', 'b')],
                                    'bubsb': [('b', 'b-u', 'u'), ('u', 'u-b', 'b'),
                                                ('b', 'b-s', 's'), ('s', 's-b', 'b')]
                                    }
        elif name_dataset == 'HNE-PubMed':
            # which is used in HeGAN
            dataset = AcademicDataset(name='HNE-PubMed', raw_dir='')
            category = 'DISEASE'
            g = dataset[0].long()
            num_classes = 8
            g = add_reverse_edges(g)
            self.meta_paths_dict = {'DCD': [('DISEASE', 'CHEMICAL-in-DISEASE-rev', 'CHEMICAL'), ('CHEMICAL', 'CHEMICAL-in-DISEASE', 'DISEASE')],
                                    'DDD': [('DISEASE', 'DISEASE-and-DISEASE', 'DISEASE'), ('DISEASE', 'DISEASE-and-DISEASE-rev', 'DISEASE')],
                                    'DGD': [('DISEASE', 'GENE-causing-DISEASE-rev', 'GENE'), ('GENE', 'GENE-causing-DISEASE', 'DISEASE')],
                                    'DSD': [('DISEASE', 'SPECIES-with-DISEASE-rev', 'SPECIES'), ('SPECIES', 'SPECIES-with-DISEASE', 'DISEASE')]
                                    }
        elif name_dataset in ['acm_han', 'acm_han_raw']:
            if name_dataset == 'acm_han':
                pass
            elif name_dataset == 'acm_han_raw':
                g, category, num_classes, self.in_dim = load_acm_raw(False)
                self.meta_paths_dict = {'PAP': [('paper', 'pa', 'author'), ('author', 'ap', 'paper')],
                                        'PFP': [('paper', 'pf', 'field'), ('field', 'fp', 'paper')]
                                        }
            else:
                return NotImplementedError('Unsupported dataset {}'.format(name_dataset))
            return g, category, num_classes
        elif name_dataset in ['demo']:
            data_path = './openhgnn/dataset/graph.bin'
            category = 'author'
            num_classes = 4
            g, _ = load_graphs(data_path)
            g = g[0].long()
            self.in_dim = g.ndata['h'][category].shape[1]
        # g, _ = load_graphs(data_path)
        # g = g[0]
        return g, category, num_classes


@register_dataset('ohgb_node_classification')
class OHGB_NodeClassification(NodeClassificationDataset):
    def __init__(self, dataset_name, *args, **kwargs):
        super(OHGB_NodeClassification, self).__init__(*args, **kwargs)
        self.dataset_name = dataset_name
        self.has_feature = True
        if dataset_name == 'ohgbn-Freebase':
            dataset = OHGBDataset(name=dataset_name, raw_dir='')
            g = dataset[0].long()
            category = 'BOOK'
            num_classes = 8
            g = add_reverse_edges(g)
            self.meta_paths_dict = {'BB': [('BOOK', 'BOOK-and-BOOK', 'BOOK')],
                                    'BFB': [('BOOK', 'BOOK-to-FILM', 'FILM'), ('FILM', 'BOOK-to-FILM-rev', 'BOOK')],
                                    'BOFB': [('BOOK', 'BOOK-about-ORGANIZATION', 'ORGANIZATION'),
                                             ('ORGANIZATION', 'ORGANIZATION-in-FILM', 'FILM'),
                                             ('FILM', 'BOOK-to-FILM-rev', 'BOOK')],
                                    'BLMB': [('BOOK', 'BOOK-on-LOCATION', 'LOCATION'),
                                             ('LOCATION', 'MUSIC-on-LOCATION-rev', 'MUSIC'),
                                             ('MUSIC', 'MUSIC-in-BOOK', 'BOOK')],
                                    'BPB': [('BOOK', 'PEOPLE-to-BOOK-rev', 'PEOPLE'),
                                            ('PEOPLE', 'PEOPLE-to-BOOK', 'BOOK')],
                                    'BPSB': [('BOOK', 'PEOPLE-to-BOOK-rev', 'PEOPLE'),
                                             ('PEOPLE', 'PEOPLE-to-SPORTS', 'SPORTS'),
                                             ('SPORTS', 'BOOK-on-SPORTS-rev', 'BOOK')],
                                    'BBuB': [('BOOK', 'BUSINESS-about-BOOK-rev', 'BUSINESS'),
                                             ('BUSINESS', 'BUSINESS-about-BOOK', 'BOOK')],
                                    }
        elif dataset_name == 'ohgbn-yelp2':
            dataset = OHGBDataset(name=dataset_name, raw_dir='')
            g = dataset[0].long()
            g = add_reverse_edges(g)
            category = 'business'
            num_classes = 16
            self.multi_label = True
        elif dataset_name == 'ohgbn-acm':
            dataset = OHGBDataset(name=dataset_name, raw_dir='')
            g = dataset[0].long()
            category = 'paper'
            num_classes = 3
            self.meta_paths_dict = {
                                    'PAP': [('paper', 'paper-author', 'author'), ('author', 'author-paper', 'paper')],
                                    'PSP': [('paper', 'paper-subject', 'subject'),
                                              ('subject', 'subject-paper', 'paper')]
                                    }
                                    
        elif dataset_name == 'ohgbn-imdb':
            dataset = OHGBDataset(name=dataset_name, raw_dir='')
            category = 'movie'
            g = dataset[0].long()
            num_classes = 3
            self.meta_paths_dict = {
                'MAM': [('movie', 'movie-actor', 'actor'), ('actor', 'actor-movie', 'movie')],
                'MDM': [('movie', 'movie-director', 'director'), ('director', 'director-movie', 'movie')]}
            
        self.g, self.category, self.num_classes = g, category, num_classes
    

@register_dataset('HGBn_node_classification')
class HGB_NodeClassification(NodeClassificationDataset):
    r"""
    The HGB dataset will be used in task *node classification*.

    Dataset Name :
    HGBn-ACM/HGBn-DBLP/HGBn-Freebase/HGBn-IMDB

    So if you want to get more information, refer to
    `HGB datasets <https://github.com/THUDM/HGB>`_
    """

    def __init__(self, dataset_name, *args, **kwargs):
        super(HGB_NodeClassification, self).__init__(*args, **kwargs)
        self.dataset_name = dataset_name
        self.has_feature = True
        if dataset_name == 'HGBn-ACM':
            dataset = HGBDataset(name=dataset_name, raw_dir='')
            g = dataset[0].long()
            category = 'paper'
            num_classes = 3
            # graph: dgl graph object, label: torch tensor of shape (num_nodes, num_tasks)
            self.meta_paths_dict = {'PAP': [('paper', 'paper-author', 'author'), ('author', 'author-paper', 'paper')],
                                    'PSP': [('paper', 'paper-subject', 'subject'),
                                            ('subject', 'subject-paper', 'paper')],
                                    'PcPAP': [('paper', 'paper-cite-paper', 'paper'),
                                              ('paper', 'paper-author', 'author'),
                                              ('author', 'author-paper', 'paper')],
                                    'PcPSP': [('paper', 'paper-cite-paper', 'paper'),
                                              ('paper', 'paper-subject', 'subject'),
                                              ('subject', 'subject-paper', 'paper')],
                                    'PrPAP': [('paper', 'paper-ref-paper', 'paper'),
                                              ('paper', 'paper-author', 'author'),
                                              ('author', 'author-paper', 'paper')],
                                    'PrPSP': [('paper', 'paper-ref-paper', 'paper'),
                                              ('paper', 'paper-subject', 'subject'),
                                              ('subject', 'subject-paper', 'paper')]
                                    }
        elif dataset_name == 'HGBn-DBLP':
            dataset = HGBDataset(name=dataset_name, raw_dir='')
            g = dataset[0].long()
            category = 'author'
            num_classes = 4
            self.meta_paths_dict = {'APA': [('author', 'author-paper', 'paper'), ('paper', 'paper-author', 'author')],
                                    'APTPA': [('author', 'author-paper', 'paper'), ('paper', 'paper-term', 'term'),
                                              ('term', 'term-paper', 'paper'), ('paper', 'paper-author', 'author')],
                                    'APVPA': [('author', 'author-paper', 'paper'), ('paper', 'paper-venue', 'venue'),
                                              ('venue', 'venue-paper', 'paper'), ('paper', 'paper-author', 'author')],
                                    }
            # graph: dgl graph object, label: torch tensor of shape (num_nodes, num_tasks)
        elif dataset_name == 'HGBn-Freebase':
            dataset = HGBDataset(name=dataset_name, raw_dir='')
            g = dataset[0].long()
            category = 'BOOK'
            num_classes = 7
            self.has_feature = False
            g = add_reverse_edges(g)
            self.meta_paths_dict = {'BB': [('BOOK', 'BOOK-and-BOOK', 'BOOK')],
                                    'BFB': [('BOOK', 'BOOK-to-FILM', 'FILM'), ('FILM', 'BOOK-to-FILM-rev', 'BOOK')],
                                    'BOFB': [('BOOK', 'BOOK-about-ORGANIZATION', 'ORGANIZATION'),
                                             ('ORGANIZATION', 'ORGANIZATION-in-FILM', 'FILM'),
                                             ('FILM', 'BOOK-to-FILM-rev', 'BOOK')],
                                    'BLMB': [('BOOK', 'BOOK-on-LOCATION', 'LOCATION'),
                                             ('LOCATION', 'MUSIC-on-LOCATION-rev', 'MUSIC'),
                                             ('MUSIC', 'MUSIC-in-BOOK', 'BOOK')],
                                    'BPB': [('BOOK', 'PEOPLE-to-BOOK-rev', 'PEOPLE'),
                                            ('PEOPLE', 'PEOPLE-to-BOOK', 'BOOK')],
                                    'BPSB': [('BOOK', 'PEOPLE-to-BOOK-rev', 'PEOPLE'),
                                             ('PEOPLE', 'PEOPLE-to-SPORTS', 'SPORTS'),
                                             ('SPORTS', 'BOOK-on-SPORTS-rev', 'BOOK')],
                                    'BBuB': [('BOOK', 'BUSINESS-about-BOOK-rev', 'BUSINESS'),
                                             ('BUSINESS', 'BUSINESS-about-BOOK', 'BOOK')],
                                    # 'BOMB': [('BOOK', 'BOOK-about-ORGANIZATION', 'ORGANIZATION'),
                                    #          ('ORGANIZATION', 'ORGANIZATION-to-MUSIC', 'MUSIC'),
                                    #          ('MUSIC', 'MUSIC-in-BOOK', 'BOOK')],
                                    # 'BOBuB': [('BOOK', 'BOOK-about-ORGANIZATION', 'ORGANIZATION'),
                                    #           ('ORGANIZATION', 'ORGANIZATION-for-BUSINESS', 'BUSINESS'),
                                    #           ('BUSINESS', 'BUSINESS-about-BOOK', 'BOOK')]
                                    }
            # graph: dgl graph object, label: torch tensor of shape (num_nodes, num_tasks)
        elif dataset_name == 'HGBn-IMDB':
            dataset = HGBDataset(name=dataset_name, raw_dir='')
            g = dataset[0].long()
            category = 'movie'
            num_classes = 5
            self.meta_paths_dict = {
                'MAM': [('movie', 'movie->actor', 'actor'), ('actor', 'actor->movie', 'movie')],
                'MDM': [('movie', 'movie->director', 'director'), ('director', 'director->movie', 'movie')],
                'MKM': [('movie', 'movie->keyword', 'keyword'), ('keyword', 'keyword->movie', 'movie')],
                'DMD': [('director', 'director->movie', 'movie'), ('movie', 'movie->director', 'director')],
                'DMAMD': [('director', 'director->movie', 'movie'), ('movie', 'movie->actor', 'actor'),
                          ('actor', 'actor->movie', 'movie'), ('movie', 'movie->director', 'director')],
                'AMA': [('actor', 'actor->movie', 'movie'), ('movie', 'movie->actor', 'actor')],
                'AMDMA': [('actor', 'actor->movie', 'movie'), ('movie', 'movie->director', 'director'),
                          ('director', 'director->movie', 'movie'), ('movie', 'movie->actor', 'actor')]
            }
            # RuntimeError: result type Float can't be cast to the desired output type Long
            self.multi_label = True
        else:
            raise ValueError
        self.g, self.category, self.num_classes = g, category, num_classes

    def save_results(self, logits, file_path):
        r"""
        To save test results of HGBn.

        Parameters
        ----------
        logits: th.Tensor
            The prediction of target nodes.
        file_path : str
            The path to save file.

        """
        test_logits = logits[self.test_idx]
        if self.dataset_name == 'HGBn-IMDB':
            pred = (test_logits.cpu().numpy() > 0).astype(int)
            multi_label = []
            for i in range(pred.shape[0]):
                label_list = [str(j) for j in range(pred[i].shape[0]) if pred[i][j] == 1]
                multi_label.append(','.join(label_list))
            pred = multi_label
        elif self.dataset_name in ['HGBn-ACM', 'HGBn-DBLP', 'HGBn-Freebase']:
            pred = test_logits.cpu().numpy().argmax(axis=1)
            pred = np.array(pred)
        else:
            return
        with open(file_path, "w") as f:
            for nid, l in zip(self.test_idx, pred):
                f.write(f"{nid}\t\t{0}\t{l}\n")

@register_dataset('ogbn_node_classification')
class OGB_NodeClassification(NodeClassificationDataset):
    def __init__(self, dataset_name, *args, **kwargs):
        super(OGB_NodeClassification, self).__init__(*args, **kwargs)
        if dataset_name == 'ogbn-mag':
            dataset = DglNodePropPredDataset(name='ogbn-mag')
            self.category = 'paper'  # graph: dgl graph object, label: torch tensor of shape (num_nodes, num_tasks)
        else:
            raise ValueError

        split_idx = dataset.get_idx_split()
        self.num_classes = dataset.num_classes
        self.train_idx, self.valid_idx, self.test_idx = split_idx["train"][self.category], split_idx["valid"][
            self.category], split_idx["test"][self.category]
        self.g, self.label_dict = dataset[0]
        self.SeHGNN_g = self.mag4sehgnn(dataset)
        self.g = self.mag4HGT(self.g)
        self.label = self.label_dict[self.category].squeeze(dim=-1)
        # 2-dim label
        self.in_dim = self.g.ndata['h'][self.category].shape[1]
        self.has_feature = True
        # pass

    def get_split(self, validation=True):
        return self.train_idx, self.valid_idx, self.test_idx

    def get_labels(self):
        return self.label

    def mag4sehgnn(self, dataset):
        g, _ = dataset[0]
        embed_size = g.nodes['paper'].data['feat'].size(0)

        author_emb = th.Tensor(g.num_nodes('author'), 256).uniform_(-0.5, 0.5)
        topic_emb = th.Tensor(g.num_nodes('field_of_study'), 256).uniform_(-0.5, 0.5)
        institution_emb = th.Tensor(g.num_nodes('institution'), 256).uniform_(-0.5, 0.5)

        g.nodes['author'].data['feat'] = author_emb
        g.nodes['institution'].data['feat'] = institution_emb
        g.nodes['field_of_study'].data['feat'] = topic_emb

        adjs = []
        i = 0
        for src_type, edge_type, dst_type in g.canonical_etypes:
            src, dst, eid = g._graph.edges(i)
            adj = dglsp.spmatrix(indices = th.stack((src, dst)), shape = (g.number_of_nodes(src_type), g.number_of_nodes(dst_type)))
            adjs.append(adj)
            i += 1
        # F --- *P --- A --- I
        # paper : [736389, 128]
        # author: [1134649, 256]
        # institution [8740, 256]
        # field_of_study [59965, 256]

        new_edges = {}
        ntypes = set()

        etypes = [  # src->tgt
            ('A', 'A-I', 'I'),
            ('A', 'A-P', 'P'),
            ('P', 'P-P', 'P'),
            ('P', 'P-F', 'F'),
        ]

        adjs[2] = to_symmetric(adjs[2])
        for etype, adj in zip(etypes, adjs):
            stype, rtype, dtype = etype
            src, dst = adj.coo()
            if stype == dtype:
                new_edges[(stype, rtype, dtype)] = (np.concatenate((src, dst)), np.concatenate((dst, src)))
            else:
                new_edges[(stype, rtype, dtype)] = (src, dst)
                new_edges[(dtype, rtype[::-1], stype)] = (dst, src)
            ntypes.add(stype)
            ntypes.add(dtype)

        new_g = dgl.heterograph(new_edges)
        new_g.nodes['P'].data['P'] = g.nodes['paper'].data['feat']
        new_g.nodes['A'].data['A'] = g.nodes['author'].data['feat']
        new_g.nodes['I'].data['I'] = g.nodes['institution'].data['feat']
        new_g.nodes['F'].data['F'] = g.nodes['field_of_study'].data['feat']

        IA, PA, PP, FP = adjs

        diag_name = f'ogbn-mag_PP_diag.pt'
        if not os.path.exists(diag_name):
            PP_rm_diag = row_norm(PP)
            th.save(PP_rm_diag, diag_name)

        diag_name = f'ogbn-mag_PPP_diag.pt'
        if not os.path.exists(diag_name):
            PP_rm_diag = row_norm(PP)
            PPP_rm_diag = dglsp.spspmm(PP_rm_diag, PP_rm_diag)
            th.save(PPP_rm_diag, diag_name)

        diag_name = f'ogbn-mag_PAP_diag.pt'
        if not os.path.exists(diag_name):
            PA_rm_diag = row_norm(PA)
            AP_rm_diag = row_norm(PA.T)
            PAP_rm_diag = dglsp.spspmm(PA_rm_diag, AP_rm_diag)
            th.save(PAP_rm_diag, diag_name)

        diag_name = f'ogbn-mag_PFP_diag.pt'
        if not os.path.exists(diag_name):
            PF_rm_diag = row_norm(FP.T)
            FP_rm_diag = row_norm(FP)
            PFP_rm_diag = dglsp.spspmm(PF_rm_diag, FP_rm_diag)
            th.save(PFP_rm_diag, diag_name)

        return new_g
    def mag4HGT(self, hg):
        # Add reverse edge types

        edges = {etype: hg.edges(etype=etype) for etype in hg.canonical_etypes}
        edges.update({(v, e + '_inv', u): (dst, src) for (u, e, v), (src, dst) in edges.items()})
        hg2 = dgl.heterograph(edges)
        hg2 = dgl.to_simple(hg2)

        # Initialize year
        hg2.nodes['paper'].data['timestamp'] = hg.nodes['paper'].data['year'].squeeze()
        for ntype in hg.ntypes:
            if ntype != 'paper':
                hg2.nodes[ntype].data['timestamp'] = th.zeros(hg2.num_nodes(ntype), dtype=th.int64)

        # Aggregate bag-of-paper features
        hg2.nodes['paper'].data['h'] = hg.nodes['paper'].data['feat']
        hg2.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'h'), etype='has_topic')  # field_of_study
        hg2.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'h'), etype='writes_inv')  # author
        hg2.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'h'), etype='affiliated_with')  # institution

        # Attach log-degree to feature of each node type
        for ntype in hg2.ntypes:
            hg2.nodes[ntype].data['deg'] = th.zeros(hg2.num_nodes(ntype))
        for utype, etype, vtype in hg2.canonical_etypes:
            hg2.nodes[vtype].data['deg'] += hg2.in_degrees(etype=etype)
        for ntype in hg2.ntypes:
            hg2.nodes[ntype].data['h'] = th.cat([
                hg2.nodes[ntype].data['h'],
                th.log10(hg2.nodes[ntype].data['deg'][:, None])], 1)
            del hg2.nodes[ntype].data['deg']

        return hg2

@register_dataset('common_node_classification')
class Common_NodeClassification(NodeClassificationDataset):
    def __init__(self, dataset_name, *args, **kwargs):
        super(Common_NodeClassification, self).__init__(*args, **kwargs)
        self.g, self.category, self.num_classes = self.load_Common_dgl(dataset_name)
        self.has_feature = True

    def load_Common_dgl(self,dataset_name):
        if dataset_name == 'Cora':
            dataset = CoraGraphDataset()
            g = dataset[0]
            num_classes = dataset.num_classes
            category = None
        elif dataset_name == 'Citeseer':
            dataset = CiteseerGraphDataset()
            g = dataset[0]
            num_classes = dataset.num_classes
            category = None
        elif dataset_name == 'Pubmed':
            dataset = PubmedGraphDataset()
            g = dataset[0]
            num_classes = dataset.num_classes
            category = None

        # elif dataset_name == 'Texas':
        #     dataset = TexasDataset()
        #     g = dataset[0]
        #     g = dgl.add_self_loop(g)
        #     # g = dgl.to_bidirected(g)
        #     num_classes = dataset.num_classes
        #     category = None
        #
        # elif dataset_name == 'Cornell':
        #     dataset = CornellDataset()
        #     g = dataset[0]
        #     g = dgl.add_self_loop(g)
        #     num_classes = dataset.num_classes
        #     category = None

        else:
            raise ValueError()

        return g,category,num_classes

