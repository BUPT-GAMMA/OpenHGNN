import os
import torch
import scipy.sparse as sp
import dgl
import torch as th
from dgl.data import extract_archive, download
import numpy as np
from . import BaseDataset
from collections import Counter, defaultdict
import torch.nn.functional as F

class SlotGATDataLoader(BaseDataset):
    def __init__(self, args):
        self.name = "acmSlotGAT"
        self.path = './openhgnn/dataset/acmSlotGAT/data/'
        self.multi_label = args.multi_label
        self.meta_paths_dict = None
        self.category = 0
        self.nodes = self.load_nodes()
        self.links = self.load_links()
        self.labels_train = self.load_labels('label.dat')
        self.labels_test = self.load_labels('label.dat.test')
        self.num_classes = self.labels_train['num_classes']
        self.g = self.load_graph(args.device)
        self.e_feat = self.get_e_feat()
        self.features_list = self.get_feature_list(args.device)
        self.process_g()
        
    def load_graph(self,device):
        adjM = sum(self.links['data'].values())
        g = dgl.DGLGraph(adjM + adjM.T)
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)
        g = g.to(device)
        return g
        
    def get_feature_list(self,device):
        features = []
        for i in range(len(self.nodes['count'])):
            th = self.nodes['attr'][i]
            if th is None:
                features.append(sp.eye(self.nodes['count'][i]))  # 对于没有特征值的添加单位矩阵
            else:
                features.append(th)
        features_list = [self.mat2tensor(feature).to(device) for feature in features] 
        save = 0
        in_dims = []#[features_list[0].shape[1]] + [10] * (len(features_list) - 1)
        for i in range(0, len(features_list)):
            if i == save:
                in_dims.append(features_list[i].shape[1])
            else:
                in_dims.append(10)
                features_list[i] = torch.zeros((features_list[i].shape[0], 10)).to(device)
        return features_list
     
    def mat2tensor(self,mat):
        if type(mat) is np.ndarray:
            return torch.from_numpy(mat).type(torch.FloatTensor)
        return self.sp_to_spt(mat)

    def sp_to_spt(self,mat):
        coo = mat.tocoo()
        values = coo.data
        indices = np.vstack((coo.row, coo.col))

        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = coo.shape

        return torch.sparse.FloatTensor(i, v, torch.Size(shape))

    def get_e_feat(self):
        edge2type = {}
        for k in self.links['data']:
            for u,v in zip(*self.links['data'][k].nonzero()):
                edge2type[(u,v)] = k
        count_self=0
        for i in range(self.nodes['total']):
            FLAG=0
            if (i,i) not in edge2type:
                edge2type[(i,i)] = len(self.links['count'])
                FLAG=1
        count_self+=FLAG
        count_reverse=0
        for k in self.links['data']:
            FLAG=0
            for u,v in zip(*self.links['data'][k].nonzero()):
                if (v,u) not in edge2type:
                    edge2type[(v,u)] = count_reverse+1+len(self.links['count'])
                    FLAG=1
            count_reverse+=FLAG
        e_feat = []
        count=0
        count_mappings={}
        counted_dict={}
        eid=0
        etype_ids={}
        g_=self.g.cpu()
        for u, v in zip(*g_.edges()):  #转换边的类型
            u =u.item() #u.cpu().item()
            v =v.item() #v.cpu().item()
            if not counted_dict.setdefault(edge2type[(u,v)],False) :
                count_mappings[edge2type[(u,v)]]=count
                counted_dict[edge2type[(u,v)]]=True
                count+=1
            e_feat.append(count_mappings[edge2type[(u,v)]])
            
            if edge2type[(u,v)] in etype_ids.keys():  # 记录转换后的边的id
                etype_ids[edge2type[(u,v)]].append(eid)
            else:
                etype_ids[edge2type[(u,v)]]=[eid]
            eid+=1
        e_feat = torch.tensor(e_feat, dtype=torch.long).to(self.g.device)
        self.g.etype_ids=etype_ids
        return e_feat
    
    def process_g(self):
        num_ntypes=len(self.features_list)
        num_nodes=self.nodes['total']
        self.g.edge_type_indexer=F.one_hot(self.e_feat).to(self.g.device)
        self.g.node_idx_by_ntype=[]
        self.g.num_ntypes=num_ntypes
        self.g.node_ntype_indexer=torch.zeros(num_nodes,num_ntypes).to(self.g.device)
        ntype_dims=[]
        idx_count=0
        ntype_count=0
        for feature in self.features_list: # 图中的每种节点类型分配唯一的全局索引，并建立“节点-类型”的映射关系
            temp=[]
            for _ in feature:
                temp.append(idx_count)
                self.g.node_ntype_indexer[idx_count][ntype_count]=1
                idx_count+=1

            self.g.node_idx_by_ntype.append(temp)
            ntype_dims.append(feature.shape[1])
            ntype_count+=1 
    
    def get_loss_fn():
        return F.nll_loss
        
    def load_links(self):
        """
        return links dict
            total: total number of links
            count: a dict of int, number of links for each type
            meta: a dict of tuple, explaining the link type is from what type of node to what type of node
            data: a dict of sparse matrices, each link type with one matrix. Shapes are all (nodes['total'], nodes['total'])
        """
        links = {'total':0, 'count':Counter(), 'meta':{}, 'data':defaultdict(list)}
        r_ids=[]
        with open(os.path.join(self.path, 'link.dat'), 'r', encoding='utf-8') as f:
            for line in f:
                th = line.split('\t')
                # 头结点id,尾节点id，边类型id，边权重
                h_id, t_id, r_id, link_weight = int(th[0]), int(th[1]), int(th[2]), float(th[3])
                if h_id in self.old_to_new_id_mapping.keys() and t_id in self.old_to_new_id_mapping.keys() :
                    h_id=self.old_to_new_id_mapping[h_id]
                    t_id=self.old_to_new_id_mapping[t_id]
                    if r_id not in links['meta']:
                        h_type = self.get_node_type(h_id)
                        t_type = self.get_node_type(t_id)
                        links['meta'][r_id] = (h_type, t_type)
                    links['data'][r_id].append((h_id, t_id, link_weight))
                    links['count'][r_id] += 1
                    links['total'] += 1
                    if r_id not in r_ids:
                        r_ids.append(r_id)
        r_ids=sorted(r_ids)
        # 数据有序化
        temp_meta={}
        for i in range(len(links['meta'].keys())):
            temp_meta[i]=links['meta'][r_ids[i]]
        links['meta']=temp_meta

        temp_count={}
        for i in range(len(links['count'].keys())):
            temp_count[i]=links['count'][r_ids[i]]
        links['count']=temp_count

        temp_data={}
        for i in range(len(links['data'].keys())):
            temp_data[i]=links['data'][r_ids[i]]
        links['data']=temp_data

        new_data = {}
        for r_id in links['data']:
            new_data[r_id] = self.list_to_sp_mat(links['data'][r_id])
        links['data'] = new_data
        return links
    
    def get_node_type(self, node_id):
        for i in range(len(self.nodes['shift'])):
            if node_id < self.nodes['shift'][i]+self.nodes['count'][i]:
                return i
    
    def load_labels(self, name):
        """
        return labels dict
            num_classes: total number of labels
            total: total number of labeled data
            count: number of labeled data for each node type
            data: a numpy matrix with shape (self.nodes['total'], self.labels['num_classes'])
            mask: to indicate if that node is labeled, if False, that line of data is masked
        """
        labels = {'num_classes':0, 'total':0, 'count':Counter(), 'data':None, 'mask':None}
        nc = 0
        mask = np.zeros(self.nodes['total'], dtype=bool)
        data = [None for i in range(self.nodes['total'])]
        with open(os.path.join(self.path, name), 'r', encoding='utf-8') as f:
            for line in f:
                th = line.split('\t')
                # 节点id，节点名字，标签针对节点类型的分类，节点标签
                node_id, node_name, node_type, node_label = int(th[0]), th[1], int(th[2]), list(map(int, th[3].split(',')))
                for label in node_label:
                    nc = max(nc, label+1)
                mask[node_id] = True
                data[node_id] = node_label
                labels['count'][node_type] += 1
                labels['total'] += 1
        labels['num_classes'] = nc
        new_data = np.zeros((self.nodes['total'], labels['num_classes']), dtype=int)
        for i,x in enumerate(data):
            if x is not None:
                for j in x:
                    new_data[i, j] = 1
        labels['data'] = new_data
        labels['mask'] = mask
        return labels
    
    def list_to_sp_mat(self, li):
        data = [x[2] for x in li]
        i = [x[0] for x in li]
        j = [x[1] for x in li]
        return sp.coo_matrix((data, (i,j)), shape=(self.nodes['total'], self.nodes['total'])).tocsr()
    
    def load_nodes(self):
        """
        return nodes dict
            total: total number of nodes
            count: a dict of int, number of nodes for each type
            attr: a dict of np.array (or None), attribute matrices for each type of nodes
            shift: node_id shift for each type. You can get the id range of a type by 
                        [ shift[node_type], shift[node_type]+count[node_type] )
        """
        nodes = {'total':0, 'count':Counter(), 'attr':{}, 'shift':{}}
        node_ids=[]
        #with open(os.path.join(self.path, 'new.txt'), 'w') as f:
            #f.write("1")

    
        with open(os.path.join(self.path, 'node.dat'), 'r', encoding='utf-8') as f:
            for line in f:
                th = line.split('\t')
                if len(th) == 4:
                    # 节点id，节点名字，节点类型，节点属性
                    node_id, node_name, node_type, node_attr = th
                    node_id = int(node_id)
                    node_type = int(node_type)
                    node_attr = list(map(float, node_attr.split(',')))
                    node_ids.append(node_id)
                    nodes['count'][node_type] += 1
                    nodes['attr'][node_id] = node_attr
                    nodes['total'] += 1
                elif len(th) == 3:
                    # Then this line of node doesn't have attribute
                    node_id, node_name, node_type = th
                    node_id = int(node_id)
                    node_type = int(node_type)
                    node_ids.append(node_id)
                    nodes['count'][node_type] += 1
                    nodes['total'] += 1
                else:
                    raise Exception("Too few information to parse!")

        #type_id mapping type_id有序化
        temp_count=Counter()
        for k in nodes['count']:
            temp_count[k]=nodes['count'][k]
        nodes['count']=temp_count
        #node_id mapping node_id有序化
        self.old_to_new_id_mapping={}
        self.new_to_old_id_mapping={}
        node_ids=sorted(node_ids)
        for new_id in range(len(node_ids)):
            self.old_to_new_id_mapping[node_ids[new_id]]=new_id
            self.new_to_old_id_mapping[new_id]=node_ids[new_id]
        temp_attr={}
        for old_id in node_ids:
            if old_id in nodes['attr'].keys():
                temp_attr[self.old_to_new_id_mapping[old_id]]=nodes['attr'][old_id]
        nodes['attr']=temp_attr

        shift = 0
        attr = {}
        for i in range(len(nodes['count'])):
            nodes['shift'][i] = shift

            if shift in nodes['attr']:
                mat = []
                for j in range(shift, shift+nodes['count'][i]):
                    mat.append(nodes['attr'][j])
                attr[i] = np.array(mat)
            else:
                attr[i] = None
            shift += nodes['count'][i]
        nodes['attr'] = attr
        return nodes
        
    def get_split(self):
        val_ratio = 0.2
        self.train_idx = np.nonzero(self.labels_train['mask'])[0]
        np.random.shuffle(self.train_idx)
        split = int(self.train_idx.shape[0]*val_ratio)
        self.val_idx = self.train_idx[:split]
        self.train_idx = self.train_idx[split:]
        self.train_idx = np.sort(self.train_idx)
        self.val_idx = np.sort(self.val_idx)
        self.test_idx = np.nonzero(self.labels_test['mask'])[0]
        
        return self.train_idx,self.val_idx,self.test_idx
    
    def get_labels(self):
        self.labels = np.zeros((self.nodes['count'][0], self.labels_train['num_classes']), dtype=int)
        self.labels[self.train_idx] = self.labels_train['data'][self.train_idx]
        self.labels[self.val_idx] = self.labels_train['data'][self.val_idx]
        if not self.multi_label:
            self.labels = self.labels.argmax(axis=1)

        self.labels = torch.LongTensor(self.labels).to(self.g.device)  if not self.multi_label else  torch.FloatTensor(self.labels).to(self.g.device)
        return self.labels


    def get_feature(
            self,
    ):
        return self.g.ndata.pop("feat")

    def to(self, device):
        self.g = self.g.to(device)
        return self

    def has_cache(self):
        return os.path.exists(self.g_path)
