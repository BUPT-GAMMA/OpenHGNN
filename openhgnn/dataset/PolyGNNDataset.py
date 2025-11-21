import numpy as np
from . import BaseDataset, register_dataset
import torch
from dgl.dataloading import GraphDataLoader
from dgl.data import DGLDataset
from dgl import load_graphs
import random
import os
import zipfile
import requests


class MyDataset(DGLDataset):
    def __init__(self, graphs,labels,data_dir=None,dataset_name=None):
        super(MyDataset, self).__init__(name='PolyGNN',
                                          url=None,
                                          raw_dir=None,
                                          force_reload=None,
                                          verbose=None)
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.graphs = graphs
        self.labels = labels
    def process(self):
        # 将数据处理为图列表和标签列表
        pass
    def __getitem__(self, idx):
        """ 通过idx获取对应的图和标签

        Parameters
        ----------
        idx : int
            Item index

        Returns
        -------
        (dgl.DGLGraph, Tensor)
        """
        return self.graphs[idx], self.labels[idx]

    def __len__(self):
        """数据集中图的数量"""
        return len(self.graphs)
@register_dataset('PolyGNNDataset')
class PolyGNNDataset(BaseDataset):
    r"""
    The class *PolyGNNDataset* contains datasets with multiple graphs which can be used in task of graph classification,
    i.e.,the task for PolyGNN.

    Attributes
    -------------
    num_classes : int
        The target graph  will be classified into num_classes categories.
    dataset_name : str
        The dataset_name should be within the list of ['building','mnist','mnist_sparse','mbuilding','sbuilding','dbp'],
        each dataset of which is the dataset with large number of graphs for the downstream task of graph classification.
    """

    def __init__(self,dataset_name, args, **kwargs):
        super(PolyGNNDataset, self).__init__(args, **kwargs)
        self.args = args
        self.dataset_name = dataset_name
        assert dataset_name in ['MNIST-P-2','Building-2-C','Building-2-R','Building-S']
        def download_and_extract_zip(url, extract_to):
            os.makedirs(extract_to, exist_ok=True)
            
            zip_path = os.path.join(extract_to, "PolyGNN.zip")
            response = requests.get(url)
            with open(zip_path, "wb") as f:
                f.write(response.content)
            
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_to)

        url = "https://s3.cn-north-1.amazonaws.com.cn/dgl-data/dataset/openhgnn/PolyGNN.zip"
        extract_to = "./openhgnn/PolyGNN"
        download_and_extract_zip(url, extract_to)

        if dataset_name in ["MNIST-P-2"]:
            x_out=90
            self.data_dir='./openhgnn/PolyGNN/MNIST-P-2.bin'
        elif dataset_name in ["Building-2-C"]:
            x_out=100
            self.data_dir='./openhgnn/PolyGNN/Building-2-C.bin'
        elif dataset_name in ["Building-2-R"]:
            x_out=100
            self.data_dir='./openhgnn/PolyGNN/Building-2-R.bin'
        elif dataset_name in ["Building-S"]:
            x_out=10
            self.data_dir='./openhgnn/PolyGNN/Building-S.bin'
        self.num_classes = x_out
    def get_labels(self):
        labels = []
        for i in range(len(self.dataset)):
            _,label = self.dataset[i]
            labels.append(label)
        return labels

    def get_split(self):
        r"""
        To split the dataset into data loaders.
        To specify the ratio of test by adjusting the value of test_ratio.
        return
        -------
        train_loader,val_loader,test_loader : torch_geometric.loader.DataLoader,torch_geometric.loader.DataLoader,torch_geometric.loader.DataLoader
        """
        test_ratio = 0.2
        if self.args.dataset in ['MNIST-P-2']:
            train_ds, val_ds, test_ds, train_labels, val_labels, test_labels= get_mnist_dataset(self.data_dir,self.args.man_seed,test_ratio=test_ratio)
        else:
            train_ds, val_ds, test_ds, train_labels, val_labels, test_labels=get_dataset(self.data_dir,self.args.man_seed,test_ratio=test_ratio)    

        train_ds= affine_transform_to_range(train_ds,target_range=(-1, 1)) # for the attr of pos,changing its range
        val_ds= affine_transform_to_range(val_ds,target_range=(-1, 1))
        test_ds= affine_transform_to_range(test_ds,target_range=(-1, 1))
        train_dataset = MyDataset(train_ds,train_labels)
        val_dataset = MyDataset(val_ds,val_labels)
        test_dataset = MyDataset(test_ds,test_labels)
        self.train_loader = GraphDataLoader(train_dataset,batch_size=self.args.train_batch, shuffle=False,pin_memory=True,drop_last=True) 
        self.val_loader = GraphDataLoader(val_dataset, batch_size=self.args.test_batch, shuffle=False, pin_memory=True)
        self.test_loader = GraphDataLoader(test_dataset,batch_size=self.args.test_batch, shuffle=False,pin_memory=True)
        return self.train_loader,self.val_loader,self.test_loader

        

def get_mnist_dataset(data_dir='./openhgnn/PolyGNN/MNIST-P-2.bin',Seed=0,test_ratio=0.2):

    random.seed(Seed)
    torch.manual_seed(Seed)  
    np.random.seed(Seed) 
    dataset, labels = load_graphs(data_dir)
    labels = labels['labels']

    labels -= 10
    indices = list(range(len(dataset)))
    random.shuffle(indices)            
    val_test_split = int(np.around(test_ratio * len(dataset)))
    train_val_split = len(dataset) - 2 * val_test_split

    # 分割数据集
    train_indices = indices[:train_val_split]
    val_indices = indices[train_val_split:train_val_split + val_test_split]
    test_indices = indices[train_val_split + val_test_split:]

    train_ds = [dataset[i] for i in train_indices]
    val_ds = [dataset[i] for i in val_indices]
    test_ds = [dataset[i] for i in test_indices]

    train_labels = labels[train_indices]
    val_labels = labels[val_indices]
    test_labels = labels[test_indices]

    return train_ds, val_ds, test_ds, train_labels, val_labels, test_labels


def get_dataset(data_dir='./openhgnn/PolyGNN/Building-2-C.bin',Seed=0,test_ratio=0.2):

    random.seed(Seed)
    torch.manual_seed(Seed)  
    np.random.seed(Seed) 
    dataset, labels = load_graphs(data_dir)
    labels = labels['labels']
    indices = list(range(len(dataset)))
    random.shuffle(indices)   

    val_test_split = int(np.around( test_ratio * len(dataset) ))
    train_val_split = int(len(dataset)-2*val_test_split)
    train_indices = indices[:train_val_split]
    val_indices = indices[train_val_split:train_val_split + val_test_split]
    test_indices = indices[train_val_split + val_test_split:]
    train_ds = [dataset[i] for i in train_indices]
    val_ds = [dataset[i] for i in val_indices]
    test_ds = [dataset[i] for i in test_indices]
    train_labels = labels[train_indices]
    val_labels = labels[val_indices]
    test_labels = labels[test_indices]
        
    return train_ds, val_ds, test_ds, train_labels, val_labels, test_labels


def affine_transform_to_range(ds, target_range=(-1, 1)):
    # Find the extent (min and max) of coordinates in both x and y directions
    for item in ds:
        min_x  = torch.min(item.ndata['pos'][:,0])
        min_y  = torch.min(item.ndata['pos'][:,1])
        
        max_x  = torch.max(item.ndata['pos'][:,0])
        max_y  = torch.max(item.ndata['pos'][:,1])
        
        scale_x = (target_range[1] - target_range[0]) / (max_x - min_x)
        scale_y = (target_range[1] - target_range[0]) / (max_y - min_y)
        translate_x = target_range[0] - min_x * scale_x
        translate_y = target_range[0] - min_y * scale_y

        # Apply the affine transformation to 
        item.ndata['pos'][:,0] = item.ndata['pos'][:,0] * scale_x + translate_x
        item.ndata['pos'][:,1] = item.ndata['pos'][:,1] * scale_y + translate_y
    return ds
