import numpy as np
from . import BaseDataset, register_dataset
import torch_geometric
import torch
import numpy as np
from torch.utils.data import Dataset
import random
import pickle as pkl
import os
import zipfile
import requests
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
        self.num_classes = None
        assert dataset_name in ['MNIST-P-2','Building-2-C','Building-2-R','Building-S','DBSR-cplx46K']
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
            self.data_dir='./openhgnn/PolyGNN/MNIST-P-2.pkl'
        elif dataset_name in ["Building-2-C"]:
            x_out=100
            self.data_dir='./openhgnn/PolyGNN/Building-2-C.pkl'
        elif dataset_name in ["Building-2-R"]:
            x_out=100
            self.data_dir='./openhgnn/PolyGNN/Building-2-R.pkl'
        elif dataset_name in ["Building-S"]:
            x_out=10
            self.data_dir='./openhgnn/PolyGNN/Building-S.pkl'
        elif dataset_name in ["DBSR-cplx46K"]:
            x_out=10
            self.data_dir='./openhgnn/PolyGNN/DBSR-cplx46K.pkl'
        self.num_classes = x_out
    def get_labels(self):
        pass

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
            train_ds,val_ds,test_ds= get_mnist_dataset(self.data_dir,self.args.man_seed,test_ratio=test_ratio)
        elif self.args.dataset in ['Building-2-C']:
            train_ds,val_ds,test_ds= get_building_dataset(self.data_dir,self.args.man_seed,test_ratio=test_ratio)
        elif self.args.dataset in ['Building-2-R']:
            train_ds,val_ds,test_ds=get_mbuilding_dataset(self.data_dir,self.args.man_seed,test_ratio=test_ratio)
        elif self.args.dataset in ['Building-S']:
            train_ds,val_ds,test_ds=get_sbuilding_dataset(self.data_dir,self.args.man_seed,test_ratio=test_ratio)
        elif self.args.dataset in ['DBSR-cplx46K']:
            train_ds,val_ds,test_ds=get_smnist_dataset(self.data_dir,self.args.man_seed,test_ratio=test_ratio)    

        train_ds= affine_transform_to_range(train_ds,target_range=(-1, 1))
        val_ds= affine_transform_to_range(val_ds,target_range=(-1, 1))
        test_ds= affine_transform_to_range(test_ds,target_range=(-1, 1))

        self.train_loader = torch_geometric.loader.DataLoader(train_ds,batch_size=self.args.train_batch, shuffle=False,pin_memory=True,drop_last=True) 
        self.val_loader = torch_geometric.loader.DataLoader(val_ds, batch_size=self.args.test_batch, shuffle=False, pin_memory=True)
        self.test_loader = torch_geometric.loader.DataLoader(test_ds,batch_size=self.args.test_batch, shuffle=False,pin_memory=True)
        return self.train_loader,self.val_loader,self.test_loader

        

valid_chars = 'EFHILOTUYZ'

alphabetic_labels = [char1 + char2 for char1 in valid_chars for char2 in valid_chars]
alphabetic_labels.sort()
label_mapping = {label: idx for idx, label in enumerate(alphabetic_labels)} # to number
reverse_label_mapping = {v: k for k, v in label_mapping.items()} # to alphabetic

single_alphabetic_labels=[char1 for char1 in valid_chars]
single_alphabetic_labels.sort()
single_label_mapping = {label: idx for idx, label in enumerate(single_alphabetic_labels)}
single_reverse_label_mapping = {v: k for k, v in single_label_mapping.items()}

def get_mnist_dataset(data_dir='./openhgnn/PolyGNN/MNIST-P-2.pkl',Seed=0,test_ratio=0.2):

    random.seed(Seed)
    torch.manual_seed(Seed)  
    np.random.seed(Seed) 

    with open(data_dir, 'rb') as f: # 加载数据集
        dataset = pkl.load(f)
    for entry in dataset: # 映射标签
        entry.y -= 10
                
    np.random.shuffle(dataset) # 分割
    val_test_split = int(np.around( test_ratio * len(dataset) ))
    train_val_split = int(len(dataset)-2*val_test_split)
    train_ds = dataset[:train_val_split]
    val_ds = dataset[train_val_split:train_val_split+val_test_split]
    test_ds = dataset[train_val_split+val_test_split:]  

    print(data_dir)
    print('Train: ' +str(len(train_ds)))
    print('Val  : ' +str(len(val_ds)))
    print('Test : ' +str(len(test_ds)))  
        
    return train_ds,val_ds,test_ds

def get_building_dataset(data_dir='./openhgnn/PolyGNN/Building-2-C.pkl',Seed=0,test_ratio=0.2):

    random.seed(Seed)
    torch.manual_seed(Seed)  
    np.random.seed(Seed) 

    with open(data_dir, 'rb') as f:
        dataset = pkl.load(f)
    for entry in dataset:
        entry.y = label_mapping[entry.y]  
         
    np.random.shuffle(dataset)
    val_test_split = int(np.around( test_ratio * len(dataset) ))
    train_val_split = int(len(dataset)-2*val_test_split)
    train_ds = dataset[:train_val_split]
    val_ds = dataset[train_val_split:train_val_split+val_test_split]
    test_ds = dataset[train_val_split+val_test_split:]  

    print(data_dir)
    print('Train: ' +str(len(train_ds)))
    print('Val  : ' +str(len(val_ds)))
    print('Test : ' +str(len(test_ds)))  
        
    return train_ds,val_ds,test_ds

def get_mbuilding_dataset(data_dir='./openhgnn/PolyGNN/Building-2-R.pkl',Seed=0,test_ratio=0.2):

    random.seed(Seed)
    torch.manual_seed(Seed)  
    np.random.seed(Seed) 

    with open(data_dir, 'rb') as f:
        dataset = pkl.load(f)
    for entry in dataset:
        entry.y = label_mapping[entry.y]  
         
    np.random.shuffle(dataset)
    val_test_split = int(np.around( test_ratio * len(dataset) ))
    train_val_split = int(len(dataset)-2*val_test_split)
    train_ds = dataset[:train_val_split]
    val_ds = dataset[train_val_split:train_val_split+val_test_split]
    test_ds = dataset[train_val_split+val_test_split:]  

    print(data_dir)
    print('Train: ' +str(len(train_ds)))
    print('Val  : ' +str(len(val_ds)))
    print('Test : ' +str(len(test_ds)))  
        
    return train_ds,val_ds,test_ds

def get_sbuilding_dataset(data_dir='./openhgnn/PolyGNN/Building-S.pkl',Seed=0,test_ratio=0.2):

    random.seed(Seed)
    torch.manual_seed(Seed)  
    np.random.seed(Seed) 

    with open(data_dir, 'rb') as f:
        dataset = pkl.load(f)
    for entry in dataset:
        entry.y = single_label_mapping[entry.y]  
         
    np.random.shuffle(dataset)
    val_test_split = int(np.around( test_ratio * len(dataset) ))
    train_val_split = int(len(dataset)-2*val_test_split)
    train_ds = dataset[:train_val_split]
    val_ds = dataset[train_val_split:train_val_split+val_test_split]
    test_ds = dataset[train_val_split+val_test_split:]  

    print(data_dir)
    print('Train: ' +str(len(train_ds)))
    print('Val  : ' +str(len(val_ds)))
    print('Test : ' +str(len(test_ds)))  
        
    return train_ds,val_ds,test_ds

def get_smnist_dataset(data_dir='./openhgnn/PolyGNN/DBSR-cplx46K.pkl',Seed=0,test_ratio=0.2):

    random.seed(Seed)
    torch.manual_seed(Seed)  
    np.random.seed(Seed) 

    with open(data_dir, 'rb') as f:
        dataset = pkl.load(f)
         
    np.random.shuffle(dataset)
    val_test_split = int(np.around( test_ratio * len(dataset) ))
    train_val_split = int(len(dataset)-2*val_test_split)
    train_ds = dataset[:train_val_split]
    val_ds = dataset[train_val_split:train_val_split+val_test_split]
    test_ds = dataset[train_val_split+val_test_split:]  

    print(data_dir)
    print('Train: ' +str(len(train_ds)))
    print('Val  : ' +str(len(val_ds)))
    print('Test : ' +str(len(test_ds)))  
        
    return train_ds,val_ds,test_ds

def affine_transform_to_range(ds, target_range=(-1, 1)):
    # Find the extent (min and max) of coordinates in both x and y directions
    for item in ds:
        min_x  = torch.min(item.pos[:,0])
        min_y  = torch.min(item.pos[:,1])
        
        max_x  = torch.max(item.pos[:,0])
        max_y  = torch.max(item.pos[:,1])
        
        scale_x = (target_range[1] - target_range[0]) / (max_x - min_x)
        scale_y = (target_range[1] - target_range[0]) / (max_y - min_y)
        translate_x = target_range[0] - min_x * scale_x
        translate_y = target_range[0] - min_y * scale_y

        # Apply the affine transformation to 
        item.pos[:,0] = item.pos[:,0] * scale_x + translate_x
        item.pos[:,1] = item.pos[:,1] * scale_y + translate_y
    return ds

class CustomDataset(Dataset):
    def __init__(self, data_list):
        super(CustomDataset, self).__init__()
        self.data_list = data_list
    
    def len(self):
        return len(self.data_list)
    
    def get(self, idx):
        return self.data_list[idx]