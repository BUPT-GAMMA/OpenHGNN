from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from dgl.data.utils import download, extract_archive
import os



class MeiRECDataset(Dataset):

    def __init__(self, phase , args = None):
      
        # 这是从云盘上下载下来的   本地zip文件
        self.zip_file = os.path.join(  args.openhgnn_dir ,'dataset','Common_Dataset','meirec.zip'  )
        #本地base_dir文件夹.
        self.base_dir = os.path.join(  args.openhgnn_dir ,'dataset','Common_Dataset','meirec_dir'  )
        #   云端的zip文件
        self.url = f'https://s3.cn-north-1.amazonaws.com.cn/dgl-data/dataset/openhgnn/meirec.zip'
        if os.path.exists(self.zip_file):  
            pass
        else:
            os.makedirs(    os.path.join(  args.openhgnn_dir ,'dataset','Common_Dataset')  ,exist_ok= True)
            download(self.url, 
                    path=os.path.join(  args.openhgnn_dir ,'dataset','Common_Dataset')
                    )     
            
        if os.path.exists( self.base_dir ):
            pass
        else:
            os.makedirs( os.path.join( self.base_dir )  ,exist_ok= True       )
            extract_archive(self.zip_file, self.base_dir)  # 把meirec.zip文件 解压到 base_dir文件夹中，得到2个文件train_data.txt和 test_data.txt

      
        assert phase in ['train', 'test']
        if phase == 'train':
            self.data_path = os.path.join(self.base_dir , "train_data.txt")
        else:
            self.data_path = os.path.join(self.base_dir, "test_data.txt")
        self.load_data(self.data_path)

    def load_data(self, path):
        self.features_list = []
        self.label_list = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                feature, label = line.split()
                self.features_list.append(feature)
                self.label_list.append(float(label))
        self.data_len = len(self.features_list)
        # print(self.data_len)

    def __getitem__(self, item):
        index = item
        feature_list = []
        feature = self.features_list[index]
        features_split = feature.split(',')

        for item in features_split[:81]:
            feature_list.append(float(item))
        for item in features_split[81:]:
            feature_list.append(int(item))

        return {
            'data': torch.from_numpy(np.array(feature_list)),
            'labels': self.label_list[index]
        }

    def __len__(self):
        return self.data_len


def get_data_loader(type, batch_size=64, num_workers=0 , args = None):
    dataset = MeiRECDataset(type , args = args)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    return dataloader

