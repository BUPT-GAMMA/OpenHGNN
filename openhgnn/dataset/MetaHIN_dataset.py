
import gc
import glob
import os
import pickle

# from DataProcessor import Movielens
from tqdm import tqdm
from multiprocessing import Process, Pool
from multiprocessing.pool import ThreadPool
import numpy as np
import torch


class Meta_DataHelper:
    def __init__(self, input_dir, config):
        self.input_dir = input_dir  
        self.config = config
        self.mp_list = ["ub", "ubab", "ubub"]

        from dgl.data.utils import download, extract_archive
        #   只有dbook这一个数据集
        dataset_name = 'dbook'
        self.zip_file = f'./openhgnn/dataset/Common_Dataset/{dataset_name}.zip'
        #   common_dataset/dbook_dir
        self.base_dir = './openhgnn/dataset/Common_Dataset/' +   dataset_name+'_dir'     
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


    def load_data(self, data_set, state, load_from_file=True):
        #   解压后的dbook目录：  input_dir下的dbook文件夹
        # data_dir = self.input_dir + data_set
        #   修改后代码
        data_dir = self.base_dir +'/'+data_set
        supp_xs_s = []
        supp_ys_s = []
        supp_mps_s = []
        query_xs_s = []
        query_ys_s = []
        query_mps_s = []

        if data_set == "yelp":
            training_set_size = int(
                len(glob.glob("{}/{}/*.npy".format(data_dir, state)))
                / self.config.file_num
            )  # support, query

            # load all data
            for idx in tqdm(range(training_set_size)):
                supp_xs_s.append(
                    torch.from_numpy(
                        np.load("{}/{}/support_x_{}.npy".format(data_dir, state, idx))
                    )
                )
                supp_ys_s.append(
                    torch.from_numpy(
                        np.load("{}/{}/support_y_{}.npy".format(data_dir, state, idx))
                    )
                )
                query_xs_s.append(
                    torch.from_numpy(
                        np.load("{}/{}/query_x_{}.npy".format(data_dir, state, idx))
                    )
                )
                query_ys_s.append(
                    torch.from_numpy(
                        np.load("{}/{}/query_y_{}.npy".format(data_dir, state, idx))
                    )
                )

                supp_mp_data, query_mp_data = {}, {}
                for mp in self.mp_list:
                    _cur_data = np.load(
                        "{}/{}/support_{}_{}.npy".format(data_dir, state, mp, idx),
                        encoding="latin1",
                    )
                    supp_mp_data[mp] = [torch.from_numpy(x) for x in _cur_data]
                    _cur_data = np.load(
                        "{}/{}/query_{}_{}.npy".format(data_dir, state, mp, idx),
                        encoding="latin1",
                    )
                    query_mp_data[mp] = [torch.from_numpy(x) for x in _cur_data]
                supp_mps_s.append(supp_mp_data)
                query_mps_s.append(query_mp_data)
        else:   #   'dbook'
            training_set_size = int(
                len(glob.glob("{}/{}/*.pkl".format(data_dir, state)))
                / self.config.file_num
            )  # support, query

            # load all data
            for idx in tqdm(range(training_set_size)):
                support_x = pickle.load(
                    open("{}/{}/support_x_{}.pkl".format(data_dir, state, idx), "rb")
                )
                if support_x.shape[0] > 5:
                    continue
                del support_x
                supp_xs_s.append(
                    pickle.load(
                        open(
                            "{}/{}/support_x_{}.pkl".format(data_dir, state, idx), "rb"
                        )
                    )
                )
                supp_ys_s.append(
                    pickle.load(
                        open(
                            "{}/{}/support_y_{}.pkl".format(data_dir, state, idx), "rb"
                        )
                    )
                )
                query_xs_s.append(
                    pickle.load(
                        open("{}/{}/query_x_{}.pkl".format(data_dir, state, idx), "rb")
                    )
                )
                query_ys_s.append(
                    pickle.load(
                        open("{}/{}/query_y_{}.pkl".format(data_dir, state, idx), "rb")
                    )
                )

                supp_mp_data, query_mp_data = {}, {}
                for mp in self.mp_list:
                    supp_mp_data[mp] = pickle.load(
                        open(
                            "{}/{}/support_{}_{}.pkl".format(data_dir, state, mp, idx),
                            "rb",
                        )
                    )
                    query_mp_data[mp] = pickle.load(
                        open(
                            "{}/{}/query_{}_{}.pkl".format(data_dir, state, mp, idx),
                            "rb",
                        )
                    )
                supp_mps_s.append(supp_mp_data)
                query_mps_s.append(query_mp_data)

        print(
            "#support set: {}, #query set: {}".format(len(supp_xs_s), len(query_xs_s))
        )
        total_data = list(
            zip(supp_xs_s, supp_ys_s, supp_mps_s, query_xs_s, query_ys_s, query_mps_s)
        )  # all training tasks
        del (supp_xs_s, supp_ys_s, supp_mps_s, query_xs_s, query_ys_s, query_mps_s)
        gc.collect()
        return total_data