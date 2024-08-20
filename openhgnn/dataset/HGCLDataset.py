import torch as t
from dgl.data import DGLDataset
from dgl.data.utils import download, extract_archive
from dgl.data.utils import load_graphs
import os
import numpy as np
import dgl
import pickle


class HGCLDataset(DGLDataset):

    _prefix = 'https://s3.cn-north-1.amazonaws.com.cn/dgl-data/'
    _urls = {

    }

    def __init__(self, name, raw_dir=None, force_reload=False, verbose=True):
        assert name in ['Epinions', 'CiaoDVD', 'Yelp']
        self.data_path = './{}.zip'.format(name)
        self.g_path = './{}/graph.bin'.format(name)
        raw_dir = './'
        url = 'https://s3.cn-north-1.amazonaws.com.cn/dgl-data/' + 'dataset/{}.zip'.format(name)

        super(HGCLDataset, self).__init__(name=name,
                                          url=url,
                                        raw_dir=raw_dir,
                                        force_reload=force_reload,
                                        verbose=verbose)
    def create_graph(self):
        '''
            raw_dataset url : https://drive.google.com/drive/folders/1s6LGibPnal6gMld5t63aK4J7hnVkNeDs
        '''
        data_path = self.data_path + '/data.pkl'
        distance_path = self.data_path + '/distanceMat_addIUUI.pkl'
        ici_path = self.data_path + '/ICI.pkl'

        with open(data_path, 'rb') as fs:
            data = pickle.load(fs)
        with open(distance_path, 'rb') as fs:
            distanceMat = pickle.load(fs)
        with open(ici_path, "rb") as fs:
            itemMat = pickle.load(fs)

        trainMat, testdata, _, categoryMat, _ = data
        userNum, itemNum = trainMat.shape
        userDistanceMat, itemDistanceMat, uiDistanceMat = distanceMat

        # trainMat
        trainMat_coo = trainMat.tocoo()
        trainMat_i, trainMat_j, trainMat_data = trainMat_coo.row, trainMat_coo.col, trainMat_coo.data

        # testdata
        testdata = np.array(testdata)

        # userDistanceMat
        userDistanceMat_coo = userDistanceMat.tocoo()
        userDistanceMat_i, userDistanceMat_j, userDistanceMat_data = userDistanceMat_coo.row, userDistanceMat_coo.col, userDistanceMat_coo.data

        # itemMat
        itemMat_coo = itemMat.tocoo()
        itemMat_i, itemMat_j, itemMat_data = itemMat_coo.row, itemMat_coo.col, itemMat_coo.data

        # uiDisantanceMat
        uiDistanceMat_coo = uiDistanceMat.tocoo()
        uiDistanceMat_i, uiDistanceMat_j, uiDistanceMat_data = uiDistanceMat_coo.row, uiDistanceMat_coo.col, uiDistanceMat_coo.data

        graph_data = {
            ('user', 'interact_train', 'item'): (t.tensor(trainMat_i), t.tensor(trainMat_j)),
            ('user', 'distance', 'user'): (t.tensor(userDistanceMat_i), t.tensor(userDistanceMat_j)),
            ('item', 'distance', 'item'): (t.tensor(itemMat_i), t.tensor(itemMat_j)),
            ('user+item', 'distance', 'user+item'): (t.tensor(uiDistanceMat_i), t.tensor(uiDistanceMat_j)),
            ('user', 'interact_test', 'item'): (t.tensor(testdata[:, 0]), t.tensor(testdata[:, 1]))
        }
        g = dgl.heterograph(graph_data)
        dgl.save_graphs(self.data_path + '/graph.bin', g)
        self.g_path = self.data_path + '/graph.bin'


    def download(self):
        # download raw data to local disk
        # path to store the file
        if os.path.exists(self.data_path):  # pragma: no cover
           pass
        else:
            file_path = os.path.join(self.raw_dir)
            # download file
            download(self.url, path=file_path)
        extract_archive(self.data_path, os.path.join(self.raw_dir, self.name))

    def process(self):
        # process raw data to graphs, labels, splitting masks
        g, _ = load_graphs(self.g_path)
        self._g = g

    def __getitem__(self, idx):
        # get one example by index
        return self._g[idx]

    def __len__(self):
        # number of data examples
        return 1

    def save(self):
        # save processed data to directory `self.save_path`
        pass

    def load(self):
        # load processed data from directory `self.save_path`
        pass

    def has_cache(self):
        # check whether there are processed data in `self.save_path`
        pass
