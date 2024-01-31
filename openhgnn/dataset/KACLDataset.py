import dgl
from dgl.data import DGLDataset
from dgl.data.utils import download, extract_archive
from dgl.data.utils import load_graphs
import os



class KACLDataset(DGLDataset):

    _prefix = 'https://s3.cn-north-1.amazonaws.com.cn/dgl-data/'
    _urls = {

    }

    def __init__(self, name, raw_dir=None, force_reload=False, verbose=True):
        self.data_path = './openhgnn/dataset/{}.zip'.format(name)
        self.g_path = './openhgnn/dataset/{}/graph.bin'.format(name)
        raw_dir = './'
        url = 'https://s3.cn-north-1.amazonaws.com.cn/dgl-data/' + 'dataset/{}.zip'.format(name)

        super(KACLDataset, self).__init__(name=name,
                                          url=url,
                                        raw_dir=raw_dir,
                                        force_reload=force_reload,
                                        verbose=verbose)



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
        self.g = load_graphs(self.g_path)[0]
        self._g = self.g[0]
        self._kg = self.g[2]
        self._g1 = self.g[1]
        self._kg1 = self.g[3]

    def __getitem__(self, idx):
        # get one example by index
        return self.g[idx]

    def __len__(self):
        # number of data examples
        return 4

    def save(self):
        # save processed data to directory `self.save_path`
        pass

    def load(self):
        # load processed data from directory `self.save_path`
        pass

    def has_cache(self):
        # check whether there are processed data in `self.save_path`
        pass

