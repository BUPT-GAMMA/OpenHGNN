import os
from dgl.data.utils import download, extract_archive
from dgl.data import DGLDataset
from dgl.data.utils import load_graphs


class HGBDataset(DGLDataset):

    _prefix = 'https://s3.cn-north-1.amazonaws.com.cn/dgl-data/'
    _urls = {

    }

    def __init__(self, name, raw_dir=None, force_reload=False, verbose=True):
        assert name in ['HGBn-ACM', 'HGBn-DBLP', 'HGBn-Freebase', 'HGBn-IMDB',
                        'HGBl-amazon', 'HGBl-LastFM', 'HGBl-PubMed', 'HGBl-twitter', 'HGBl-twitter_test']
        self.prefix_task = name[:4]
        # HGBn means node classification
        # HGBl means link prediction
        self.data_path = './openhgnn/dataset/{}.zip'.format(self.prefix_task)
        self.g_path = './openhgnn/dataset/{}/{}.bin'.format(self.prefix_task, name)
        raw_dir = './openhgnn/dataset'
        url = self._prefix + 'dataset/{}.zip'.format(self.prefix_task)
        super(HGBDataset, self).__init__(name=name,
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
        extract_archive(self.data_path, os.path.join(self.raw_dir, self.prefix_task))

    def process(self):
        # process raw data to graphs, labels, splitting masks
        g, _ = load_graphs(self.g_path)
        self._g = g[0]

    def __getitem__(self, idx):
        # get one example by index
        assert idx == 0, "This dataset has only one graph"
        return self._g

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