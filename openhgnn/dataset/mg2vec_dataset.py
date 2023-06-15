import os
from dgl.data.utils import download, extract_archive
from dgl.data import DGLDataset
from dgl.data.utils import load_graphs
from dgl.data.utils import save_graphs


# Download and process the dataset for mg2vec

class Mg2vecDataSet(DGLDataset):
    _prefix = 'https://s3.cn-north-1.amazonaws.com.cn/dgl-data/'
    _urls = {
        'dblp4Mg2vec_4': 'dataset/openhgnn/dblp4Mg2vec_4.zip',
        'dblp4Mg2vec_5': 'dataset/openhgnn/dblp4Mg2vec_5.zip',
    }

    def __init__(self, name, raw_dir=None, force_reload=False, verbose=False):
        assert name in ['dblp4Mg2vec_4', 'dblp4Mg2vec_5']
        raw_dir = './openhgnn/dataset'
        self.data_path = './openhgnn/dataset/' + name + '.zip'
        self.g_path = './openhgnn/dataset/' + name + '/graph.bin'
        url = self._prefix + self._urls[name]
        super(Mg2vecDataSet, self).__init__(name=name,
                                            url=url,
                                            raw_dir=raw_dir,
                                            force_reload=force_reload,
                                            verbose=verbose)

    def download(self):
        if os.path.exists(self.data_path):
            pass
        else:
            file_path = os.path.join(self.raw_dir)
            download(self.url, path=file_path)
        tem_path = os.path.join(self.raw_dir, self.name)
        extract_archive(self.data_path, tem_path)

    def process(self):
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
