import os
from dgl.data.utils import download, extract_archive
from dgl.data import DGLDataset
from dgl.data.utils import load_graphs


class AcademicDataset(DGLDataset):

    _prefix = 'https://s3.cn-north-1.amazonaws.com.cn/dgl-data/'
    _urls = {
        'academic4HetGNN': 'dataset/academic4HetGNN.zip',
        'acm4GTN': 'dataset/acm4GTN.zip',
        'acm4NSHE': 'dataset/acm4NSHE.zip',
        'acm4NARS': 'dataset/acm4NARS.zip',
        'acm4HeCo': 'dataset/acm4HeCo.zip',
        'imdb4MAGNN': 'dataset/imdb4MAGNN.zip',
        'imdb4GTN': 'dataset/imdb4GTN.zip',
        'DoubanMovie': 'dataset/DoubanMovie.zip',
        'dblp4MAGNN': 'dataset/dblp4MAGNN.zip',
        'yelp4HeGAN': 'dataset/yelp4HeGAN.zip',
        'yelp4rec': 'dataset/yelp4rec.zip',
        'HNE-PubMed': 'dataset/HNE-PubMed.zip',
        'amazon4SLICE': 'dataset/amazon4SLICE.zip'
    }

    def __init__(self, name, raw_dir=None, force_reload=False, verbose=True):
        assert name in ['acm4GTN', 'acm4NSHE', 'academic4HetGNN', 'imdb4MAGNN', 'imdb4GTN', 'HNE-PubMed',
                        'DoubanMovie', 'dblp4MAGNN', 'acm4NARS', 'acm4HeCo', 'yelp4rec', 'yelp4HeGAN', 'amazon4SLICE']
        self.data_path = './openhgnn/' + self._urls[name]
        self.g_path = './openhgnn/dataset/' + name + '/graph.bin'
        raw_dir = './openhgnn/dataset'
        url = self._prefix + self._urls[name]
        super(AcademicDataset, self).__init__(name=name,
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