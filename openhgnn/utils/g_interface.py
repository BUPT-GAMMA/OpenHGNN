from dataset import build_dataset
from dataset import AcademicDataset
from dataset import Mg2vecDataSet
import dgl
from dgl.data import DGLDataset
from dgl import transforms as T
from dataset import AsLinkPredictionDataset, AsNodeClassificationDataset

def construct_dataset(name,task):
    if task == 'node_classification':
        if name in [
                    'acm4NSHE', 'acm4GTN', 'academic4HetGNN', 'acm4HetGNN','acm_han_raw',
                    'acm4HeCo', 'dblp4MAGNN', 'imdb4MAGNN', 'imdb4GTN',
                    'acm4NARS', 'yelp4HeGAN',
                    'HGBn-ACM', 'HGBn-DBLP', 'HGBn-Freebase', 'HGBn-IMDB',
                    'alircd_session1',
                    'ohgbn-Freebase', 'ohgbn-yelp2', 'ohgbn-acm', 'ohgbn-imdb',
                    'dblp4GTN',
                    'HNE-PubMed', #Fixed: should be in 'node_classification' task, instead of 'link_prediction'
                    'ogbn-mag',# Download speed low ...
                    'aifb', 'mutag', 'bgs', 'am',
                    'alircd_small',
                    'ICDM',
                    ]:
            if name == "acm4HetGNN":
                name = 'academic4HetGNN'
            return build_dataset(dataset=name, task=task, logger=None)
        
        elif name  == "academic4HetGNN":
            return AcademicDataset("academic4HetGNN")
        
        # Fixed: change 'dblp4Mg2vec_4'、'dblp4Mg2vec_5' into 'dblp4mg2vec_4'、'dblp4mg2vec_5'.
        # No model named 'mg2vec', corresponding dataset had no node labels.

    elif task == 'link_prediction':
        #amazon4SLICE can't use for now.
        if name in ['amazon4SLICE', 
                    'MTWM', 'HGBl-ACM',
                    'HGBl-DBLP', 'HGBl-IMDB',
                    'wn18', 'FB15k', 'FB15k-237',
                    'HGBl-amazon', 'HGBl-LastFM', 'HGBl-PubMed',
                    'ohgbl-MTWM', 'ohgbl-yelp1', 'ohgbl-yelp2', 'ohgbl-Freebase']:   
            return build_dataset(dataset=name, task=task, logger=None)
        elif name in ['DoubanMovie']: # Fixed, no class registered in hin_link_prediction
            ds = AcademicDataset('DoubanMovie')
            ds.g = ds[0]
            return ds
        

    elif task == 'recommendation': 
        if name in ['LastFM4KGCN','yelp4rec']:
            return build_dataset(dataset=name, task=task, logger=None)

    elif task == 'edge_classification':
        if name in ['dblp4Mg2vec_4', 'dblp4Mg2vec_5']:
            ds = Mg2vecDataSet(name=name)
            ds.g = ds[0]
            return ds
class MyDataset(DGLDataset):
    def __init__(self,name,path,reverse=True):
        self.path = path
        self.reverse = reverse
        super().__init__(name=name)
        

    def process(self):
        gs, _ = dgl.load_graphs(self.path)
        if(self.reverse):
            self._g = T.AddReverse()(gs[0])
        else:
            self._g = gs[0]

    def __getitem__(self, idx):
        return self._g

    def __len__(self):
        return 1

