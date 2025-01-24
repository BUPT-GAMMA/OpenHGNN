from unittest import TestCase
from openhgnn.dataset import kg_lp_datasets, DBLP4GTNDataset, ACM4GTNDataset, IMDB4GTNDataset
from utils import datasets_info


class TestAcademicDataset(TestCase):
    datasets_info(['aifb', 'mutag', 'bgs', 'am'], task='node_classification')
    datasets_info(['acm4NSHE', 'acm4GTN', 'academic4HetGNN', 'acm_han', 'acm_han_raw', 'acm4HeCo', 'dblp',
                   'dblp4MAGNN', 'imdb4MAGNN', 'imdb4GTN', 'acm4NARS', 'yelp4HeGAN', 'DoubanMovie', 'Book-Crossing', 'yelp4HGSL'],
                  task='node_classification')
    datasets_info(['amazon4SLICE', 'MTWM', 'HNE-PubMed', 'HGBl-ACM', 'HGBl-DBLP', 'HGBl-IMDB'], task='link_prediction')
    datasets_info(kg_lp_datasets, task='link_prediction')

    datasets_info(['LastFM4KGCN'], task='recommendation')


class TestDBLP4GTNDataset(TestCase):
    dataset = DBLP4GTNDataset()
    g = dataset[0]
    print('Information of dblp4GTN dataset:')
    print(g)
    assert dataset.category == 'author'
    assert dataset.num_classes == 4


class TestACM4GTNDataset(TestCase):
    dataset = ACM4GTNDataset()
    g = dataset[0]
    print('Information of acm4GTN dataset:')
    print(g)
    assert dataset.category == 'paper'
    assert dataset.num_classes == 3


class TestIMDB4GTNDataset(TestCase):
    dataset = IMDB4GTNDataset()
    g = dataset[0]
    print('Information of imdb4GTN dataset:')
    print(g)
    assert dataset.category == 'movie'
    assert dataset.num_classes == 4
