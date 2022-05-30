from unittest import TestCase
from openhgnn.dataset import DBLP4GTNDataset, ACM4GTNDataset, IMDB4GTNDataset


class TestDBLP4GTNDataset(TestCase):
    dataset = DBLP4GTNDataset()
    g = dataset[0]
    print(g)
    assert dataset.category == 'author'
    assert dataset.num_classes == 4


class TestACM4GTNDataset(TestCase):
    dataset = ACM4GTNDataset()
    g = dataset[0]
    print(g)
    assert dataset.category == 'paper'
    assert dataset.num_classes == 3


class TestIMDB4GTNDataset(TestCase):
    dataset = IMDB4GTNDataset()
    g = dataset[0]
    print(g)
    assert dataset.category == 'movie'
    assert dataset.num_classes == 4
