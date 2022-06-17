from unittest import TestCase
from openhgnn.dataset import Amazon4GATNEDataset, Twitter4GATNEDataset, Youtube4GATNEDataset


class TestAmazon4GATNEDataset(TestCase):
    dataset = Amazon4GATNEDataset()
    g = dataset[0]
    print(g)


class TestTwitter4GATNEDataset(TestCase):
    dataset = Twitter4GATNEDataset()
    g = dataset[0]
    print(g)


class TestYoutube4GATNEDataset(TestCase):
    dataset = Youtube4GATNEDataset()
    g = dataset[0]
    print(g)
