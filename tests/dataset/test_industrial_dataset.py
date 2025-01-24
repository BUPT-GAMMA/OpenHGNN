from unittest import TestCase
from utils import datasets_info
from openhgnn.dataset import AliRCDDataset


class TestIndustrialDataset(TestCase):
    datasets_info(['MTWM'], task='link_prediction')


class TestDBLP4GTNDataset(TestCase):
    dataset = AliRCDDataset(session='small', force_reload=True)
    g = dataset[0]
    print('Information of AliRCDDataset dataset:')
    print(g)
    dataset = AliRCDDataset(session='small', force_reload=False)
    g = dataset[0]
    print('Information of AliRCDDataset dataset:')
    print(g)

