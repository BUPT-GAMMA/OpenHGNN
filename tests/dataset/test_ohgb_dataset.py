from unittest import TestCase
from openhgnn.dataset import ohgbn_datasets, ohgbl_datasets
from utils import datasets_info


class TestHGBnDataset(TestCase):
    datasets_info(ohgbn_datasets, task='node_classification')


class TestHGBlDataset(TestCase):
    datasets_info(ohgbl_datasets, task='link_prediction')
