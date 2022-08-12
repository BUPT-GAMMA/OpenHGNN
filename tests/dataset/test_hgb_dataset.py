from unittest import TestCase
from openhgnn.dataset import hgbn_datasets, hgbl_datasets
from utils import datasets_info


class TestHGBnDataset(TestCase):
    datasets_info(hgbn_datasets, task='node_classification')


class TestHGBlDataset(TestCase):
    datasets_info(hgbl_datasets, task='link_prediction')
