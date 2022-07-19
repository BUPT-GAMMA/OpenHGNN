from unittest import TestCase
from utils import datasets_info


class TestIndustrialDataset(TestCase):
    datasets_info(['MTWM'], task='link_prediction')
