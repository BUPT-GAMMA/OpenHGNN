from unittest import TestCase
from openhgnn.dataset import ICDM2022Dataset


class TestICDM2022Dataset(TestCase):
    dataset = ICDM2022Dataset()
    g = dataset[0]
    print(g)

