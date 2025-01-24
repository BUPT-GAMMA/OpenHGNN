from unittest import TestCase
from openhgnn.dataset import IMDB4GTNDataset
from openhgnn.utils import *


class Test_plot_degree_dist(TestCase):
    dataset = IMDB4GTNDataset()
    g = dataset[0]
    plot_degree_dist(g)
    plot_degree_dist(g, save_path='plot_degree_dist.png')


class Test_plot_portion(TestCase):
    dataset = IMDB4GTNDataset()
    g = dataset[0]
    plot_portion(g)
    plot_portion(g, save_path='plot_portion.png')


class Test_plot_number_metapath(TestCase):
    dataset = IMDB4GTNDataset()
    g = dataset[0]
    plot_number_metapath(g, meta_paths_dict=dataset.meta_paths_dict)
    plot_number_metapath(g, meta_paths_dict=dataset.meta_paths_dict, save_path='plot_number_metapath.png')
