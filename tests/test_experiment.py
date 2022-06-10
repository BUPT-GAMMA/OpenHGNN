from openhgnn import Experiment
from unittest import TestCase


class TestExperiment(TestCase):
    experiment = Experiment(model='RGCN', task='node_classification', dataset='imdb4GTN', gpu=-1, use_best_config=True)
    pass
