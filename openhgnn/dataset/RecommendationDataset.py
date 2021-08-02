import dgl
import numpy as np
from dgl.data.knowledge_graph import load_data
import torch as th
from openhgnn.dataset import BaseDataset, register_dataset
from . import AcademicDataset
from dgl.data.utils import load_graphs


@register_dataset('recommendation')
class RecommendationDataset(BaseDataset):
    """

    """
    def __init__(self,):
        super(RecommendationDataset, self).__init__()


@register_dataset('test_link_prediction')
class Test_Recommendation(RecommendationDataset):
    def __init__(self, dataset_name):
        super(RecommendationDataset, self).__init__()
        self.g = self.load_HIN('./openhgnn/debug/data.bin')
        self.target_link = 'user-item'
        self.has_feature = False
        self.preprocess()
        #self.generate_negative()

    def load_HIN(self, dataset_name):
        g, _ = load_graphs(dataset_name)
        return g[0]

    def preprocess(self):
        test_mask = self.g.edges[self.target_link].data['test_mask']
        index = th.nonzero(test_mask).squeeze()
        self.test_edge = self.g.find_edges(index, self.target_link)
        self.pos_test_graph = dgl.heterograph({('user', 'user-item', 'item'): self.test_edge}, {ntype: self.g.number_of_nodes(ntype) for ntype in ['user', 'item']})
        self.g.remove_edges(index, self.target_link)
        self.g.remove_edges(index, 'item-user')
        self.neg_test_graph, _ = dgl.load_graphs('./openhgnn/debug/neg.bin')
        self.neg_test_graph = self.neg_test_graph[0]
        return

    def generate_negative(self):
        k = 99
        e = self.pos_test_graph.edges()
        neg_src = []
        neg_dst = []
        for i in range(self.pos_test_graph.number_of_edges()):
            src = e[0][i]
            exp = self.pos_test_graph.successors(src)
            dst = th.randint(high=self.g.number_of_nodes('item'), size=(k,))
            for d in range(len(dst)):
                while dst[d] in exp:
                    dst[d] = th.randint(high=self.g.number_of_nodes('item'), size=(1,))
            src = src.repeat_interleave(k)
            neg_src.append(src)
            neg_dst.append(dst)
        neg_edge = (th.cat(neg_src), th.cat(neg_dst))
        neg_graph = dgl.heterograph({('user', 'user-item', 'item'): neg_edge}, {ntype: self.g.number_of_nodes(ntype) for ntype in ['user', 'item']})
        dgl.save_graphs('./openhgnn/debug/neg.bin', neg_graph)