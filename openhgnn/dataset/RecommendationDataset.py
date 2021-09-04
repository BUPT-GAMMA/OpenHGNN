import dgl
import torch as th
from . import BaseDataset, register_dataset
from dgl.data.utils import load_graphs


@register_dataset('recommendation')
class RecommendationDataset(BaseDataset):
    """

    """
    def __init__(self,):
        super(RecommendationDataset, self).__init__()


@register_dataset('hin_recommendation')
class HINRecommendation(RecommendationDataset):
    def __init__(self, dataset_name):
        super(HINRecommendation, self).__init__()
        self.dataset_name = dataset_name
        self.num_neg = 20
        #self.neg_dir = os.path.join(self.raw_dir, dataset_name, 'neg_{}.bin'.format(self.num_neg))
        self.g = self.load_HIN('./openhgnn/dataset/yelp.bin')
        self.target_link = 'user-item'
        self.target_link_r = 'item-user'
        self.has_feature = False

        if not os.path.isfile(self.neg_dir):
            self.generate_negative()
        neg_g, _ = dgl.load_graphs(self.neg_dir)
        self.neg_g = neg_g[0]

    def load_HIN(self, dataset_name):
        g, _ = dgl.load_graphs(dataset_name)
        return g[0]

    def process(self, g):
        # sub 1 for every node
        new = {}
        for etype in g.canonical_etypes:
            edges = g.edges(etype=etype)
            new[etype] = (edges[0]-1, edges[1]-1)
        hg = dgl.heterograph(new)
        from dgl.data.utils import save_graphs
        save_graphs("./yelp.bin", hg)

    def get_idx(self):
        val_mask = self.g.edges[self.target_link].data['val_mask'].squeeze()
        val_index = th.nonzero(val_mask).squeeze()
        val_edge = self.g.find_edges(val_index, self.target_link)

        test_mask = self.g.edges[self.target_link].data['test_mask'].squeeze()
        test_index = th.nonzero(test_mask).squeeze()
        test_edge = self.g.find_edges(test_index, self.target_link)

        val_graph = dgl.heterograph({('user', 'user-item', 'item'): val_edge},
                                         {ntype: self.g.number_of_nodes(ntype) for ntype in ['user', 'item']})
        test_graph = dgl.heterograph({('user', 'user-item', 'item'): test_edge},
                                          {ntype: self.g.number_of_nodes(ntype) for ntype in ['user', 'item']})

        train_graph = dgl.remove_edges(self.g, th.cat((val_index, test_index)), self.target_link)
        train_graph = dgl.remove_edges(train_graph, th.cat((val_index, test_index)), self.target_link_r)
        return train_graph, val_graph, test_graph

    def generate_negative(self):
        k = self.num_neg
        e = self.g.edges(etype=self.target_link)
        neg_src = []
        neg_dst = []
        for i in range(self.g.number_of_edges(self.target_link)):
            src = e[0][i]
            exp = self.g.successors(src, etype=self.target_link)
            dst = th.randint(low=0, high=self.g.number_of_nodes('item'), size=(k,))
            for d in range(len(dst)):
                while dst[d] in exp:
                    dst[d] = th.randint(low=0, high=self.g.number_of_nodes('item'), size=(1,))
            src = src.repeat_interleave(k)
            neg_src.append(src)
            neg_dst.append(dst)
        neg_edge = (th.cat(neg_src), th.cat(neg_dst))
        neg_g = dgl.heterograph({('user', 'user-item', 'item'): neg_edge},
                                {ntype: self.g.number_of_nodes(ntype) for ntype in ['user', 'item']})
        dgl.save_graphs(self.neg_dir, neg_g)


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