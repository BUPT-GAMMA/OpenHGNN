import torch.nn as nn
import dgl.nn as dglnn
import torch.nn.functional as F
from openhgnn.dataset import IMDB4GTNDataset, HGBDataset
from openhgnn import Experiment


# Define a Heterograph Conv model. The example is from https://docs.dgl.ai/guide/training-node.html#heterogeneous-graph

class MyRGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__()
        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feats, hid_feats)
            for rel in rel_names}, aggregate='sum')
        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, out_feats)
            for rel in rel_names}, aggregate='sum')

    def forward(self, graph, inputs):
        # inputs are features of nodes
        h = self.conv1(graph, inputs)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(graph, h)
        return h


def node_classification():
    # node_classification task for imdb dataset
    dataset = IMDB4GTNDataset()
    hg = dataset[0]
    model = MyRGCN(64, 64, dataset.num_classes, hg.etypes)
    experiment = Experiment(model=model, dataset='imdb4GTN', task='node_classification', gpu=-1)
    experiment.run()


def link_prediction():
    # link prediction task for amazon dataset
    dataset = HGBDataset(name='HGBl-amazon')
    hg = dataset[0]
    model = MyRGCN(64, 64, 64, hg.etypes)
    experiment = Experiment(model=model, dataset='HGBl-amazon', task='link_prediction', gpu=-1)
    experiment.run()


if __name__ == '__main__':
    node_classification()
    link_prediction()
