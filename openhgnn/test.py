import dgl
import torch
import pickle
import numpy as np

# from dgl.data import CoraGraphDataset
#
# dataset = CoraGraphDataset()
# graph = dataset[0]
dataset = 'ACM'

with open('./data/' + dataset + '/node_features.pkl', 'rb') as f:
    node_features = pickle.load(f)
with open('data/' + dataset + '/edges.pkl', 'rb') as f:
    edges = pickle.load(f)
with open('data/' + dataset + '/labels.pkl', 'rb') as f:
    labels = pickle.load(f)
num_nodes = edges[0].shape[0]

for i, edge in enumerate(edges):
    if i == 0:
        A = torch.from_numpy(edge.todense()).type(torch.FloatTensor).unsqueeze(-1)
    else:
        A = torch.cat([A, torch.from_numpy(edge.todense()).type(torch.FloatTensor).unsqueeze(-1)], dim=-1)
A = torch.cat([A, torch.eye(num_nodes).type(torch.FloatTensor).unsqueeze(-1)], dim=-1)

node_features = torch.from_numpy(node_features).type(torch.FloatTensor)
train_node = torch.from_numpy(np.array(labels[0])[:, 0]).type(torch.LongTensor)
train_target = torch.from_numpy(np.array(labels[0])[:, 1]).type(torch.LongTensor)
valid_node = torch.from_numpy(np.array(labels[1])[:, 0]).type(torch.LongTensor)
valid_target = torch.from_numpy(np.array(labels[1])[:, 1]).type(torch.LongTensor)
test_node = torch.from_numpy(np.array(labels[2])[:, 0]).type(torch.LongTensor)
test_target = torch.from_numpy(np.array(labels[2])[:, 1]).type(torch.LongTensor)

