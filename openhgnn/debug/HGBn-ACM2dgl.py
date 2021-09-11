import os
import numpy as np
import torch
import dgl
import dgl.function as fn
import pandas as pd
import json

with open('ACM/info.dat','r') as load_f:
    info_dict = json.load(load_f)
    link_info = info_dict['link.dat']['link type']
    node_info = info_dict['node.dat']['node type']

print(link_info)
print(node_info)


# read nodes
nodes = pd.read_csv('ACM/node.dat',sep="\t",header=None)
edges = pd.read_csv('ACM/link.dat',sep="\t",header=None)
label_train = pd.read_csv('ACM/label.dat',sep="\t",header=None)
label_test = pd.read_csv('ACM/label.dat.test',sep="\t",header=None)

# add node features
paper_feature = []
author_feature = []
subject_feature = []

for i in range(len(nodes)):
    if(nodes.iloc[i,2] == 0):
        paper_feature.append(list(map(int, nodes.iloc[i,3].split(','))))
    elif (nodes.iloc[i, 2] == 1):
        author_feature.append(list(map(int, nodes.iloc[i,3].split(','))))
    elif (nodes.iloc[i, 2] == 2):
        subject_feature.append(list(map(int, nodes.iloc[i,3].split(','))))
    else:
        continue

meta_graphs = {}
for i in range(8):
    edge = edges[edges[2] == i]
    source_node = edge.iloc[:,0].values - np.min(edge.iloc[:,0].values)  # 使link.dat中的每种节点id从0开始
    target_node = edge.iloc[:,1].values - np.min(edge.iloc[:,1].values)
    meta_graphs[(node_info[str(link_info[str(i)]['start'])], link_info[str(i)]['meaning'], node_info[str(link_info[str(i)]['end'])])] = (torch.tensor(source_node), torch.tensor(target_node))

g = dgl.heterograph(meta_graphs)

# assign node featuer

g.nodes['paper'].data['h'] = torch.FloatTensor(paper_feature)

g.nodes['author'].data['h'] = torch.FloatTensor(author_feature)

g.nodes['subject'].data['h'] = torch.FloatTensor(subject_feature)

# get idx for train
train_paper_id = torch.LongTensor(label_train.iloc[:,0].values)
# get idx for test
test_paper_id = torch.LongTensor(label_test.iloc[:,0].values)

# get labels in graph for train and test
train_paper_label_value = torch.FloatTensor(label_train.iloc[:,3].values)
test_paper_label_value = torch.FloatTensor(label_test.iloc[:,3].values)

train_paper_mask_value = torch.full((1,len(train_paper_id)),True).bool()
test_paper_mask_value = torch.full((1,len(test_paper_id)),True).bool()

train_paper_mask = torch.full((len(paper_feature), ), False).bool()
test_paper_mask = torch.full((len(paper_feature), ), False).bool()

paper_label = torch.full((len(paper_feature),),-1.)

paper_label[train_paper_id] = train_paper_label_value
paper_label[test_paper_id] = test_paper_label_value

train_paper_mask[train_paper_id] = train_paper_mask_value
test_paper_mask[test_paper_id] = test_paper_mask_value

g.nodes['paper'].data['label'] = paper_label
g.nodes['paper'].data['train_mask'] = train_paper_mask
g.nodes['paper'].data['test_mask'] = test_paper_mask
# save graph
dgl.save_graphs("./acm.bin", g)

print(g)