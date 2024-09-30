from unittest import TestCase
from openhgnn.dataset import IMDB4GTNDataset, DBLP4GTNDataset
import torch as th
import dgl
from dgl.sparse import spspmm
from dgl.sparse import SparseMatrix
from dgl.sparse import SparseMatrix as sp

def meta_path_heterophily(g, meta_paths_dict, strength=1):
    meta_path_names = []
    meta_path_nums = []
    connectivity_strength = []
    homogeneity = []
    heterophily = []
    src_label = g.srcdata['label']
    dst_label = g.dstdata['label']

    #convert etype  [src,edge,dst] -> (src,edge,dst)
    etype_is_list = False
    new_meta_paths_dict = {}
    for meta_path_name, meta_path in meta_paths_dict.items():
        new_meta_paths_dict[meta_path_name] = []
        for i, etype in enumerate(meta_path):
            if(isinstance(etype,list)):
                etype_is_list = True
                _new_type = (etype[0],etype[1],etype[2])
                new_meta_paths_dict[meta_path_name].append(_new_type)
    if  etype_is_list:
        meta_paths_dict = new_meta_paths_dict
    for meta_path_name, meta_path in meta_paths_dict.items():
        meta_path_names.append(meta_path_name)
        src, dst = meta_path[0][0], meta_path[-1][-1]
        for i, etype in enumerate(meta_path):
            if i == 0:
                adj = g.adj(etype=etype)
            else:
                #adj = th.sparse.mm(adj, g.adj(etype=etype))
                adj = spspmm(adj, g.adj(etype=etype))
        meta_path_nums.append(int(SparseMatrix.sum(adj)))
        # print(adj)

        # 计算连通性并记录满足条件节点坐标
        row, col = [], []
        adj = sp.to_dense(adj)
        adj = adj.to_sparse()
        length = len(adj._values())
        conn = 0
        for i in range(length):
            if adj._values()[i] >= strength:
                conn += 1
                row.append(int(adj._indices()[0][i]))
                col.append(int(adj._indices()[1][i]))
        connectivity_strength.append(1-conn/length)

        # 计算同构性
        length = len(row)
        slabel = src_label[src].numpy().tolist()
        dlabel = dst_label[dst].numpy().tolist()
        ans = 0
        for i in range(length):
            if slabel[row[i]] == dlabel[col[i]]:
                ans += 1
        homogeneity.append(ans/conn)
        heterophily.append(1-ans/conn)


    i = 0
    for meta_path_name in meta_paths_dict.keys():
        print(meta_path_name,":")
        print("edges:",meta_path_nums[i])
        print("heterophily :",heterophily)
        print("edge/total:",connectivity_strength[i])
        print()
        i += 1
    return meta_path_nums, heterophily, connectivity_strength


# if __name__ == "__main__":
#     dataset = DBLP4GTNDataset()
#     # dataset = IMDB4GTNDataset()
#     g = dataset[0]
#     meta_path_nums, connectivity_strength, homogeneity = number_meta_path(g, meta_paths_dict=dataset.meta_paths_dict, strength=2)
#     print(meta_path_nums)
#     print(connectivity_strength)
#     print(homogeneity)

#     meta_path_nums, connectivity_strength, homogeneity = number_meta_path(g, meta_paths_dict=dataset.meta_paths_dict, strength=4)
#     print(meta_path_nums)
#     print(connectivity_strength)
#     print(homogeneity)

#     meta_path_nums, connectivity_strength, homogeneity = number_meta_path(g, meta_paths_dict=dataset.meta_paths_dict, strength=8)
#     print(meta_path_nums)
#     print(connectivity_strength)
#     print(homogeneity)