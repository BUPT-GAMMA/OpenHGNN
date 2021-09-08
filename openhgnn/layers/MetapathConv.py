"""Heterograph NN modules"""
import torch as th
import torch.nn as nn


class MetapathConv(nn.Module):
    def __init__(self, meta_paths, mods, macro_func, **kargs):
        super(MetapathConv, self).__init__()
        # One GAT layer for each meta path based adjacency matrix
        self.mods = nn.ModuleList(mods)
        self.meta_paths = meta_paths
        self.SemanticConv = macro_func

    def forward(self, g_list, h):
        semantic_embeddings = []

        for i, meta_path in enumerate(self.meta_paths):
            new_g = g_list[meta_path]
            semantic_embeddings.append(self.mods[i](new_g, h).flatten(1))
        #semantic_embeddings = th.stack(semantic_embeddings, dim=1)  # (N, M, D * K)

        return self.SemanticConv(semantic_embeddings)  # (N, D * K)