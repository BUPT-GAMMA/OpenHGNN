"""Heterograph NN modules"""
import torch as th
import torch.nn as nn


class MetapathConv(nn.Module):
    r"""
    Description
    ------------
    MetapathConv is a aggregation function based on meta-path.
    We could choose Attention/ APPNP or any GraphConvLayer to aggregate node features.
    After that we will get embeddings based on all meta-path and fusion them.

    .. math::
        \mathbf{Z}=\mathcal{F}(Z^{\Phi_1},Z^{\Phi_2},...,Z^{\Phi_p})=\mathcal{F}(f(H,\Phi_1),f(H,\Phi_2),...,f(H,\Phi_p))

    where :math:`\mathcal{F}` denotes semantic fusion function, such as semantic-attention. :math:`\Phi_i` denotes meta-path and
    :math:`f` denotes the aggregation way, such as GAT, APPNP.

    Parameters
    ------------
    meta_paths : list[tuple]
        contain multiple meta-paths.
    mods : nn.ModuleList
        aggregation function
    macro_func : callable aggregation func
        A semantic aggregation way, e.g. 'mean', 'max', 'sum' or 'attention'


    """
    def __init__(self, meta_paths, mods, macro_func, **kargs):
        super(MetapathConv, self).__init__()
        # One GAT layer for each meta path based adjacency matrix
        self.mods = nn.ModuleList(mods)
        self.meta_paths = meta_paths
        self.SemanticConv = macro_func

    def forward(self, g_list, h):
        r"""
        Parameters
        -----------
        g_list : list of DGLGraph
            A list of DGLGraph extracted by metapaths.
        h : Tensor
            The input features

        Returns
        --------
        h : tensor
            The output features
        """
        semantic_embeddings = []

        for i, meta_path in enumerate(self.meta_paths):
            new_g = g_list[meta_path]
            semantic_embeddings.append(self.mods[i](new_g, h).flatten(1))
        #semantic_embeddings = th.stack(semantic_embeddings, dim=1)  # (N, M, D * K)

        return self.SemanticConv(semantic_embeddings)  # (N, D * K)