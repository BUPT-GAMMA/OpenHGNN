"""Heterograph NN modules"""
import torch as th
import torch.nn as nn


class MetapathConv(nn.Module):
    r"""
    MetapathConv is an aggregation function based on meta-path, which is similar with `dgl.nn.pytorch.HeteroGraphConv`.
    We could choose Attention/ APPNP or any GraphConvLayer to aggregate node features.
    After that we will get embeddings based on different meta-paths and fusion them.

    .. math::
        \mathbf{Z}=\mathcal{F}(Z^{\Phi_1},Z^{\Phi_2},...,Z^{\Phi_p})=\mathcal{F}(f(H,\Phi_1),f(H,\Phi_2),...,f(H,\Phi_p))

    where :math:`\mathcal{F}` denotes semantic fusion function, such as semantic-attention. :math:`\Phi_i` denotes meta-path and
    :math:`f` denotes the aggregation function, such as GAT, APPNP.

    Parameters
    ------------
    meta_paths_dict : dict[str, list[tuple(meta-path)]]
        contain multiple meta-paths.
    mods : nn.ModuleDict
        aggregation function
    macro_func : callable aggregation func
        A semantic aggregation way, e.g. 'mean', 'max', 'sum' or 'attention'

    """

    def __init__(self, meta_paths_dict, mods, macro_func, **kargs):
        super(MetapathConv, self).__init__()
        # One GAT layer for each meta path based adjacency matrix
        self.mods = mods
        self.meta_paths_dict = meta_paths_dict
        self.SemanticConv = macro_func

    def forward(self, g_dict, h_dict):
        r"""
        Parameters
        -----------
        g_dict : dict[str: dgl.DGLGraph]
            A dict of DGLGraph(full batch) or DGLBlock(mini batch) extracted by metapaths.
        h_dict : dict[str: torch.Tensor]
            The input features

        Returns
        --------
        h : dict[str: torch.Tensor]
            The output features dict
        """
        outputs = {g.dsttypes[0]: [] for s, g in g_dict.items()}

        for meta_path_name, meta_path in self.meta_paths_dict.items():
            new_g = g_dict[meta_path_name]

            # han minibatch
            if h_dict.get(meta_path_name) is not None:
                h = h_dict[meta_path_name][new_g.srctypes[0]]
            # full batch
            else:
                h = h_dict[new_g.srctypes[0]]
            outputs[new_g.dsttypes[0]].append(self.mods[meta_path_name](new_g, h).flatten(1))
        # semantic_embeddings = th.stack(semantic_embeddings, dim=1)  # (N, M, D * K)
        # Aggregate the results for each destination node type
        rsts = {}
        for ntype, ntype_outputs in outputs.items():
            if len(ntype_outputs) != 0:
                rsts[ntype] = self.SemanticConv(ntype_outputs)  # (N, D * K)
        return rsts
