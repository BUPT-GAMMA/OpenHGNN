import torch as th
import torch.nn as nn

class HeteroEmbedLayer(nn.Module):
    r"""Embedding layer for featureless heterograph."""
    def __init__(self,
                 n_nodes,
                 embed_size,
                 embed_name='embed',
                 activation=None,
                 dropout=0.0):
        super(HeteroEmbedLayer, self).__init__()

        self.embed_size = embed_size
        self.embed_name = embed_name
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

        # create weight embeddings for each node for each relation
        self.embeds = nn.ParameterDict()
        for ntype, nodes in n_nodes.items():
            embed = nn.Parameter(th.FloatTensor(nodes, self.embed_size))
           # initrange = 1.0 / self.embed_size
            #nn.init.uniform_(embed, -initrange, initrange)
            nn.init.xavier_uniform_(embed, gain=nn.init.calculate_gain('relu'))
            self.embeds[ntype] = embed

    def forward(self, block=None):
        """Forward computation
        Parameters
        ----------
        block : DGLHeteroGraph, optional
            If not specified, directly return the full graph with embeddings stored in
            :attr:`embed_name`. Otherwise, extract and store the embeddings to the block
            graph and return.
        Returns
        -------
        DGLHeteroGraph
            The block graph fed with embeddings.
        """
        return self.embeds