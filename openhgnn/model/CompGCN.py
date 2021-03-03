import dgl
import torch as th
import torch.nn as nn
import dgl.function as fn
import torch.nn.functional as F
from openhgnn.utils.utils import ccorr

class CompGraphConv(nn.Module):
    """One layer of simplified CompGCN."""

    def __init__(self,
                 in_dim,
                 out_dim,
                 comp_fn='sub',
                 activation=None,
                 batchnorm=False,
                 dropout=0):
        super(CompGraphConv, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.comp_fn = comp_fn
        self.actvation = activation
        self.batchnorm = batchnorm

        # define dropout layer
        self.dropout = nn.Dropout(dropout)

        # define batch norm layer
        if self.batchnorm:
            self.bn = nn.BatchNorm1d(out_dim)

        # define weights of 3 node matrices
        self.W_O = nn.Linear(self.in_dim, self.out_dim)
        self.W_I = nn.Linear(self.in_dim, self.out_dim)
        self.W_S = nn.Linear(self.in_dim, self.out_dim)

        # define weights of the 1 relation matrix
        self.W_R = nn.Linear(self.in_dim, self.out_dim)

    def forward(self, g, n_in_feats, r_feats):
        """Compute one layer of composition transfer for one relation only in a
        homogeneous graph with bidirectional edges.
        """
        with g.local_scope():
            # Assign values to source nodes. In a homogeneous graph, this is equal to
            # assigning them to all nodes.
            g.srcdata['h'] = n_in_feats

            # Assign feature to all edges with the same value, the r_feats.
            g.edata['h'] = F.embedding(g.edata[dgl.ETYPE].long(), r_feats)

            # Compute composition function in 4 steps
            # Step 1: compute composition by edge in the edge direction, and store results in edges.
            if self.comp_fn == 'sub':
                g.apply_edges(fn.u_sub_e('h', 'h', out='comp_h'))
            elif self.comp_fn == 'mul':
                g.apply_edges(fn.u_mul_e('h', 'h', out='comp_h'))
            elif self.comp_fn == 'ccorr':
                g.apply_edges(lambda edges: {'comp_h': ccorr(edges.src['h'], edges.data['h'])})
            else:
                raise Exception('Only supports sub, mul, and ccorr')

            # Step 2: use extracted edge direction to compute in and out edges
            comp_h = g.edata['comp_h']

            in_edges_idx = th.nonzero(g.edata['in_edges_mask'], as_tuple=False).squeeze()
            out_edges_idx = th.nonzero(g.edata['out_edges_mask'], as_tuple=False).squeeze()

            comp_h_O = self.W_O(comp_h[out_edges_idx])
            comp_h_I = self.W_I(comp_h[in_edges_idx])

            new_comp_h = th.zeros(comp_h.shape[0], self.out_dim).to(comp_h.device)
            new_comp_h[out_edges_idx] = comp_h_O
            new_comp_h[in_edges_idx] = comp_h_I

            g.edata['new_comp_h'] = new_comp_h

            # Step 3: sum comp results to both src and dst nodes
            g.update_all(fn.copy_e('new_comp_h', 'm'), fn.sum('m', 'comp_edge'))

            # Step 4: add results of self-loop
            if self.comp_fn == 'sub':
                comp_h_s = n_in_feats - r_feats[-1]
            elif self.comp_fn == 'mul':
                comp_h_s = n_in_feats * r_feats[-1]
            elif self.comp_fn == 'ccorr':
                comp_h_s = ccorr(n_in_feats, r_feats[-1])
            else:
                raise Exception('Only supports sub, mul, and ccorr')

            # Sum all of the comp results as output of nodes
            n_out_feats = self.W_S(comp_h_s) + g.ndata['comp_edge']

            # Compute relation output
            r_out_feats = self.W_R(r_feats)

            # Use batch norm
            if self.batchnorm:
                n_out_feats = self.bn(n_out_feats)

            # Use drop out
            n_out_feats = self.dropout(n_out_feats)
            r_out_feats = self.dropout(r_out_feats)

            # Use activation function
            if self.actvation is not None:
                n_out_feats = self.actvation(n_out_feats)
                r_out_feats = self.actvation(r_out_feats)

        return n_out_feats, r_out_feats


class CompGCN(nn.Module):
    """The model of the simplified CompGCN, without using basis vector, for a homogeneous graph.
    """

    def __init__(self,
                 in_dim,
                 hid_dim,
                 out_dim,
                 n_nodes,
                 n_rels,
                 num_layers=2,
                 comp_fn='sub',
                 dropout=0.0,
                 activation=None,
                 batchnorm=False):
        super(CompGCN, self).__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.n_rels = n_rels
        self.n_nodes = n_nodes
        self.num_layer = num_layers
        self.comp_fn = comp_fn
        self.dropout = dropout
        self.activation = activation
        self.batchnorm = batchnorm

        self.layers = nn.ModuleList()

        # Input layer and initial relation embedding
        # relations & nodes don't have the feature, so set the initial embedding
        self.n_embedding = nn.Parameter(th.FloatTensor(self.n_nodes, self.in_dim))
        self.r_embedding = nn.Parameter(th.FloatTensor(self.n_rels + 1, self.in_dim))

        self.layers.append(CompGraphConv(self.in_dim,
                                         self.hid_dim,
                                         comp_fn=self.comp_fn,
                                         activation=self.activation,
                                         batchnorm=self.batchnorm,
                                         dropout=self.dropout))

        # Hidden layers with n - 1 CompGraphConv layers
        for i in range(self.num_layer - 2):
            self.layers.append(CompGraphConv(self.hid_dim,
                                             self.hid_dim,
                                             comp_fn=self.comp_fn,
                                             activation=self.activation,
                                             batchnorm=self.batchnorm,
                                             dropout=self.dropout))

        # Output layer with the output class
        self.layers.append(CompGraphConv(self.hid_dim,
                                         self.out_dim,
                                         comp_fn=self.comp_fn))

        nn.init.xavier_uniform_(self.n_embedding)
        nn.init.xavier_uniform_(self.r_embedding)
    def forward(self, g):

        # For full graph training, directly use the graph

        # Forward of n layers of CompGraphConv
        r_feats = self.r_embedding
        n_feats = self.n_embedding

        for layer in self.layers:
            n_feats, r_feats = layer(g, n_feats, r_feats)

        return n_feats
