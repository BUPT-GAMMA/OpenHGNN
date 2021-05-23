import dgl
import torch as th
import torch.nn as nn
import dgl.function as fn
import torch.nn.functional as F
from openhgnn.utils.utils import ccorr
from . import BaseModel, register_model
from openhgnn.utils.dgl_graph import edata_in_out_mask
from . import HeteroEmbedLayer
from ..utils import get_nodes_dict
'''
Here, we present the implementation details for each task used for evaluation in the paper. 
For all the tasks, we used COMPGCN build on PyTorch geometric framework (Fey & Lenssen, 2019).

Link Prediction: For evaluation, 200-dimensional embeddings for node and relation embeddings are used. 
    For selecting the best model we perform a hyperparameter search using the validation data over the values listed in Table 8. 
    For training link prediction models, we use the standard binary cross entropy loss with label smoothing Dettmers et al. (2018).

Node Classification: Following Schlichtkrull et al. (2017), we use 10% training data as validation for selecting the best model for both the datasets. 
    We restrict the number of hidden units to 32. We use cross-entropy loss for training our model.

For all the experiments, training is done using Adam optimizer (Kingma & Ba, 2014) and Xavier initialization (Glorot & Bengio, 2010) is used for initializing parameters.
'''


@register_model('CompGCN')
class CompGCN(BaseModel):
    """
    The models of the simplified CompGCN, without using basis vector, for a heterogeneous graph.
    """
    @classmethod
    def build_model_from_args(cls, args, hg):
        return cls(args.h_dim, args.h_dim, args.out_dim, get_nodes_dict(hg), len(hg.etypes),
            args.n_layers, comp_fn=args.comp_fn, dropout=args.dropout
        )

    def __init__(self, in_dim, hid_dim, out_dim, n_nodes, n_rels, num_layers=2, comp_fn='sub', dropout=0.0,
                 activation=F.relu, batchnorm=True):
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
        self.h_n_dict = HeteroEmbedLayer(n_nodes, self.in_dim)

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


        nn.init.xavier_uniform_(self.r_embedding)

    def forward(self, hg, n_feats=None):

        # For full graph training, directly use the graph
        if n_feats is None:
            # full graph training
            n_feats = self.h_n_dict()

        # Forward of n layers of CompGraphConv
        r_feats = self.r_embedding

        if hasattr(hg, 'ntypes'):

            # full graph training
            for layer in self.layers:
                n_feats, r_feats = layer(hg, n_feats, r_feats)
        else:
            # minibatch training
            pass

        return n_feats


    def preprocess(self, hg):
        edata_in_out_mask(hg)

    def trainer(self):
        pass

    def node_classification_loss(self, hg, label, mask, category):
        #assert mask.shape[0] == label.shape[0]
        pred = self.forward(hg)[category]
        return self.loss_fn(pred[mask], label[mask])


class CompGraphConv(nn.Module):
    """One layer of simplified CompGCN."""

    def __init__(self, in_dim, out_dim, comp_fn='sub', activation=None, batchnorm=False, dropout=0):
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

    def forward(self, hg, n_in_feats, r_feats):
        """Compute one layer of composition transfer for one relation only in a
        homogeneous graph with bidirectional edges.
        """
        with hg.local_scope():
            # Assign values to source nodes. In a homogeneous graph, this is equal to
            # assigning them to all nodes.

            # Assign feature to all edges with the same value, the r_feats.
            outputs = {}
            for i, e in enumerate(hg.canonical_etypes):
                stype, etype, dtype = e
                h_e = r_feats[i]
                # hg.edata['h'] = {e: r_feats[i].expand(n, -1)}
                rel_graph = hg[stype, etype, dtype]

                rel_graph.edata['h'] = h_e.expand(rel_graph.num_edges(), -1)
                rel_graph.srcdata['h'] = n_in_feats[stype]

                # Compute composition function in 4 steps
                # Step 1: compute composition by edge in the edge direction, and store results in edges.
                if self.comp_fn == 'sub':
                    rel_graph.apply_edges(fn.u_sub_e('h', 'h', out='comp_h'))
                elif self.comp_fn == 'mul':
                    rel_graph.apply_edges(fn.u_mul_e('h', 'h', out='comp_h'))
                elif self.comp_fn == 'ccorr':
                    rel_graph.apply_edges(lambda edges: {'comp_h': ccorr(edges.src['h'], edges.data['h'])})
                else:
                    raise Exception('Only supports sub, mul, and ccorr')

                # Step 2: use extracted edge direction to compute in and out edges
                comp_h = rel_graph.edata['comp_h']
                # Specific
                if etype[:4] == 'rev-':
                    I_O_flag = 'I'
                    comp_h = self.W_I(comp_h)
                else:
                    comp_h = self.W_O(comp_h)
                rel_graph.edata['new_comp_h'] = comp_h

                # Step 3: sum comp results to both src and dst nodes
                rel_graph.update_all(fn.copy_e('new_comp_h', 'm'), fn.sum('m', 'comp_edge'))
                temp = rel_graph.ndata['comp_edge']
                if isinstance(temp, dict):
                    temp = temp[dtype]
                if outputs.get(dtype) is None:
                    outputs[dtype] = temp
                else:
                    outputs[dtype].add_(temp)

            for n in hg.ntypes:
                # Step 4: add results of self-loop
                if self.comp_fn == 'sub':
                    h_self = self.W_S(n_in_feats[n] - r_feats[-1])
                elif self.comp_fn == 'mul':
                    h_self = self.W_S(n_in_feats[n] * r_feats[-1])
                elif self.comp_fn == 'ccorr':
                    h_self = self.W_S(n_in_feats[n], r_feats[-1])
                else:
                    raise Exception('Only supports sub, mul, and ccorr')
                h_self.add_(outputs[n])
                # Compute relation output

                # Use batch norm
                if self.batchnorm:
                    h_self = self.bn(h_self)

                # Use drop out
                n_out_feats = self.dropout(h_self)
                # Use activation function
                if self.actvation is not None:
                    n_out_feats = self.actvation(n_out_feats)

                outputs[n] = n_out_feats

        r_out_feats = self.W_R(r_feats)
        r_out_feats = self.dropout(r_out_feats)
        if self.actvation is not None:
            r_out_feats = self.actvation(r_out_feats)
        return outputs, r_out_feats