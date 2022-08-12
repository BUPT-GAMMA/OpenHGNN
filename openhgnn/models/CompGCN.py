import torch as th
import torch.nn as nn
import dgl.nn as dglnn
import torch.nn.functional as F
from . import BaseModel, register_model
from openhgnn.layers.micro_layer import CompConv
from ..utils.dgl_graph import edata_in_out_mask
from ..utils import get_nodes_dict
from ..utils.utils import ccorr

@register_model('CompGCN')
class CompGCN(BaseModel):
    """
    The models of the simplified CompGCN, without using basis vector, for a heterogeneous graph.

    Here, we present the implementation details for each task used for evaluation in the paper. 
    For all the tasks, we used COMPGCN build on PyTorch geometric framework (Fey & Lenssen, 2019).

    Link Prediction: For evaluation, 200-dimensional embeddings for node and relation embeddings are used. 
    For selecting the best model we perform a hyperparameter search using the validation data over the values listed in Table 8. 
    For training link prediction models, we use the standard binary cross entropy loss with label smoothing Dettmers et al. (2018).

    Node Classification: Following Schlichtkrull et al. (2017), we use 10% training data as validation for selecting the best model for both the datasets. 
    We restrict the number of hidden units to 32. We use cross-entropy loss for training our model.

    For all the experiments, training is done using Adam optimizer (Kingma & Ba, 2014) and Xavier initialization (Glorot & Bengio, 2010) is used for initializing parameters.
    """

    @classmethod
    def build_model_from_args(cls, args, hg):
        return cls(args.hidden_dim, args.hidden_dim, args.out_dim,
                   hg.etypes,
                   get_nodes_dict(hg), len(hg.etypes),
                   args.num_layers, comp_fn=args.comp_fn, dropout=args.dropout
                   )

    def __init__(self, in_dim, hid_dim, out_dim, etypes, n_nodes, n_rels, num_layers=2, comp_fn='sub', dropout=0.0,
                 activation=F.relu, batchnorm=True):
        super(CompGCN, self).__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.rel_names = list(set(etypes))
        self.rel_names.sort()
        self.n_rels = n_rels
        self.n_nodes = n_nodes
        self.num_layer = num_layers
        self.comp_fn = comp_fn
        self.dropout = dropout
        self.activation = activation
        self.batchnorm = batchnorm

        self.layers = nn.ModuleList()

        self.r_embedding = nn.Parameter(th.FloatTensor(self.n_rels + 1, self.in_dim))

        self.layers.append(CompGraphConvLayer(self.in_dim,
                                              self.hid_dim,
                                              self.rel_names,
                                              comp_fn=self.comp_fn,
                                              activation=self.activation,
                                              batchnorm=self.batchnorm,
                                              dropout=self.dropout))

        # Hidden layers with n - 1 CompGraphConv layers
        for i in range(self.num_layer - 2):
            self.layers.append(CompGraphConvLayer(self.hid_dim,
                                                  self.hid_dim,
                                                  self.rel_names,
                                                  comp_fn=self.comp_fn,
                                                  activation=self.activation,
                                                  batchnorm=self.batchnorm,
                                                  dropout=self.dropout))

        # Output layer with the output class
        self.layers.append(CompGraphConvLayer(self.hid_dim,
                                              self.out_dim,
                                              self.rel_names,
                                              comp_fn=self.comp_fn))
        nn.init.xavier_uniform_(self.r_embedding)

    def forward(self, hg, n_feats):

        # For full graph training, directly use the graph
        # Forward of n layers of CompGraphConv
        r_feats = self.r_embedding
        if hasattr(hg, 'ntypes'):
            # full graph training
            for layer in self.layers:
                n_feats, r_feats = layer(hg, n_feats, r_feats)
        else:
            # mini-batch training
            for layer, block in zip(self.layers, hg):
                n_feats, r_feats = layer(block, n_feats, r_feats)

        return n_feats

    def preprocess(self, hg):
        edata_in_out_mask(hg)


class CompGraphConvLayer(nn.Module):
    """One layer of simplified CompGCN."""

    def __init__(self,
                 in_dim,
                 out_dim,
                 rel_names,
                 comp_fn='sub',
                 activation=None,
                 batchnorm=False,
                 dropout=0):
        super(CompGraphConvLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.comp_fn = comp_fn
        self.actvation = activation
        self.batchnorm = batchnorm
        self.rel_names = rel_names
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
        self.conv = dglnn.HeteroGraphConv({
            rel: CompConv(comp_fn=comp_fn,
                          norm='right',
                          _allow_zero_in_degree=True)
            for rel in rel_names
        })

    def forward(self, hg, n_in_feats, r_feats):
        """
        Compute one layer of composition transfer for one relation only in a
        homogeneous graph with bidirectional edges.
        """
        with hg.local_scope():
            # Assign values to source nodes. In a homogeneous graph, this is equal to
            # assigning them to all nodes.

            # Assign feature to all edges with the same value, the r_feats.

            wdict = {}
            for i, etype in enumerate(self.rel_names):
                if etype[:4] == 'rev-' or etype[-4:] == '-rev':
                    W = self.W_I
                else:
                    W = self.W_O
                wdict[etype] = {'Linear': W, 'h_e': r_feats[i + 1]}

            if hg.is_block:
                inputs_src = n_in_feats
                inputs_dst = {k: v[:hg.number_of_dst_nodes(k)] for k, v in n_in_feats.items()}
            else:
                inputs_src = inputs_dst = n_in_feats
            outputs = self.conv(hg, inputs_src, mod_kwargs=wdict)

            for n, emd in outputs.items():
                # Step 4: add results of self-loop
                if self.comp_fn == 'sub':
                    h_self = self.W_S(inputs_dst[n] - r_feats[-1])
                elif self.comp_fn == 'mul':
                    h_self = self.W_S(inputs_dst[n] * r_feats[-1])
                elif self.comp_fn == 'ccorr':
                    h_self = self.W_S(ccorr(inputs_dst[n], r_feats[-1]))
                else:
                    raise Exception('Only supports sub, mul, and ccorr')
                h_self.add_(emd)
                # Compute relation output

                # Use batch norm
                if self.batchnorm:
                    if h_self.shape[0] > 1:
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
