import numpy as np
import pandas as pd
import scipy
import dgl
from dgl import function as fn
from dgl.nn.functional import edge_softmax
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dgl.utils import expand_as_pair
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC
import pickle
import joblib
import time
import warnings
from operator import itemgetter
import argparse
# back_log4

# the metapath instances datasets can be generated
# using dgl heterogeneous graph
# Todo: implement the methods of generating metapath instances datasets
# Todo: split the graph and features: ok lack of experiment
# Todo: adjust params list in all models
# Todo: label mask
# from the codes of original author we find that he used xavier initialization on relu instead of kaiming which I think
# of would be better
# Todo: deal with absolute address
# Todo: transfer data on cpu to gpu, possibly transfer data with dgl graph to gpu makes it too slow
# Todo: try split data and graph firstly then transfer data only to gpu.
# Todo：severely over fitting, considering adding dropout?




class MAGNN(nn.Module):
    def __init__(self, in_feats, h_feats, inter_attn_feats, num_heads, num_classes, num_layers,
                 metapath_list, edge_type_list, dropout_rate, activation=F.elu):
        '''
        :param in_feats: dict,

        in_feats['M']: the input dimension of movies
        in_feats['D']: the input dimension of directors
        in_feats['A']: the input dimension of actos

        :param h_feats: hidden dimension
        :param num_classes: output classes (output dimension)
        :param num_layers: number of hidden layers

        There're only 2 times when dimension of nodes' feature change. The first one is after input projection which
        decides the dimension of embedding, and the second one is after output projection based on downstream works.
        '''
        super(MAGNN, self).__init__()

        self.in_feats = in_feats
        self.h_feats = h_feats
        self.inter_attn_feats = inter_attn_feats
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.metapath_list = metapath_list
        self.edge_type_list = edge_type_list
        self.activation = activation

        # input projection
        self.ntypes = in_feats.keys()
        self.input_layers = [nn.Linear(in_features=in_feats[node], out_features=h_feats * num_heads) for node in
                       in_feats.keys()]
        self.input_projection = dict(zip(self.ntypes, self.input_layers))
        for layer in self.input_projection.values():
            nn.init.xavier_normal_(layer.weight, gain=1.414)

        # dropout
        self.feat_drop = nn.Dropout(p=dropout_rate)

        # hidden layers
        self.layers = nn.ModuleList()
        for i in range(num_layers - 1):
            self.layers.append(
                MAGNN_layer(in_feats=h_feats, inter_attn_feats=inter_attn_feats, out_feats=h_feats, num_heads=num_heads,
                            metapath_list=metapath_list, ntypes=self.ntypes,
                            edge_type_list=edge_type_list, last_layer=False))

        # output layer
        self.layers.append(
            MAGNN_layer(in_feats=h_feats, inter_attn_feats=inter_attn_feats, out_feats=num_classes,
                        num_heads=num_heads, metapath_list=metapath_list, ntypes=self.ntypes,
                        edge_type_list=edge_type_list, last_layer=True))

    # todo: using feat outside of graph
    def forward(self, g, feat_dict):
        '''

        :param g: the heterogeneous dgl graph
        :param feat_dict: feat_dict['M'] is the feature matrix of movies, so are the feat_dict['A'/'D']
        :return: the first is output logits of all node types, the second is embedding of all node types.
        the first one is the result after linear transformation of the second one
        '''
        with g.local_scope():
            # input projection
            for node in self.input_projection.keys():
                # g.nodes[node].data['feat'] = self.feat_drop(self.input_projection[node](g.nodes[node].data['feat']))
                g.nodes[node].data['feat'] = self.feat_drop(self.input_projection[node](feat_dict[node]))


            # hidden layer
            for i in range(self.num_layers - 1):
                h, _ = self.layers[i](g)
                for key in h.keys():
                    h[key] = self.activation(h[key])
                g.ndata['feat'] = h

            # output layer
            h_output, embedding = self.layers[-1](g)

           # return h_output, embedding
            return embedding
            # return h_output['M'], embedding


class MAGNN_attn_intra(nn.Module):
    def __init__(self, in_feats, out_feats, num_heads, feat_drop=0.5, attn_drop=0.5, negative_slope=0.01,
                 activation=F.elu):
        '''

        :param in_feats: (feature dimension of metapath instances, feature dimension of dst nodes)
        :param out_feats: feature dimension after attention mechanism
        :param num_heads: number of heads of attention
        :param feat_drop: feature dropout rate
        :param attn_drop: attention dropout rate
        :param negative_slope: negative slope rate of LeakyReLu
        :param activation: activation function in attention formula
        '''
        super(MAGNN_attn_intra, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats

        self.attn_r = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        # nn.init.xavier_normal_(self.attn_l, gain=1.414)
        nn.init.xavier_normal_(self.attn_r, gain=1.414)

    def forward(self, g, feat, metapath, movies_):
        '''

        :param g: the heterogeneous dgl graph
        :param feat: feat[0] is the the feature matrix of metapath instances,
                     feat[1] is the feature matrix of dst nodes,
                     if inter metapath aggregation, then only feat = feat[0]
        :return: the feature matrix of dgl graph nodes after attention
        '''
        with g.local_scope():
            # todo: need this feat drop?
            h_meta = self.feat_drop(feat[0]).view(-1, self._num_heads, self._out_feats) # feature matrix of metapath instances

            # metapath(right) part of attention
            er = (h_meta * self.attn_r).sum(dim=-1).unsqueeze(-1)

            graph_data = {
                ('meta_inst', 'meta2{}'.format(metapath[0]), metapath[0]): (th.arange(0, movies_.shape[0]),
                                                          th.tensor(movies_.iloc[:, 0].values))
            }


            g_meta = dgl.heterograph(graph_data)

            # feature vector of metapath instances
            g_meta.nodes['meta_inst'].data.update({'feat_src':h_meta, 'er':er})

            # compute attention without concat with hv
            g_meta.apply_edges(func=fn.copy_u('er', 'e'), etype='meta2{}'.format(metapath[0]))

            e = self.leaky_relu(g_meta.edata.pop('e'))
            g_meta.edata['a'] = self.attn_drop(edge_softmax(g_meta, e))

            # message passing, there's only one edge type
            g_meta.update_all(message_func=fn.u_mul_e('feat_src', 'a', 'm'), reduce_func=fn.sum('m', 'feat'))

            feat = self.activation(g_meta.dstdata['feat'])

            # return dst nodes' features after attention
            return feat.flatten(1)


class MAGNN_layer(nn.Module):
    def __init__(self, in_feats, inter_attn_feats, out_feats, num_heads, metapath_list,
                 ntypes, edge_type_list, last_layer=False):
        '''

        :param last_layer:
        :param num_heads: number of heads in attention
        :param in_feats: feature dimension of nodes
        :param out_feats: feature dimension of nodes after embedding
        :param inter_attn_feats: vector dimension of attention vector in inter metapath aggregation.
        :param metapath_list: metapath list, e.g ['MAM', 'MDM', ...]
        :param last_layer: True if this MAGNN layer is the last layer in the model
        '''
        super(MAGNN_layer, self).__init__()
        self.in_feats = in_feats
        self.inter_attn_feats = inter_attn_feats
        self.out_feats = out_feats
        self.num_heads = num_heads
        self.metapath_list = metapath_list # ['MDM', 'MAM', ...]
        self.ntypes = ntypes # ['M', 'D', 'A']
        self.edge_type_list = edge_type_list # ['M-A', 'M-D'], the MA is the same as AM in bidirectional graph
        self.last_layer = last_layer

        # in_feats_dst_meta = (feature dimension of dst nodes,
        #                      feature dimension of metapath instances after encoding)
        in_feats_dst_meta = tuple((in_feats, in_feats))

        self.intra_attn_layers = nn.ModuleDict()
        for metapath in self.metapath_list:
            self.intra_attn_layers[metapath] = \
                MAGNN_attn_intra(in_feats=in_feats_dst_meta, out_feats=in_feats, num_heads=num_heads)

        # The linear transformation at the beginning of inter metapath aggregation, including all metapath
        # The attention mechanism in inter metapath aggregation
        self.inter_linear = nn.ModuleDict()
        self.inter_attn_vec = nn.ModuleDict()
        for ntype in ntypes:
            self.inter_linear[ntype] = \
                nn.Linear(in_features=in_feats * num_heads, out_features=inter_attn_feats, bias=True)
            self.inter_attn_vec[ntype] = nn.Linear(in_features=inter_attn_feats, out_features=1, bias=False)
            nn.init.xavier_normal_(self.inter_linear[ntype].weight, gain=1.414)
            nn.init.xavier_normal_(self.inter_attn_vec[ntype].weight, gain=1.414)


        # r_vec: [r1, r1_inverse, r2, r2_inverse, ...., rn, rn_inverse]
        r_vec_ = nn.Parameter(th.empty(size=(len(edge_type_list) // 2, in_feats * num_heads // 2, 2)))
        nn.init.xavier_normal_(r_vec_.data, gain=1.414)
        self.r_vec = F.normalize(r_vec_, p=2, dim=2)
        self.r_vec = th.stack([self.r_vec, self.r_vec], dim=1)
        self.r_vec[:, 1, :, 1] = -self.r_vec[:, 1, :, 1]
        self.r_vec = self.r_vec.reshape(r_vec_.shape[0] * 2, r_vec_.shape[1], 2)
        self.r_vec_dict = nn.ParameterDict()
        for i, edge_type in zip(range(len(edge_type_list)), edge_type_list):
            self.r_vec_dict[edge_type] = nn.Parameter(self.r_vec[i])

        # output layer
        if last_layer:
            self._output_projection = nn.Linear(in_features=num_heads * in_feats, out_features=out_feats)
        else:
            self._output_projection = nn.Linear(in_features=num_heads * in_feats, out_features=num_heads * out_feats)
        nn.init.xavier_normal_(self._output_projection.weight, gain=1.414)

    def forward(self, g):
        with g.local_scope():
            # Todo: construct metapath from dgl graph

            # Intra-metapath latent transformation
            for _metapath in self.metapath_list:
                self.intra_metapath_trans(g, metapath=_metapath)

            # Inter-metapath latent transformation
            self.inter_metapath_trans(g, metapath_list=self.metapath_list)

            # output projection
            self.output_projection(g)

            # return final features after output projection (without nonlinear activation) and embedding
            # nonlinear activation will be added in MAGNN
            return g.ndata['output'], g.ndata['feat_']

    def intra_metapath_trans(self, g, metapath):
        # Construct content of movie_ from dgl graph, note that the order of columns of movie_ should be the same
        # as metapath, e.g MDM <--> ['movie_idx_x', 'director_idx', 'movie_idx_y']
        # because the metapaths are all symmetric, we only need to treat
        # metapaths as one directional and calculate the features
        # we treat the direction as from the right to the left, that is
        # the left node is v (dstnode), the right node is u (srcnode)

        data_dir = './data/imdb_datasets/movies_'

        with open(data_dir + '{}'.format(metapath) + '.pkl', 'rb') as file:
            movies_ = pickle.load(file)

        col_dict = {'M': 'movie_idx_x', 'D': 'director_idx_x', 'A': 'actor_idx_x'}
        movies_.sort_values(by=col_dict[metapath[0]], inplace=True)

        # encoder metapath instances
        # intra_metapath_feat: feature matrix of every metapath instance of param metapath
        intra_metapath_feat = self.encoder(g, metapath, movies_)

        # aggregate metapath instances into metapath using ATTENTION
        feat_attn = \
            self.intra_attn_layers[metapath](g, [intra_metapath_feat, g.nodes[metapath[0]].data['feat']], metapath, movies_)

        g.nodes[metapath[0]].data['{}'.format(metapath) + '_feat_attn'] = feat_attn

    def inter_metapath_trans(self, g, metapath_list):
        meta_s = {}

        # construct spi, where pi = ['MAM', 'MDM', ...]
        for metapath in metapath_list:
            meta_feat = g.nodes[metapath[0]].data['{}_feat_attn'.format(metapath)]
            meta_feat = F.tanh(self.inter_linear[metapath[0]](meta_feat)).mean(dim=0) # s_pi
            meta_s[metapath] = self.inter_attn_vec[metapath[0]](meta_feat) # e_pi

        for ntype in g.ntypes:
            # extract the metapath with the dst node type of ntype to construct a tensor
            # in order to compute softmax
            # metapaths: e.g if ntype is M, then ['MAM', 'MDM']
            metapaths = np.array(metapath_list)[[meta[0] == ntype for meta in metapath_list]]
            # extract the e_pi of metapaths
            # e.g the e_pi of ['MAM', 'MDM'] if ntype is M
            meta_b = th.tensor(itemgetter(*metapaths)(meta_s))
            # compute softmax, obtain b_pi, which is attention score of metapaths
            # e.g the b_pi of ['MAM', 'MDM'] if ntype is M
            meta_b = F.softmax(meta_b)
            # extract corresbonding features of metapath
            # e.g ['MDM_feat_attn', 'MAM_feat_attn'] if ntype is M
            meta_feat = itemgetter(*np.char.add(metapaths, '_feat_attn'))(g.nodes[ntype].data)
            # compute the embedding feature of nodes
            g.nodes[ntype].data['feat_'] = \
                th.stack([meta_b[i] * meta_feat[i] for i in range(len(meta_b))], dim=0).sum(dim=0)

    def encoder(self, g, metapath, movies_):

        feat = th.zeros((len(metapath), movies_.shape[0], g.nodes[metapath[0]].data['feat'].shape[1]))
        for i, ntype, col in zip(range(len(metapath)), metapath, movies_.columns):
            feat[i] = g.nodes[ntype].data['feat'][movies_[col].values]
        feat = feat.reshape(feat.shape[0], feat.shape[1], feat.shape[2] // 2, 2)

        temp_r_vec = th.zeros((len(metapath),  feat.shape[-2], 2))
        temp_r_vec[0, :, 0] = 1

        for i in range(1, len(metapath), 1):
            edge_type = '{}-{}'.format(metapath[i-1], metapath[i])
            temp_r_vec[i] = self.complex_hada(temp_r_vec[i-1], self.r_vec_dict[edge_type])
            feat[i] = self.complex_hada(feat[i], temp_r_vec[i], opt = 'feat')

        feat = feat.reshape(feat.shape[0], feat.shape[1], -1)
        return th.mean(feat, dim=0)

    @staticmethod
    def complex_hada(h, v, opt = 'r_vec'):
        """
        Complex hadamard product

        if opt == 'r_vec', the operands of complex hada are r_vecs
        else the h is feature matrix
        """

        # todo: where use clone here? what time to use clone? clone() would affect the effect of result?
        # using clone can solve error:
        # one of the variables needed for gradient computation has been modified by an inplace operation
        # inplace: the variable modifies itself in the unchanged(original) address
        if opt == 'r_vec':
            h_h, l_h = h[:, 0].clone(), h[:, 1].clone()
        else:
            h_h, l_h = h[:, :, 0].clone(), h[:, :, 1].clone()
        h_v, l_v = v[:, 0].clone(), v[:, 1].clone()
        res = th.zeros_like(h)

        if opt == 'r_vec':
            res[:, 0] = h_h * h_v - l_h * l_v
            res[:, 1] = h_h * l_v + l_h * h_v
        else:
            res[:, :, 0] = h_h * h_v - l_h * l_v
            res[:, :, 1] = h_h * l_v + l_h * h_v
        return res

    def output_projection(self, g):
        for node in g.ntypes:
            g.nodes[node].data['output'] = self._output_projection(g.nodes[node].data['feat_'])

