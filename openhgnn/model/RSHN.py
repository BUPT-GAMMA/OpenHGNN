# The implementation of ICDM 2019 paper "Relation Structure-Aware Heterogeneous Graph Neural Network" RSHN.
import dgl
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl import function as fn
from dgl.utils import expand_as_pair
from dgl.nn.functional import edge_softmax

class RSHN(nn.Module):
    def __init__(self, in_feats1, in_feats2, dim, num_classes, num_node_layer, num_edge_layer, dropout):
        super(RSHN, self).__init__()
        # map the node feature
        self.feats = nn.Embedding(in_feats1, dim)
        # map the edge feature
        self.linear_e = nn.Linear(in_features=in_feats2, out_features=dim, bias=False)
        self.cl_conv1 = AGNNConv()
        self.num_node_layer = num_node_layer
        self.num_edge_layer = num_edge_layer
        if num_edge_layer == 2:
            self.cl_conv2 = AGNNConv()

        self.nn_conv1 = GraphConv(in_feats=dim, out_feats=dim, activation=th.tanh)
        if num_node_layer == 2:
            self.nn_conv2 = GraphConv(in_feats=dim, out_feats=dim, activation=th.tanh)
        self.dropout = dropout

        self.emd2pred = nn.Linear(dim, num_classes)
        self.init_para()

    def init_para(self):
        nn.init.xavier_uniform_(self.linear_e.weight)
        nn.init.xavier_uniform_(self.feats.weight)
        nn.init.xavier_uniform_(self.emd2pred.weight)


    def forward(self, g, cl_graph):
        with cl_graph.local_scope():
            h = cl_graph.ndata['h']
            h_e = cl_graph.edata['w']
            h = th.relu(self.cl_conv1(cl_graph, h, h_e))
            h = F.dropout(h, p=self.dropout, training=False)
            if self.num_edge_layer == 2:
                h = th.relu(self.cl_conv2(cl_graph, h, h_e))
                h = F.dropout(h, p=self.dropout, training=False)

        edge_weight = F.embedding(g.edata[dgl.ETYPE].long(), h)
        with g.local_scope():
            h = self.feats(th.arange(g.number_of_nodes(), device=g.device))
            edge_weight = self.linear_e(edge_weight)
            x = self.nn_conv1(g, h, edge_weight=edge_weight)
            x = F.dropout(x, p=self.dropout, training=False)
            if self.num_node_layer == 2:
                x = self.nn_conv2(g, x, edge_weight=edge_weight)
                x = F.dropout(x, p=self.dropout, training=False)

            x = self.emd2pred(x)
            x = F.dropout(x, p=self.dropout, training=False)

            return x


class AGNNConv(nn.Module):
    def __init__(self,
                 init_beta=1,
                 eps=0.,
                 train_eps=True,
                 learn_beta=True):
        super(AGNNConv, self).__init__()
        if learn_beta:
            self.beta = nn.Parameter(th.Tensor(init_beta))
        else:
            self.register_buffer('beta', th.Tensor(init_beta))
        if train_eps:
            self.eps = th.nn.Parameter(th.Tensor([eps]))
        else:
            self.register_buffer('eps', th.Tensor([eps]))

    def forward(self, graph, feat, edge_weight):

        with graph.local_scope():
            feat_src, feat_dst = expand_as_pair(feat, graph)

            graph.srcdata['h'] = feat_src
            graph.srcdata['norm_h'] = F.normalize(feat_src, p=2, dim=-1)
            if isinstance(feat, tuple) or graph.is_block:
                graph.dstdata['norm_h'] = F.normalize(feat_dst, p=2, dim=-1)

            e = self.beta * edge_weight

            graph.edata['p'] = edge_softmax(graph, e, norm_by='src')
            graph.update_all(fn.u_mul_e('norm_h', 'p', 'm'), fn.sum('m', 'h'))
            rst = graph.dstdata.pop('h')
            rst = (1 + self.eps) * feat + rst
            return rst

class GraphConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 activation=None,
                 ):
        super(GraphConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats

        self.weight1 = nn.Parameter(th.Tensor(in_feats, out_feats))
        self.weight2 = nn.Parameter(th.Tensor(in_feats, out_feats))

        self.reset_parameters()

        self.activation = activation

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight1)
        nn.init.xavier_uniform_(self.weight2)


    def forward(self, graph, feat, edge_weight=None):

        with graph.local_scope():

            aggregate_fn = fn.copy_src('h', 'm')
            if edge_weight is not None:
                assert edge_weight.shape[0] == graph.number_of_edges()
                graph.edata['_edge_weight'] = edge_weight
                aggregate_fn = fn.u_sub_e('h', '_edge_weight', 'm')

            feat_src, feat_dst = expand_as_pair(feat, graph)

            # aggregate first then mult W
            graph.srcdata['h'] = feat_src
            graph.update_all(aggregate_fn, fn.sum(msg='m', out='h'))
            rst = graph.dstdata['h']
            rst = th.matmul(feat_src, self.weight1) + th.matmul(rst, self.weight2)


            if self.activation is not None:
                rst = self.activation(rst)

            return rst

