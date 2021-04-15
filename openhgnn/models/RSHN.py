# The implementation of ICDM 2019 paper "Relation Structure-Aware Heterogeneous Graph Neural Network" RSHN.
# @Time   : 2021/3/1
# @Author : Tianyu Zhao
# @Email  : tyzhao@bupt.edu.cn
import dgl
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl import function as fn
from dgl.utils import expand_as_pair
from dgl.nn.functional import edge_softmax
from . import HeteroEmbedLayer, BaseModel, register_model
from ..utils import get_nodes_dict, coarsened_line_graph


@register_model('RSHN')
class RSHN(BaseModel):
    '''
    Note that
    The four KG dataset in dgl(AIFB, MUTAG, BGS, AM) do not have nodes/edges feature.
    So we should initialize the feature use nn.Parameter, dimension is the hyper-parameter.
    '''

    @classmethod
    def build_model_from_args(cls, args, hg):
        rshn = cls(get_nodes_dict(hg), 10,
                   args.dim, args.num_classes, args.num_node_layer, args.num_edge_layer, dropout=args.dropout
            )
        cl = coarsened_line_graph(rw_len=args.rw_len, batch_size=args.batch_size, n_dataset=args.dataset,
                                  symmetric=True)
        cl_graph = cl.get_cl_graph(hg).to(args.device)
        cl_graph = cl.init_cl_graph(cl_graph)
        rshn.cl_graph = cl_graph

        linear_e1 = nn.Linear(in_features=cl_graph.num_nodes(), out_features=args.dim, bias=False)
        nn.init.xavier_uniform_(linear_e1.weight)
        rshn.linear_e1 = linear_e1
        return rshn

    def __init__(self, n_nodes, in_feats2, dim, num_classes, num_node_layer, num_edge_layer, dropout):
        super(RSHN, self).__init__()
        # map the node feature
        self.h_n_dict = HeteroEmbedLayer(n_nodes, dim)
        # map the edge feature
        self.num_node_layer = num_node_layer
        self.edge_layers = nn.ModuleList()
        for i in range(num_edge_layer):
            self.edge_layers.append(AGNNConv())

        self.node_layers = nn.ModuleList()
        for i in range(num_node_layer):
            self.node_layers.append(GraphConv(in_feats=dim, out_feats=dim, activation=th.tanh))

        self.dropout = dropout

        self.emd2pred = nn.Linear(dim, num_classes)
        self.init_para()

    def init_para(self):
        nn.init.xavier_uniform_(self.emd2pred.weight)

    def forward(self, hg, n_feats=None):

        # For full graph training, directly use the graph
        if n_feats is None:
            # full graph training
            n_feats = self.h_n_dict()

        # Forward of n layers of CompGraphConv

        for layer in self.edge_layers:
            h = self.cl_graph.ndata['h']
            h_e = self.cl_graph.edata['w']
            h = th.relu(layer(self.cl_graph, h, h_e))
            h = F.dropout(h, p=self.dropout, training=False)

        edge_weight = {}
        for i, e in enumerate(hg.canonical_etypes):
            edge_weight[e] = h[i].expand(hg.num_edges(e), -1)
        temp = self.h_n_dict()
        n_feats = {}
        for n in hg.ntypes:
            n_feats[n] = temp[n]
        if hasattr(hg, 'ntypes'):
            #edge_weight = F.embedding(hg.edata[dgl.ETYPE].long(), h)

            # full graph training
           # edge_weight = self.linear_e1(edge_weight)
            for layer in self.node_layers:
                n_feats = layer(hg, n_feats, edge_weight)
                n_feats = F.dropout(n_feats, p=self.dropout, training=False)
        else:
            # minibatch training
            pass
        n_feats = self.emd2pred(n_feats)
        n_feats = F.dropout(n_feats, p=self.dropout, training=False)
        return n_feats


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
                #assert edge_weight.shape[0] == graph.number_of_edges()
                graph.edata['_edge_weight'] = edge_weight
                aggregate_fn = fn.u_sub_e('h', '_edge_weight', 'm')

            feat_src, feat_dst = expand_as_pair(feat, graph)

            # aggregate first then mult W
            graph.srcdata['h'] = feat_src
            graph.apply_edges(lambda edges: {'h': edges.data['_edge_weight'] * 2})
            graph.update_all(aggregate_fn, fn.sum(msg='m', out='h'))
            rst = graph.dstdata['h']
            rst = th.matmul(feat_src, self.weight1) + th.matmul(rst, self.weight2)


            if self.activation is not None:
                rst = self.activation(rst)

            return rst

