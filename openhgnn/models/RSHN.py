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
from . import BaseModel, register_model
from ..sampler.RSHN_sampler import coarsened_line_graph


@register_model('RSHN')
class RSHN(BaseModel):
    r"""
    Relation structure-aware heterogeneous graph neural network (RSHN) builds coarsened line graph to obtain edge features first,
    then uses a novel Message Passing Neural Network (MPNN) to propagate node and edge features.

    We implement a API build a coarsened line graph.

    Attributes
    -----------
    edge_layers : AGNNConv
        Applied in Edge Layer.
    coarsened line graph : dgl.DGLGraph
        Propagate edge features.
    """

    @classmethod
    def build_model_from_args(cls, args, hg):
        rshn = cls(dim=args.hidden_dim,
                   out_dim=args.out_dim,
                   num_node_layer=args.num_node_layer,
                   num_edge_layer=args.num_edge_layer,
                   dropout=args.dropout
            )
        cl = coarsened_line_graph(rw_len=args.rw_len, batch_size=args.batch_size, n_dataset=args.dataset,
                                  symmetric=True)
        cl_graph = cl.get_cl_graph(hg).to(args.device)
        cl_graph = cl.init_cl_graph(cl_graph)
        rshn.cl_graph = cl_graph

        linear_e1 = nn.Linear(in_features=cl_graph.num_nodes(), out_features=args.hidden_dim, bias=False)
        nn.init.xavier_uniform_(linear_e1.weight)
        rshn.linear_e1 = linear_e1
        return rshn

    def __init__(self, dim, out_dim, num_node_layer, num_edge_layer, dropout):
        super(RSHN, self).__init__()
        # map the edge feature
        self.num_node_layer = num_node_layer
        self.edge_layers = nn.ModuleList()
        for i in range(num_edge_layer):
            self.edge_layers.append(AGNNConv())

        self.node_layers = nn.ModuleList()
        for i in range(num_node_layer):
            self.node_layers.append(GraphConv(in_feats=dim, out_feats=dim, dropout=dropout, activation=th.tanh))
        self.linear = nn.Linear(in_features=dim, out_features=out_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.init_para()

    def init_para(self):
        return

    def forward(self, hg, n_feats, *args, **kwargs):
        r"""
        First, apply edge_layer in cl_graph to get edge embedding.
        Then, propagate node and edge features through GraphConv.
        """
        # For full graph training, directly use the graph
        # Forward of n layers of CompGraphConv
        h = self.cl_graph.ndata['h']
        h_e = self.cl_graph.edata['w']
        for layer in self.edge_layers:
            h = th.relu(layer(self.cl_graph, h, h_e))
            h = self.dropout(h)

        h = self.linear_e1(h)
        edge_weight = {}
        for i, e in enumerate(hg.canonical_etypes):
            edge_weight[e] = h[i].expand(hg.num_edges(e), -1)
        if hasattr(hg, 'ntypes'):
            # edge_weight = F.embedding(hg.edata[dgl.ETYPE].long(), h)

            # full graph training
            for layer in self.node_layers:
                n_feats = layer(hg, n_feats, edge_weight)
        else:
            # minibatch training
            pass
        for n in n_feats:
            #n_feats[n] = self.dropout(self.linear(n_feats[n]))
            n_feats[n] = self.linear(n_feats[n])
        return n_feats


class AGNNConv(nn.Module):
    def __init__(self,
                 eps=0.,
                 train_eps=False,
                 learn_beta=True):
        super(AGNNConv, self).__init__()
        self.initial_eps = eps
        if learn_beta:
            self.beta = nn.Parameter(th.Tensor(1))
        else:
            self.register_buffer('beta', th.Tensor(1))
        self.learn_beta = learn_beta
        if train_eps:
            self.eps = th.nn.Parameter(th.ones([eps]))
        else:
            self.register_buffer('eps', th.Tensor([eps]))

        self.reset_parameters()

    def reset_parameters(self):
        self.eps.data.fill_(self.initial_eps)
        if self.learn_beta:
            self.beta.data.fill_(1)

    def forward(self, graph, feat, edge_weight):

        with graph.local_scope():
            feat_src, feat_dst = expand_as_pair(feat, graph)

            graph.srcdata['norm_h'] = F.normalize(feat_src, p=2, dim=-1)

            e = self.beta * edge_weight
            #graph.edata['p'] = e
            graph.edata['p'] = edge_softmax(graph, e, norm_by='src')
            graph.update_all(fn.u_mul_e('norm_h', 'p', 'm'), fn.sum('m', 'h'))
            rst = graph.dstdata.pop('h')
            rst = (1 + self.eps) * feat + rst
            return rst


class GraphConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats, dropout,
                 activation=None,
                 ):
        super(GraphConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats

        self.weight1 = nn.Parameter(th.Tensor(in_feats, out_feats))
        #self.weight2 = nn.Parameter(th.Tensor(in_feats, out_feats))

        self.reset_parameters()
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight1)
        #nn.init.xavier_uniform_(self.weight2)

    def forward(self, hg, feat, edge_weight=None):

        with hg.local_scope():
            outputs = {}
            norm = {}
            aggregate_fn = fn.copy_u('h', 'm')
            if edge_weight is not None:
                #assert edge_weight.shape[0] == graph.number_of_edges()
                hg.edata['_edge_weight'] = edge_weight
                aggregate_fn = fn.u_mul_e('h', '_edge_weight', 'm')
            for e in hg.canonical_etypes:
                if e[0] == e[1]:
                    hg = dgl.remove_self_loop(hg, etype=e)
            feat_src, feat_dst = expand_as_pair(feat, hg)

            # aggregate first then mult W
            hg.srcdata['h'] = feat_src
            for e in hg.canonical_etypes:
                stype, etype, dtype = e
                sub_graph = hg[stype, etype, dtype]
                sub_graph.update_all(aggregate_fn, fn.sum(msg='m', out='out'))
                temp = hg.ndata['out'].pop(dtype)
                degs = sub_graph.in_degrees().float().clamp(min=1)
                if isinstance(temp, dict):
                    temp = temp[dtype]
                if outputs.get(dtype) is None:
                    outputs[dtype] = temp
                    norm[dtype] = degs
                else:
                    outputs[dtype].add_(temp)
                    norm[dtype].add_(degs)

            def _apply(ntype, h, norm):
                h = th.matmul(h+feat[ntype], self.weight1)

                if self.activation:
                    h = self.activation(h)
                return self.dropout(h)

            return {ntype: _apply(ntype, h, norm) for ntype, h in outputs.items()}



