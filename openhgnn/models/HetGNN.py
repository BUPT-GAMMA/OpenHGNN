"""
HetGNN[KDD2019]
Heterogeneous Graph Neural Network
Paper: https://dl.acm.org/doi/abs/10.1145/3292500.3330961
Code: https://github.com/chuxuzhang/KDD2019_HetGNN

The author of the paper only gives the academic dataset.

"""
import dgl
import torch as th
import torch.nn as nn
import dgl.function as fn
import torch.nn.functional as F
from . import BaseModel, register_model


@register_model('HetGNN')
class HetGNN(BaseModel):
    @classmethod
    def build_model_from_args(cls, args, hg):
        return cls(hg, args)

    def __init__(self, hg, args):
        super(HetGNN, self).__init__()
        self.Het_Aggrate = Het_Aggregate(hg.ntypes, args.dim)
        self.ntypes = hg.ntypes
        self.device = args.device

        self.loss_fn = HetGNN.compute_loss
        #self.pred = ScorePredictor()

    def forward(self, hg, h=None):
        if h is None:
            h = self.extract_feature(hg, self.ntypes)
        x = self.Het_Aggrate(hg, h)
        return x

    def evaluator(self):
        self.link_preddiction()
        self.node_classification()

    def get_embedding(self):
        input_features = self.model.extract_feature(self.hg, self.hg.ntypes)
        x = self.model(self.model.preprocess(self.hg, self.args).to(self.args.device), input_features)
        return x

    def link_preddiction(self):
        x = self.get_embedding()
        self.model.lp_evaluator(x[self.category].to('cpu').detach(), self.train_batch, self.test_batch)

    def node_classification(self):
        x = self.get_embedding()
        self.model.nc_evaluator(x[self.category].to('cpu').detach(), self.labels, self.train_idx, self.test_idx)

    @staticmethod
    def compute_loss(pos_score, neg_score):
        # an example hinge loss
        loss = []
        for i in pos_score:
            loss.append(F.logsigmoid(pos_score[i]))
            loss.append(F.logsigmoid(-neg_score[i]))
        loss = th.cat(loss)
        return -loss.mean()

    @staticmethod
    def extract_feature(g, ntypes):
        input_features = {}
        for n in ntypes:
            ndata = g.srcnodes[n].data
            data = {}
            data['dw_embedding'] = ndata['dw_embedding']
            data['abstract'] = ndata['abstract']
            if n == 'paper':
                data['title'] = ndata['title']
                data['venue'] = ndata['venue']
                data['author'] = ndata['author']
                data['reference'] = ndata['reference']
            input_features[n] = data

        return input_features

    @staticmethod
    def pred(edge_subgraph, x):
        with edge_subgraph.local_scope():
            edge_subgraph.ndata['x'] = x
            for etype in edge_subgraph.canonical_etypes:
                edge_subgraph.apply_edges(
                    dgl.function.u_dot_v('x', 'x', 'score'), etype=etype)
            return edge_subgraph.edata['score']


class ScorePredictor(nn.Module):
    def forward(self, edge_subgraph, x):
        with edge_subgraph.local_scope():
            edge_subgraph.ndata['x'] = x
            for etype in edge_subgraph.canonical_etypes:
                edge_subgraph.apply_edges(
                    dgl.function.u_dot_v('x', 'x', 'score'), etype=etype)
            return edge_subgraph.edata['score']

class Het_Aggregate(nn.Module):
    def __init__(self, ntypes, dim):
        super(Het_Aggregate, self).__init__()
        # ntypes means nodes type name
        self.ntypes =ntypes
        self.dim = dim

        self.content_rnn = encoder_het_content(ntypes, dim)
        self.neigh_rnn = aggregate_het_neigh(ntypes, dim)

        self.atten_w = nn.ModuleDict({})
        for n in self.ntypes:
            self.atten_w[n] = nn.Linear(in_features=dim * 2, out_features=1)

        self.softmax = nn.Softmax(dim=1)
        self.activation = nn.LeakyReLU()
        self.drop = nn.Dropout(p=0.5)
        self.bn = nn.BatchNorm1d(dim)
        self.embed_d = dim

    def forward(self, hg, h_dict):
        with hg.local_scope():
            content_h = self.content_rnn(hg, h_dict)
            neigh_h = self.neigh_rnn(hg, content_h)
            # the content feature of the dst nodes
            dst_h = {k: v[:hg.number_of_dst_nodes(k)] for k, v in content_h.items()}
            out_h = {}
            for n in self.ntypes:
                d_h = dst_h[n]
                batch_size = d_h.shape[0]
                concat_h = []
                concat_emd = []
                for i in range(len(neigh_h[n])):
                    concat_h.append(th.cat((d_h, neigh_h[n][i]), 1))
                    concat_emd.append(neigh_h[n][i])
                concat_h.append(th.cat((d_h, d_h), 1))
                concat_emd.append(d_h)

                concat_h = th.hstack(concat_h).view(batch_size * (len(self.ntypes) + 1), self.dim *2)
                atten_w = self.activation(self.atten_w[n](concat_h)).view(batch_size, len(self.ntypes) + 1)

                atten_w = self.softmax(atten_w).view(batch_size, 1, 4)

                # weighted combination
                concat_emd = th.hstack(concat_emd).view(batch_size, len(self.ntypes) + 1, self.dim)

                weight_agg_batch = th.bmm(atten_w, concat_emd).view(batch_size, self.dim)
                out_h[n] = weight_agg_batch
            return out_h



class aggregate_het_neigh(nn.Module):
    '''
    Aggregating Heterogeneous Neighbors(C3)
    Same Type Neighbors Aggregation
    '''
    def __init__(self, ntypes, dim):
        super(aggregate_het_neigh, self).__init__()
        self.neigh_rnn = nn.ModuleDict({})
        self.ntypes =ntypes
        for n in ntypes:
            self.neigh_rnn[n] = lstm_aggr(dim)

    def forward(self, hg, inputs):
        with hg.local_scope():
            outputs = {}
            for i in self.ntypes:
                outputs[i] = []
            if isinstance(inputs, tuple) or hg.is_block:
                if isinstance(inputs, tuple):
                    src_inputs, dst_inputs = inputs
                else:
                    src_inputs = inputs
                    dst_inputs = {k: v[:hg.number_of_dst_nodes(k)] for k, v in inputs.items()}

                for stype, etype, dtype in hg.canonical_etypes:
                    rel_graph = hg[stype, etype, dtype]
                    if rel_graph.number_of_edges() == 0:
                        continue
                    if stype not in src_inputs or dtype not in dst_inputs:
                        continue
                    dstdata = self.neigh_rnn[stype](
                        rel_graph,
                        (src_inputs[stype], dst_inputs[dtype]))
                    outputs[dtype].append(dstdata)
            else:
                for stype, etype, dtype in hg.canonical_etypes:
                    rel_graph = hg[stype, etype, dtype]
                    if rel_graph.number_of_edges() == 0:
                        continue
                    if stype not in inputs:
                        continue
                    dstdata = self.neigh_rnn[stype](
                        rel_graph,
                        inputs[stype])
                    outputs[dtype].append(dstdata)
            return outputs


class lstm_aggr(nn.Module):
    '''
    Aggregate the same neighbors with LSTM
    '''
    def __init__(self, dim):
        super(lstm_aggr, self).__init__()
        self.lstm = nn.LSTM(dim, int(dim / 2), 1, batch_first=True, bidirectional=True)

        self.lstm.flatten_parameters()
    def _lstm_reducer(self, nodes):
        m = nodes.mailbox['m']  # (B, L, D)
        batch_size = m.shape[0]
        all_state, last_state = self.lstm(m)
        return {'neigh': th.mean(all_state, 1)}

    def forward(self, g, inputs):
        with g.local_scope():
            if isinstance(inputs, tuple) or g.is_block:
                if isinstance(inputs, tuple):
                    src_inputs, dst_inputs = inputs
                else:
                    src_inputs = inputs
                    dst_inputs = {k: v[:g.number_of_dst_nodes(k)] for k, v in inputs.items()}
                g.srcdata['h'] = src_inputs
                g.update_all(fn.copy_u('h', 'm'), self._lstm_reducer)
                h_neigh = g.dstdata['neigh']
            else:
                g.srcdata['h'] = inputs
                g.update_all(fn.copy_u('h', 'm'), self._lstm_reducer)
                h_neigh = g.dstdata['neigh']
            return h_neigh


class encoder_het_content(nn.Module):
    '''
    The Encoding Heterogeneous Contents(C2) in the paper
    For each node type, encoder all content features with the LSTM
    '''
    def __init__(self, ntypes, dim):
        super(encoder_het_content, self).__init__()
        self.content_rnn = nn.ModuleDict({})
        for n in ntypes:
            self.content_rnn[n] = nn.LSTM(dim, int(dim / 2), 1, batch_first=True, bidirectional=True)
            self.content_rnn[n].flatten_parameters()
        self.ntypes = ntypes
        self.dim = dim

    def forward(self, hg, h_dict):
        # conten_h is output dict whose key is the name of node types.
        with hg.local_scope():
            content_h = {}
            for ntype in self.ntypes:
                h = h_dict[ntype]
                for i, n_h in enumerate(h):
                    if i == 0:
                        concate_embed = h[n_h]
                    else:
                        concate_embed = th.cat((concate_embed, h[n_h]), 1)
                concate_embed = concate_embed.view(concate_embed.shape[0], -1, self.dim)
                all_state, last_state = self.content_rnn[ntype](concate_embed)
                out_h = th.mean(all_state, 1).squeeze()
                content_h[ntype] = out_h
            return content_h



# from openhgnn.models.micro_layer.LSTM_conv import LSTMConv
# from openhgnn.models.HeteroGraphConv import HeteroGraphConv
# class HetGNNConv(nn.Module):
#     def __init__(self, graph, ntypes, dim):
#         super(HetGNNConv, self).__init__()
#         # ntypes means nodes type name
#         self.ntypes =ntypes
#         self.dim = dim
#
#         # hetero conv modules
#         self.micro_conv = HeteroGraphConv({
#             etype: LSTMConv(dim=dim)
#             for srctype, etype, dsttype in graph.canonical_etypes
#         })
#
#         # different types aggregation module
#         self.macro_conv = AttConv(in_feats=hidden_dim * n_heads, out_feats=hidden_dim,
#                                                              num_heads=n_heads,
#                                                              dropout=dropout, negative_slope=0.2)
#
#         self.atten_w = nn.ModuleDict({})
#         for n in self.ntypes:
#             self.atten_w[n] = nn.Linear(in_features=dim * 2, out_features=1)
#
#         self.softmax = nn.Softmax(dim=1)
#         self.activation = nn.LeakyReLU()
#         self.drop = nn.Dropout(p=0.5)
#         self.bn = nn.BatchNorm1d(dim)
#         self.embed_d = dim
#
#     def forward(self, hg, h):
#         x = self.Het_Aggrate(hg, h)
#         return x

