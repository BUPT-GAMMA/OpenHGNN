import os

import math
import numpy as np
import torch
import torch.nn as nn
import dgl.function as fn
import torch.nn.functional as F
from . import BaseModel, register_model
import torch.nn.functional as F
from torch.nn import Identity
from dgl import mean_nodes
import abc

@register_model('ComPILE')
class ComPILE(BaseModel):
    @classmethod
    def build_model_from_args(cls, args, relation2id):
        return cls(args,relation2id)

    def __init__(self, args, relation2id):
        super(ComPILE, self).__init__()
        super().__init__()
        self.relation2id = relation2id
        self.max_label_value = args.max_label_value
        self.params = args
        self.latent_dim = self.params.emb_dim
        self.output_dim = 1
        self.node_emb = self.params.inp_dim
        self.relation_emb = self.params.rel_emb_dim
        self.edge_emb = self.node_emb * 2 + self.relation_emb
        self.hidden_size = self.params.emb_dim
        self.num_relation = self.params.num_rels

        self.final_relation_embeddings = nn.Parameter(torch.randn(self.params.num_rels, self.params.rel_emb_dim))
        self.relation_to_edge = nn.Linear(self.params.rel_emb_dim, self.hidden_size)

        #  self.linear1 = nn.Linear(self.params.emb_dim + self.relation_emb + 2*self.params.emb_dim, 16)
        self.linear1 = nn.Linear(self.params.emb_dim, 16)
        self.linear2 = nn.Linear(16, 1)

        self.node_fdim = self.node_emb
        self.edge_fdim = self.edge_emb

        self.bias = False
        self.depth = 3
        self.dropout = 0.5
        self.layers_per_message = 1
        self.undirected = False
        self.node_messages = False
        self.args = args
        # Dropout
        self.dropout_layer = nn.Dropout(p=self.dropout)

        # Activation
        self.act_func = get_activation_function('ReLU')

        # Cached zeros
        self.cached_zero_vector = nn.Parameter(torch.zeros(self.hidden_size), requires_grad=False)
        # Input
        input_dim = self.node_fdim
        self.W_i_node = nn.Linear(input_dim, self.hidden_size, bias=self.bias)
        input_dim = self.edge_fdim
        self.W_i_edge = nn.Linear(input_dim, self.hidden_size, bias=self.bias)

        w_h_input_size_node = self.hidden_size + self.edge_fdim
        self.W_h_node = nn.Linear(w_h_input_size_node, self.hidden_size, bias=self.bias)

        self.input_attention1 = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=self.bias)
        self.input_attention2 = nn.Linear(self.hidden_size, 1, bias=self.bias)

        w_h_input_size_edge = self.hidden_size
        for depth in range(self.depth - 1):
            self._modules['W_h_edge_{}'.format(depth)] = nn.Linear(w_h_input_size_edge, self.hidden_size,
                                                                   bias=self.bias)
            #  self._modules['W_h_edge_{}'.format(depth)] = nn.Linear(w_h_input_size_edge * 3 + self.params.rel_emb_dim, self.hidden_size, bias=self.bias)
            self._modules['Attention1_{}'.format(depth)] = nn.Linear(self.hidden_size + self.relation_emb,
                                                                     self.hidden_size, bias=self.bias)
            self._modules['Attention2_{}'.format(depth)] = nn.Linear(self.hidden_size, 1, bias=self.bias)

        self.W_o = nn.Linear(self.hidden_size * 2, self.hidden_size)

        self.gru = BatchGRU(self.hidden_size)

        self.communicate_mlp = nn.Linear(self.hidden_size * 3, self.hidden_size, bias=self.bias)

        for depth in range(self.depth - 1):
            self._modules['W_h_node_{}'.format(depth)] = nn.Linear(self.hidden_size, self.hidden_size, bias=self.bias)

    def forward(self, subgraph):

        target_relation = []
        for i in range(len(subgraph)):
            graph = subgraph[i]
            target = graph.edata['label'][-1].squeeze()
            target_relation.append(self.final_relation_embeddings[target, :].unsqueeze(0))
        target_relation = torch.cat(target_relation, dim=0)
        graph_embed, source_embed, target_embed = self.batch_subgraph(subgraph)
        #  print(graph_embed.shape, source_embed.shape, target_embed.shape, target_relation.shape)
        #  conv_input = torch.cat((source_embed, target_relation, target_embed, graph_embed), dim=1)
        # conv_input =  torch.cat([graph_embed, source_embed + target_relation -target_embed], dim=-1)

        # conv_input = (graph_embed) + torch.tanh(source_embed + target_relation -target_embed)

        conv_input = torch.tanh(source_embed + target_relation - target_embed)
        out_conv = (self.linear1(conv_input))
        out_conv = self.linear2(out_conv)
        return out_conv

    def batch_subgraph(self, subgraph):

        graph_sizes = [];
        node_feat = []
        list_num_nodes = np.zeros((len(subgraph),), dtype=np.int32)
        list_num_edges = np.zeros((len(subgraph),), dtype=np.int32)
        node_count = 0;
        edge_count = 0;
        edge_feat = []
        total_edge = [];
        source_node = [];
        target_node = []
        total_target_relation = [];
        total_edge2 = []
        total_source = [];
        total_target = []
        for i in range(len(subgraph)):
            graph = subgraph[i]
            node_embedding = graph.ndata['feat']
            node_feat.append(node_embedding)

            graph_sizes.append(graph.number_of_nodes())
            list_num_nodes[i] = graph.number_of_nodes()
            list_num_edges[i] = graph.number_of_edges()

            nodes = list((graph.nodes()).data.numpy())
            source = list((graph.edges()[0]).data.numpy())
            target = list((graph.edges()[1]).data.numpy())
            relation = graph.edata['type']
            relation_now = self.final_relation_embeddings[relation, :]

            target_relation = graph.edata['label']
            target_relation_now = self.final_relation_embeddings[target_relation, :]
            total_target_relation.append(target_relation_now)

            mapping = dict(zip(nodes, [i for i in range(node_count, node_count + list_num_nodes[i])]))

            source_map_now = np.array([mapping[v] for v in source]) - node_count
            target_map_now = np.array([mapping[v] for v in target]) - node_count
            source_embed = node_embedding[source_map_now, :]
            target_embed = node_embedding[target_map_now, :]
            source_embed = source_embed.to(device=self.final_relation_embeddings.device)
            target_embed = target_embed.to(device=self.final_relation_embeddings.device)

            edge_embed = torch.cat([source_embed, relation_now, target_embed], dim=1)
            #  edge_embed = source_embed + relation_now - target_embed
            edge_feat.append(edge_embed)

            source_now = (graph.ndata['id'] == 1).nonzero().squeeze() + node_count
            target_now = (graph.ndata['id'] == 2).nonzero().squeeze() + node_count
            source_node.append(source_now)
            target_node.append(target_now)

            target_now = target_now.unsqueeze(0).repeat(list_num_edges[i], 1).long()
            source_now = source_now.unsqueeze(0).repeat(list_num_edges[i], 1).long()
            total_source.append(source_now);
            total_target.append(target_now)

            node_count += list_num_nodes[i]

            source_map = torch.LongTensor(np.array([mapping[v] for v in source])).unsqueeze(0)
            target_map = torch.LongTensor(np.array([mapping[v] for v in target])).unsqueeze(0)

            edge_pair = torch.cat([target_map, torch.LongTensor(
                np.array(range(edge_count, edge_count + list_num_edges[i]))).unsqueeze(0)], dim=0)

            edge_pair2 = torch.cat([source_map, torch.LongTensor(
                np.array(range(edge_count, edge_count + list_num_edges[i]))).unsqueeze(0)], dim=0)

            edge_count += list_num_edges[i]
            total_edge.append(edge_pair)
            total_edge2.append(edge_pair2)

        source_node = np.array(source_node);
        target_node = np.array(target_node)

        total_edge = torch.cat(total_edge, dim=1)
        total_edge2 = torch.cat(total_edge2, dim=1)
        total_target_relation = torch.cat(total_target_relation, dim=0)
        total_source = torch.cat(total_source, dim=0)
        total_target = torch.cat(total_target, dim=0)

        total_num_nodes = np.sum(list_num_nodes)
        total_num_edges = np.sum(list_num_edges)

        e2n_value = torch.FloatTensor(torch.ones(total_edge.shape[1]))
        e2n_sp = torch.sparse.FloatTensor(total_edge, e2n_value, torch.Size([total_num_nodes, total_num_edges]))
        e2n_sp2 = torch.sparse.FloatTensor(total_edge2, e2n_value, torch.Size([total_num_nodes, total_num_edges]))
        # e2n_sp = F.normalize(e2n_sp, dim=2, p=1)

        node_feat = torch.cat(node_feat, dim=0)
        e2n_sp = e2n_sp.to(device=self.final_relation_embeddings.device)
        e2n_sp2 = e2n_sp2.to(device=self.final_relation_embeddings.device)
        node_feat = node_feat.to(device=self.final_relation_embeddings.device)

        edge_feat = torch.cat(edge_feat, dim=0)
        graph_embed, source_embed, target_embed = self.gnn(node_feat, edge_feat, e2n_sp, e2n_sp2, graph_sizes,
                                                           total_target_relation, total_source, total_target,
                                                           source_node, target_node, list(list_num_edges))

        return graph_embed, source_embed, target_embed

    def gnn(self, node_feat, edge_feat, e2n_sp, e2n_sp2, graph_sizes, target_relation, total_source, total_target,
            source_node, target_node, edge_sizes=None, node_degs=None):

        input_node = self.W_i_node(node_feat)  # num_nodes x hidden_size
        input_node = self.act_func(input_node)
        message_node = input_node.clone()
        relation_embed = (edge_feat[:, self.node_emb: self.node_emb + self.relation_emb])

        input_edge = self.W_i_edge(edge_feat)  # num_edges x hidden_size
        message_edge = self.act_func(input_edge)
        input_edge = self.act_func(input_edge)

        graph_source_embed = message_node[total_source, :].squeeze(1)
        graph_target_embed = message_node[total_target, :].squeeze(1)
        graph_edge_embed = graph_source_embed + target_relation - graph_target_embed
        edge_target_message = gnn_spmm(e2n_sp.t(), message_node)
        edge_source_message = gnn_spmm(e2n_sp2.t(), message_node)
        edge_message = edge_source_message + relation_embed - edge_target_message
        #  print(total_source.shape, total_target.shape, graph_source_embed.shape)
        attention = torch.cat([graph_edge_embed, edge_message], dim=1)
        attention = torch.relu(self.input_attention1(attention))
        attention = torch.sigmoid(self.input_attention2(attention))

        # Message passing
        for depth in range(self.depth - 1):
            #   agg_message = index_select_ND(message_edge, a2b)
            #   agg_message = agg_message.sum(dim=1) * agg_message.max(dim=1)[0]
            #  agg_message = gnn_spmm(e2n_sp, message_edge)/e2n_sp.sum(1, keepdim=True)
            message_edge = (message_edge * attention)
            agg_message = gnn_spmm(e2n_sp, message_edge)
            message_node = message_node + agg_message
            message_node = self.act_func(self._modules['W_h_node_{}'.format(depth)](message_node))

            # directed graph
            #  rev_message = message_edge[b2revb]  # num_edges x hidden
            #   message_edge = message_node[b2a] - rev_message  # num_edges x hidden
            edge_target_message = gnn_spmm(e2n_sp.t(), message_node)
            edge_source_message = gnn_spmm(e2n_sp2.t(), message_node)
            # message_edge = torch.cat([message_edge, edge_source_message, relation_embed, edge_target_message], dim=-1)
            message_edge = torch.relu(
                message_edge + torch.tanh(edge_source_message + relation_embed - edge_target_message))
            message_edge = self._modules['W_h_edge_{}'.format(depth)](message_edge)
            message_edge = self.act_func(input_edge + message_edge)
            message_edge = self.dropout_layer(message_edge)  # num_edges x hidden

            graph_source_embed = message_node[total_source, :].squeeze(1)
            graph_target_embed = message_node[total_target, :].squeeze(1)
            graph_edge_embed = graph_source_embed + target_relation - graph_target_embed
            edge_message = edge_source_message + relation_embed - edge_target_message
            attention = torch.cat([graph_edge_embed, edge_message], dim=1)
            attention = torch.relu(self._modules['Attention1_{}'.format(depth)](attention))
            attention = torch.sigmoid(self._modules['Attention2_{}'.format(depth)](attention))

        # communicate
        #      agg_message = index_select_ND(message_edge, a2b)
        #     agg_message = agg_message.sum(dim=1) * agg_message.max(dim=1)[0]

        #        agg_message = gnn_spmm(e2n_sp, message_edge)/e2n_sp.sum(1, keepdim=True)
        message_edge = (message_edge * attention)
        agg_message = gnn_spmm(e2n_sp, message_edge)
        agg_message2 = self.communicate_mlp(torch.cat([agg_message, message_node, input_node], 1))
        # =============================================================================
        #
        # =============================================================================
        # readout

        # node_hiddens = agg_message2

        a_message = torch.relu(self.gru(agg_message2, graph_sizes))
        node_hiddens = self.act_func(self.W_o(a_message))  # num_nodes x hidden
        node_hiddens = self.dropout_layer(node_hiddens)  # num_nodes x hidden

        # Readout
        mol_vecs = []
        a_start = 0
        for a_size in graph_sizes:
            if a_size == 0:
                assert 0
            cur_hiddens = node_hiddens.narrow(0, a_start, a_size)
            mol_vecs.append(cur_hiddens.mean(0))
            a_start += a_size
        mol_vecs = torch.stack(mol_vecs, dim=0)

        source_embed = node_hiddens[source_node, :]
        target_embed = node_hiddens[target_node, :]

        # print(mol_vecs.shape, source_embed.shape, target_embed.shape)

        return mol_vecs, source_embed, target_embed


from torch.autograd import Variable


class MySpMM(torch.autograd.Function):

    @staticmethod
    def forward(ctx, sp_mat, dense_mat):
        ctx.save_for_backward(sp_mat, dense_mat)

        return torch.mm(sp_mat, dense_mat)

    @staticmethod
    def backward(ctx, grad_output):
        sp_mat, dense_mat = ctx.saved_variables
        grad_matrix1 = grad_matrix2 = None

        assert not ctx.needs_input_grad[0]
        if ctx.needs_input_grad[1]:
            grad_matrix2 = Variable(torch.mm(sp_mat.data.t(), grad_output.data))

        return grad_matrix1, grad_matrix2


def gnn_spmm(sp_mat, dense_mat):
    return MySpMM.apply(sp_mat, dense_mat)


def get_activation_function(activation):
    """
    Gets an activation function module given the name of the activation.

    :param activation: The name of the activation function.
    :return: The activation function module.
    """
    if activation == 'ReLU':
        return nn.ReLU()
    elif activation == 'LeakyReLU':
        return nn.LeakyReLU(0.1)
    elif activation == 'PReLU':
        return nn.PReLU()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'SELU':
        return nn.SELU()
    elif activation == 'ELU':
        return nn.ELU()
    else:
        raise ValueError('Activation "{}" not supported.'.format(activation))


class BatchGRU(nn.Module):
    def __init__(self, hidden_size=300):
        super(BatchGRU, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True,
                          bidirectional=True)
        self.bias = nn.Parameter(torch.Tensor(self.hidden_size))
        self.bias.data.uniform_(-1.0 / math.sqrt(self.hidden_size),
                                1.0 / math.sqrt(self.hidden_size))

    def forward(self, node, a_scope):
        hidden = node
        #  print(hidden.shape)
        message = F.relu(node + self.bias)
        MAX_node_len = max(a_scope)
        # padding
        message_lst = []
        hidden_lst = []
        a_start = 0
        for i in a_scope:
            i = int(i)
            if i == 0:
                assert 0
            cur_message = message.narrow(0, a_start, i)
            cur_hidden = hidden.narrow(0, a_start, i)
            hidden_lst.append(cur_hidden.max(0)[0].unsqueeze(0).unsqueeze(0))
            a_start += i
            cur_message = torch.nn.ZeroPad2d((0, 0, 0, MAX_node_len - cur_message.shape[0]))(cur_message)
            message_lst.append(cur_message.unsqueeze(0))

        message_lst = torch.cat(message_lst, 0)
        hidden_lst = torch.cat(hidden_lst, 1)
        hidden_lst = hidden_lst.repeat(2, 1, 1)
        cur_message, cur_hidden = self.gru(message_lst, hidden_lst)

        # unpadding
        cur_message_unpadding = []
        kk = 0
        for a_size in a_scope:
            a_size = int(a_size)
            cur_message_unpadding.append(cur_message[kk, :a_size].view(-1, 2 * self.hidden_size))
            kk += 1
        cur_message_unpadding = torch.cat(cur_message_unpadding, 0)

        #   message = torch.cat([torch.cat([message.narrow(0, 0, 1), message.narrow(0, 0, 1)], 1),
        #                        cur_message_unpadding], 0)
        #   print(cur_message_unpadding.shape)
        return cur_message_unpadding



class RGCN(nn.Module):
    def __init__(self, params):
        super(RGCN, self).__init__()

        self.max_label_value = params.max_label_value
        self.inp_dim = params.inp_dim
        self.emb_dim = params.emb_dim
        self.attn_rel_emb_dim = params.attn_rel_emb_dim
        self.num_rels = params.num_rels
        self.aug_num_rels = params.aug_num_rels
        self.num_bases = params.num_bases
        self.num_hidden_layers = params.num_gcn_layers
        self.dropout = params.dropout
        self.edge_dropout = params.edge_dropout
        # self.aggregator_type = params.gnn_agg_type
        self.has_attn = params.has_attn

        self.device = params.device

        if self.has_attn:
            self.attn_rel_emb = nn.Embedding(self.num_rels, self.attn_rel_emb_dim, sparse=False)
        else:
            self.attn_rel_emb = None

        if params.gnn_agg_type == "sum":
            self.aggregator = SumAggregator(self.emb_dim)
        elif params.gnn_agg_type == "mlp":
            self.aggregator = MLPAggregator(self.emb_dim)
        elif params.gnn_agg_type == "gru":
            self.aggregator = GRUAggregator(self.emb_dim)

        self.layers = nn.ModuleList()
        #input layer
        self.layers.append(RGCNBasisLayer(self.inp_dim,
                         self.emb_dim,
                         # self.input_basis_weights,
                         self.aggregator,
                         self.attn_rel_emb_dim,
                         self.aug_num_rels,
                         self.num_bases,
                         activation=F.relu,
                         dropout=self.dropout,
                         edge_dropout=self.edge_dropout,
                         is_input_layer=True,
                         has_attn=self.has_attn))
        #hidden layer
        for idx in range(self.num_hidden_layers - 1):
            self.layers.append(RGCNBasisLayer(self.emb_dim,
                         self.emb_dim,
                         # self.basis_weights,
                         self.aggregator,
                         self.attn_rel_emb_dim,
                         self.aug_num_rels,
                         self.num_bases,
                         activation=F.relu,
                         dropout=self.dropout,
                         edge_dropout=self.edge_dropout,
                         has_attn=self.has_attn))


    def forward(self,g):
        for layer in self.layers:
            layer(g, self.attn_rel_emb)
        return g.ndata.pop('h')

class RGCNLayer(nn.Module):
    def __init__(self, inp_dim, out_dim, aggregator, bias=None, activation=None, dropout=0.0, edge_dropout=0.0, is_input_layer=False):
        super(RGCNLayer, self).__init__()
        self.bias = bias
        self.activation = activation

        if self.bias:
            self.bias = nn.Parameter(torch.Tensor(out_dim))
            nn.init.xavier_uniform_(self.bias,
                                    gain=nn.init.calculate_gain('relu'))

        self.aggregator = aggregator

        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        if edge_dropout:
            self.edge_dropout = nn.Dropout(edge_dropout)
        else:
            self.edge_dropout = Identity()    #Identify需要注意，和原模型有一定出入

    # define how propagation is done in subclass
    def propagate(self, g):
        raise NotImplementedError

    def forward(self, g, attn_rel_emb=None):

        self.propagate(g, attn_rel_emb)

        # apply bias and activation
        node_repr = g.ndata['h']
        if self.bias:
            node_repr = node_repr + self.bias
        if self.activation:
            node_repr = self.activation(node_repr)
        if self.dropout:
            node_repr = self.dropout(node_repr)

        g.ndata['h'] = node_repr

        if self.is_input_layer:
            g.ndata['repr'] = g.ndata['h'].unsqueeze(1)
        else:
            g.ndata['repr'] = torch.cat([g.ndata['repr'], g.ndata['h'].unsqueeze(1)], dim=1)

class RGCNBasisLayer(RGCNLayer):
    def __init__(self, inp_dim, out_dim, aggregator, attn_rel_emb_dim, num_rels, num_bases=-1, bias=None,
                 activation=None, dropout=0.0, edge_dropout=0.0, is_input_layer=False, has_attn=False):
        super(
            RGCNBasisLayer,
            self).__init__(
            inp_dim,
            out_dim,
            aggregator,
            bias,
            activation,
            dropout=dropout,
            edge_dropout=edge_dropout,
            is_input_layer=is_input_layer)
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.attn_rel_emb_dim = attn_rel_emb_dim
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.is_input_layer = is_input_layer
        self.has_attn = has_attn

        if self.num_bases <= 0 or self.num_bases > self.num_rels:
            self.num_bases = self.num_rels

        # add basis weights
        # self.weight = basis_weights
        self.weight = nn.Parameter(torch.Tensor(self.num_bases, self.inp_dim, self.out_dim))
        self.w_comp = nn.Parameter(torch.Tensor(self.num_rels, self.num_bases))
        print("here: ______________________________________________________________________________________")
        print(self.w_comp)
        print(self.w_comp.shape)

        if self.has_attn:
            self.A = nn.Linear(2 * self.inp_dim + 2 * self.attn_rel_emb_dim, inp_dim)
            self.B = nn.Linear(inp_dim, 1)

        self.self_loop_weight = nn.Parameter(torch.Tensor(self.inp_dim, self.out_dim))

        nn.init.xavier_uniform_(self.self_loop_weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.w_comp, gain=nn.init.calculate_gain('relu'))

    import multiprocessing

    def propagate(self, g, attn_rel_emb=None):
        # generate all weights from bases
        #torch.cuda.init()
        print(self.num_bases)
        print(self.inp_dim)
        print(self.out_dim)
        print(self.weight.shape)
        weight = self.weight.view(self.num_bases,
                                  self.inp_dim * self.out_dim)
        print(weight.shape)
        print(self.w_comp)
        weight = torch.matmul(self.w_comp, weight).view(
            self.num_rels, self.inp_dim, self.out_dim)
        print("after")
        print(self.w_comp)
        print(weight.shape)
        g.edata['w'] = self.edge_dropout(torch.ones(g.number_of_edges(), 1).to(weight.device))

        input_ = 'feat' if self.is_input_layer else 'h'

        def msg_func(edges):
            w = weight.index_select(0, edges.data['type'])
            msg = edges.data['w'] * torch.bmm(edges.src[input_].unsqueeze(1), w).squeeze(1)
            curr_emb = torch.mm(edges.dst[input_], self.self_loop_weight)  # (B, F)

            if self.has_attn:
                e = torch.cat([edges.src[input_], edges.dst[input_], attn_rel_emb(edges.data['type']), attn_rel_emb(edges.data['label'])], dim=1)
                a = torch.sigmoid(self.B(F.relu(self.A(e))))
            else:
                a = torch.ones((len(edges), 1)).to(device=w.device)

            return {'curr_emb': curr_emb, 'msg': msg, 'alpha': a}

        g.update_all(msg_func, self.aggregator, None)


class Aggregator(nn.Module):
    def __init__(self, emb_dim):
        super(Aggregator, self).__init__()

    def forward(self, node):
        curr_emb = node.mailbox['curr_emb'][:, 0, :]  # (B, F)
        nei_msg = torch.bmm(node.mailbox['alpha'].transpose(1, 2), node.mailbox['msg']).squeeze(1)  # (B, F)
        # nei_msg, _ = torch.max(node.mailbox['msg'], 1)  # (B, F)

        new_emb = self.update_embedding(curr_emb, nei_msg)

        return {'h': new_emb}

    @abc.abstractmethod
    def update_embedding(curr_emb, nei_msg):
        raise NotImplementedError


class SumAggregator(Aggregator):
    def __init__(self, emb_dim):
        super(SumAggregator, self).__init__(emb_dim)

    def update_embedding(self, curr_emb, nei_msg):
        new_emb = nei_msg + curr_emb

        return new_emb


class MLPAggregator(Aggregator):
    def __init__(self, emb_dim):
        super(MLPAggregator, self).__init__(emb_dim)
        self.linear = nn.Linear(2 * emb_dim, emb_dim)

    def update_embedding(self, curr_emb, nei_msg):
        inp = torch.cat((nei_msg, curr_emb), 1)
        new_emb = F.relu(self.linear(inp))

        return new_emb


class GRUAggregator(Aggregator):
    def __init__(self, emb_dim):
        super(GRUAggregator, self).__init__(emb_dim)
        self.gru = nn.GRUCell(emb_dim, emb_dim)

    def update_embedding(self, curr_emb, nei_msg):
        new_emb = self.gru(nei_msg, curr_emb)

        return new_emb


