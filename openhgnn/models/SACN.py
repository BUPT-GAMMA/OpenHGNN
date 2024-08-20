from . import BaseModel, register_model
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_
from dgl import function as fn
from numpy.random.mtrand import set_state
import pandas
import torch
import math
@register_model('SACN')
class SACN(BaseModel):
    @classmethod
    def build_model_from_args(cls, config):
        return cls(config)

    def __init__(self, config):
        super().__init__()

        self.model = WGCN_Base(config)

    def forward(self, *args):
        return self.model(*args)

    def extra_loss(self):
        pass
class GraphConvolution(torch.nn.Module):
    def __init__(self, in_features, out_features, num_relations, bias=True, wsi=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.FloatTensor(in_features, out_features))
        self.num_relations = num_relations
        self.alpha = torch.nn.Embedding(num_relations + 1, 1, padding_idx=0)
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.wsi = wsi

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, g, all_edge_type, input):
        with g.local_scope():
            feats = torch.mm(input, self.weight)
            g.srcdata['ft'] = feats
            if not self.wsi:
                train_edge_num = int(
                    (all_edge_type.shape[0] - input.shape[0]) / 2)
                transpose_all_edge_type = torch.cat((all_edge_type[train_edge_num:train_edge_num * 2],
                                                    all_edge_type[:train_edge_num], all_edge_type[-input.shape[0]:]))
            else:
                train_edge_num = int((all_edge_type.shape[0]))
                transpose_all_edge_type = torch.cat((all_edge_type[train_edge_num:train_edge_num * 2],
                                                    all_edge_type[:train_edge_num]))
            alp = self.alpha(all_edge_type) + \
                self.alpha(transpose_all_edge_type)
            g.edata['a'] = alp

            g.update_all(fn.u_mul_e('ft', 'a', 'm'), fn.sum('m', 'ft'))

            output = g.dstdata['ft']

            if self.bias is not None:
                return output + self.bias
            else:
                return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'
class WGCN_Base(torch.nn.Module):
    def __init__(self,  args):
        super(WGCN_Base, self).__init__()
        num_entities=args.num_entities
        num_relations=args.num_relations
        self.rat = args.rat
        self.wni = args.wni

        self.fa = args.final_act
        self.fb = args.final_bn
        self.fd = args.final_drop

        self.decoder_name = args.decoder
        self.num_layers = args.n_layer
        self.emb_e = torch.nn.Embedding(
            num_entities, args.init_emb_size, padding_idx=0)
        self.emb_rel = torch.nn.Embedding(
            num_relations, args.embedding_dim, padding_idx=0)

        nn.init.xavier_normal_(
            self.emb_e.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.emb_rel.weight,
                               gain=nn.init.calculate_gain('relu'))

        if self.num_layers == 3:
            self.gc1 = GraphConvolution(
                args.init_emb_size, args.gc1_emb_size, num_relations, wsi=args.wsi)
            self.gc2 = GraphConvolution(
                args.gc1_emb_size, args.gc1_emb_size, num_relations, wsi=args.wsi)
            self.gc3 = GraphConvolution(
                args.gc1_emb_size, args.embedding_dim, num_relations, wsi=args.wsi)
        elif self.num_layers == 2:
            self.gc2 = GraphConvolution(
                args.init_emb_size, args.gc1_emb_size, num_relations, wsi=args.wsi)
            self.gc3 = GraphConvolution(
                args.gc1_emb_size, args.embedding_dim, num_relations, wsi=args.wsi)
        else:
            self.gc3 = GraphConvolution(
                args.init_emb_size, args.embedding_dim, num_relations, wsi=args.wsi)

        self.inp_drop = torch.nn.Dropout(args.input_dropout)
        self.hidden_drop = torch.nn.Dropout(args.dropout_rate)
        self.feature_map_drop = torch.nn.Dropout(args.dropout_rate)
        self.loss = torch.nn.BCELoss()
        self.conv1 = nn.Conv1d(2, args.channels, args.kernel_size, stride=1,
                               padding=int(math.floor(args.kernel_size / 2)))
        self.bn0 = torch.nn.BatchNorm1d(2)
        self.bn1 = torch.nn.BatchNorm1d(args.channels)
        self.bn2 = torch.nn.BatchNorm1d(args.embedding_dim)
        self.register_parameter('b', nn.Parameter(torch.zeros(num_entities)))
        self.fc = torch.nn.Linear(
            args.embedding_dim * args.channels, args.embedding_dim)
        self.bn3 = torch.nn.BatchNorm1d(args.gc1_emb_size)
        self.bn4 = torch.nn.BatchNorm1d(args.embedding_dim)
        self.bn5 = torch.nn.BatchNorm1d(args.gc1_emb_size)
        self.bn_init = torch.nn.BatchNorm1d(args.init_emb_size)
        self.args=args
        print(num_entities, num_relations)

        if args.decoder == "transe":
            self.decoder = self.transe
            self.gamma = args.gamma
        elif args.decoder == "distmult":
            self.decoder = self.distmult
            self.bias = nn.Parameter(torch.zeros(num_entities))
        elif args.decoder == "conve":
            self.decoder = self.conve
        else:
            raise NotImplementedError

    def conve(self, e1_embedded, rel_embedded, e1_embedded_all):
        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 1)
        stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)

        x = self.conv1(x)
        x = self.bn1(x)

        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(e1_embedded.shape[0], -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = torch.mm(x, e1_embedded_all.transpose(1, 0))
        pred = torch.sigmoid(x)
        return pred

    def transe(self, e1_embedded, rel_embedded, e1_embedded_all):
        obj_emb = e1_embedded + rel_embedded

        x = self.gamma - \
            torch.norm(obj_emb - e1_embedded_all.unsqueeze(0), p=1, dim=2)
        pred = torch.sigmoid(x)

        return pred

    def distmult(self, e1_embedded, rel_embedded, e1_embedded_all):
        obj_emb = e1_embedded * rel_embedded

        x = torch.mm(obj_emb.squeeze(1), e1_embedded_all.transpose(1, 0))
        x += self.bias.expand_as(x)
        pred = torch.sigmoid(x)

        return pred

    def init(self):
        xavier_normal_(self.emb_e.weight.data)
        xavier_normal_(self.emb_rel.weight.data)
        xavier_normal_(self.gc1.weight.data)
        xavier_normal_(self.gc2.weight.data)
        xavier_normal_(self.gc3.weight.data)

    def forward(self, g, all_edge, e1, rel, entity_id):
        emb_initial = self.emb_e(entity_id)

        if self.num_layers == 3:
            x = self.gc1(g, all_edge, emb_initial)
            x = self.bn5(x)
            x = torch.tanh(x)
            x = F.dropout(x, self.args.dropout_rate, training=self.training)
        else:
            x = emb_initial

        if self.num_layers >= 2:
            x = self.gc2(g, all_edge, x)
            x = self.bn3(x)
            x = torch.tanh(x)
            x = F.dropout(x, self.args.dropout_rate, training=self.training)

        if self.num_layers >= 1:
            x = self.gc3(g, all_edge, x)

        if self.fb:
            x = self.bn4(x)
        if self.fa:
            x = torch.tanh(x)
        if self.fd:
            x = F.dropout(x, self.args.dropout_rate, training=self.training)

        e1_embedded_all = x

        e1_embedded = e1_embedded_all[e1]
        rel_embedded = self.emb_rel(rel)

        pred = self.decoder(e1_embedded, rel_embedded, e1_embedded_all)
        return pred