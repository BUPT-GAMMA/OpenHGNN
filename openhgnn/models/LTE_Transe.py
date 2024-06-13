from . import BaseModel, register_model

@register_model('LTE_Transe')
class LTE_Transe(BaseModel):

    @classmethod
    def build_model_from_args(cls, config):
        return cls(config)

    def __init__(self, config):
        super().__init__()

        self.model = GCN_TransE(config)

    def forward(self, *args):
        return self.model(*args)

    def extra_loss(self):
        pass

import torch
from torch import nn
import dgl
from ..layers.rgcn_layer import RelGraphConv
from ..layers.compgcn_layer import CompGCNCov
import torch.nn.functional as F


class GCNs(nn.Module):
    def __init__(self,args, num_ent, num_rel, num_base, init_dim, gcn_dim, embed_dim, n_layer, edge_type, edge_norm,
                 conv_bias=True, gcn_drop=0., opn='mult', wni=False, wsi=False, encoder='compgcn', use_bn=True, ltr=True):
        super(GCNs, self).__init__()
        num_ent=args.num_ent
        num_rel=args.num_rel
        num_base=args.num_base
        init_dim=args.init_dim
        gcn_dim=args.gcn_dim
        embed_dim=args.embed_dim
        n_layer=args.n_layer
        edge_type=args.edge_type
        edge_norm=args.edge_norm
        conv_bias = True
        if args.conv_bias is not None:
            conv_bias=args.conv_bias
        gcn_drop = 0.
        if args.gcn_drop is not None:
            gcn_drop=args.gcn_drop
        opn = 'mult'
        if args.opn is not None:
            opn=args.opn
        wni = False
        if args.wni is not None:
            wni=args.wni
        wsi = False
        if args.wsi is not None:
            wsi=args.wsi
        encoder = 'compgcn'
        if args.encoder is not None:
            encoder=args.encoder
        use_bn = True
        if args.use_bn is not None:
            use_bn=args.use_bn
        ltr = True
        if args.ltr is not None:
            ltr=args.ltr
        self.act = torch.tanh
        self.loss = nn.BCELoss()
        self.num_ent, self.num_rel, self.num_base = num_ent, num_rel, num_base
        self.init_dim, self.gcn_dim, self.embed_dim = init_dim, gcn_dim, embed_dim
        self.conv_bias = conv_bias
        self.gcn_drop = gcn_drop
        self.opn = opn
        self.edge_type = edge_type  # [E]
        self.edge_norm = edge_norm  # [E]
        self.n_layer = n_layer

        self.wni = wni

        self.encoder = encoder

        self.init_embed = self.get_param([self.num_ent, self.init_dim])
        self.init_rel = self.get_param([self.num_rel * 2, self.init_dim])

        if encoder == 'compgcn':
            if n_layer < 3:
                self.conv1 = CompGCNCov(self.init_dim, self.gcn_dim, self.act, conv_bias, gcn_drop, opn, num_base=-1,
                                        num_rel=self.num_rel, wni=wni, wsi=wsi, use_bn=use_bn, ltr=ltr)
                self.conv2 = CompGCNCov(self.gcn_dim, self.embed_dim, self.act, conv_bias, gcn_drop,
                                        opn, wni=wni, wsi=wsi, use_bn=use_bn, ltr=ltr) if n_layer == 2 else None
            else:
                self.conv1 = CompGCNCov(self.init_dim, self.gcn_dim, self.act, conv_bias, gcn_drop, opn, num_base=-1,
                                        num_rel=self.num_rel, wni=wni, wsi=wsi, use_bn=use_bn, ltr=ltr)
                self.conv2 = CompGCNCov(self.gcn_dim, self.gcn_dim, self.act, conv_bias, gcn_drop, opn, num_base=-1,
                                        num_rel=self.num_rel, wni=wni, wsi=wsi, use_bn=use_bn, ltr=ltr)
                self.conv3 = CompGCNCov(self.gcn_dim, self.embed_dim, self.act, conv_bias, gcn_drop,
                                        opn, wni=wni, wsi=wsi, use_bn=use_bn, ltr=ltr)
        elif encoder == 'rgcn':
            self.conv1 = RelGraphConv(self.init_dim, self.gcn_dim, self.num_rel*2, "bdd",
                                      num_bases=self.num_base, activation=self.act, self_loop=(not wsi), dropout=gcn_drop, wni=wni)
            self.conv2 = RelGraphConv(self.gcn_dim, self.embed_dim, self.num_rel*2, "bdd", num_bases=self.num_base,
                                      activation=self.act, self_loop=(not wsi), dropout=gcn_drop, wni=wni) if n_layer == 2 else None

        self.bias = nn.Parameter(torch.zeros(self.num_ent))

    def get_param(self, shape):
        param = nn.Parameter(torch.Tensor(*shape))
        nn.init.xavier_normal_(param, gain=nn.init.calculate_gain('relu'))
        return param

    def calc_loss(self, pred, label):
        return self.loss(pred, label)

    def forward_base(self, g, subj, rel, drop1, drop2):
        """
        :param g: graph
        :param sub: subjects in a batch [batch]
        :param rel: relations in a batch [batch]
        :param drop1: dropout rate in first layer
        :param drop2: dropout rate in second layer
        :return: sub_emb: [batch, D]
                 rel_emb: [num_rel*2, D]
                 x: [num_ent, D]
        """
        x, r = self.init_embed, self.init_rel  # embedding of relations

        if self.n_layer > 0:
            if self.encoder == 'compgcn':
                if self.n_layer < 3:
                    x, r = self.conv1(g, x, r, self.edge_type, self.edge_norm)
                    x = drop1(x)  # embeddings of entities [num_ent, dim]
                    x, r = self.conv2(
                        g, x, r, self.edge_type, self.edge_norm) if self.n_layer == 2 else (x, r)
                    x = drop2(x) if self.n_layer == 2 else x
                else:
                    x, r = self.conv1(g, x, r, self.edge_type, self.edge_norm)
                    x = drop1(x)  # embeddings of entities [num_ent, dim]
                    x, r = self.conv2(g, x, r, self.edge_type, self.edge_norm)
                    x = drop1(x)  # embeddings of entities [num_ent, dim]
                    x, r = self.conv3(g, x, r, self.edge_type, self.edge_norm)
                    x = drop2(x)
            elif self.encoder == 'rgcn':
                x = self.conv1(g, x, self.edge_type,
                               self.edge_norm.unsqueeze(-1))
                x = drop1(x)  # embeddings of entities [num_ent, dim]
                x = self.conv2(
                    g, x, self.edge_type, self.edge_norm.unsqueeze(-1)) if self.n_layer == 2 else x
                x = drop2(x) if self.n_layer == 2 else x

        # filter out embeddings of subjects in this batch
        sub_emb = torch.index_select(x, 0, subj)
        # filter out embeddings of relations in this batch
        rel_emb = torch.index_select(r, 0, rel)

        return sub_emb, rel_emb, x


class GCN_TransE(GCNs):
    def __init__(self, args):
        super(GCN_TransE, self).__init__(args)

        self.drop = nn.Dropout(args.hid_drop)
        self.gamma = args.gamma

    def forward(self, g, subj, rel):
        """
        :param g: dgl graph
        :param sub: subject in batch [batch_size]
        :param rel: relation in batch [batch_size]
        :return: score: [batch_size, ent_num], the prob in link-prediction
        """
        sub_emb, rel_emb, all_ent = self.forward_base(
            g, subj, rel, self.drop, self.drop)
        obj_emb = sub_emb + rel_emb

        x = self.gamma - torch.norm(obj_emb.unsqueeze(1) - all_ent, p=1, dim=2)

        score = torch.sigmoid(x)

        return score

