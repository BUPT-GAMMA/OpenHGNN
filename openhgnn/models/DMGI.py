import dgl
from dgl.nn.pytorch import GATConv

from openhgnn.models import register_model, BaseModel
import torch.nn as nn
import numpy as np

import dgl.nn.pytorch as dglnn
import torch
import torch.nn.functional as F




@register_model('DMGI')
class DMGI(BaseModel):
    @classmethod
    def build_model_from_args(cls, args, hg):
        etypes = hg.canonical_etypes
        mps = []
        for etype in etypes:
            if etype[0] == args.category:
                for dst_e in etypes:
                    if etype[0] == dst_e[2] and etype[2] == dst_e[0]:
                        mps.append([etype, dst_e])
        num_nodes = hg.num_nodes(args.category)

        return cls(meta_paths=mps, sc=args.sc,
                   category=args.category, in_size=args.in_dim,
                   hid_unit=args.hid_unit, dropout=args.dropout,
                   num_nodes=num_nodes, num_classes=args.num_classes,
                   isSemi=args.isSemi,isAttn=args.isAttn, isBias=args.isBias)


    def __init__(self, meta_paths, sc, category, in_size, hid_unit,
                 dropout, num_nodes, num_classes, isBias, isAttn, isSemi):
        super(DMGI, self).__init__()
        self.category = category
        # self.layers = nn.ModuleList()
        self.hid = hid_unit
        self.meta_paths = meta_paths
        self.isAttn = isAttn
        self.isSemi = isSemi
        self.sc = sc
        self.gcn = nn.ModuleList([GraphConvLayer(in_size,
                                                 hid_unit,
                                                 dropout,
                                                 bias=isBias) for _ in range(len(meta_paths))])
        self.disc = Discriminator(hid_unit)
        self.readout = AvgReadout()
        self.readout_act_func = nn.Sigmoid()

        self.num_nodes = num_nodes
        # num_head = 1
        self.H = nn.Parameter(torch.FloatTensor(1, num_nodes, hid_unit))

        self.logistic = LogReg(hid_unit, num_classes)

        if self.isAttn:
            self.attn = Attention(hid_units=hid_unit, num_mps=len(meta_paths), num_ndoes=num_nodes)

        self.init_weight()
        print("category:{}, category's classes:{}, isBias:{},"
              " isAttn:{}, isSemi:{}".format(category, num_classes,isBias,isAttn,isSemi))
    def init_weight(self):
        nn.init.xavier_normal_(self.H)
    # samp_bias1, samp_bias2  default  None
    def forward(self, hg, samp_bias1=None, samp_bias2=None ):

        h_1_all = [];h_2_all = [];c_all = [];logits = []
        result = {}
        # process features
        features = hg.srcdata['h']
        feats = self.normal_feat(features, self.meta_paths)
        # shuffled features
        shuf_feats = self.shuf_feats(feats)

        for idx, meta_path in enumerate(self.meta_paths):
            new_g = dgl.metapath_reachable_graph(hg, meta_path)
            for i in range(self.sc):
                new_g = dgl.add_self_loop(new_g)

            h_1 = self.gcn[idx](new_g, feats[idx])
            c = self.readout(h_1)
            c = self.readout_act_func(c)
            h_2 = self.gcn[idx](new_g, shuf_feats[idx])


            logit = self.disc(c, h_1, h_2, samp_bias1, samp_bias2)
            h_1_all.append(h_1.unsqueeze(0))
            h_2_all.append(h_2.unsqueeze(0))
            c_all.append(c)
            logits.append(logit)
        result['logits'] = logits

        # Attention or not
        if self.isAttn:
            h_1_all_lst = [];h_2_all_lst = []
            h_1_all_, h_2_all_, c_all_ = self.attn(h_1_all, h_2_all, c_all)
            h_1_all_lst.append(h_1_all_); h_2_all_lst.append(h_2_all_);

            h_1_all = torch.mean(torch.cat(h_1_all, 0), 0).unsqueeze(0)
            h_2_all = torch.mean(torch.cat(h_2_all, 0), 0).unsqueeze(0)

        else:
            h_1_all = torch.mean(torch.cat(h_1_all, 0), 0).unsqueeze(0)
            h_2_all = torch.mean(torch.cat(h_2_all, 0), 0).unsqueeze(0)

        # Lcs = [Z − AVG { H(r)|r∈ R }]^2 - [Z − AVG { ~H(r)|r∈ R }]^2
        pos_reg_loss = ((self.H - h_1_all) ** 2).sum()
        neg_reg_loss = ((self.H - h_2_all) ** 2).sum()
        reg_loss = pos_reg_loss - neg_reg_loss
        result['reg_loss'] = reg_loss

        # semi-supervised module
        if self.isSemi:
            semi = self.logistic(self.H).squeeze(0)
            result['semi'] = semi

        # result: ['logits','reg_loss','semi']
        return result


    '''feature_normalize'''
    def normal_feat(self, feats, meta_paths):
        feat = []
        feats = feats[self.category].data
        for mp in meta_paths:
            rowsum = feats.sum(1)
            r_inv = torch.pow(rowsum, -1).flatten()  
            r_inv[torch.isinf(r_inv)] = 0.
            r_mat_inv = torch.diag(r_inv)
            feats = torch.spmm(r_mat_inv, feats)
            feat.append(feats)
        return feat
    '''corrupt the original attribute matrix by shuffling it'''
    def shuf_feats(self, feats):
        shuf_feats = []
        for feat in feats:
            idx = np.random.permutation(feat.shape[0])
            shuf = feat[idx]
            shuf_feats.append(shuf)
        return shuf_feats


'''The encoder is a single–layered GCN'''
class GraphConvLayer(nn.Module):
    def __init__(self,
                 in_feat,
                 out_feat,
                 dropout,
                 bias=True):
        super(GraphConvLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.bias = bias
        self.dropout = dropout
        self.conv = dglnn.GraphConv(in_feats=in_feat,
                                    out_feats=out_feat,
                                    bias=bias,
                                    allow_zero_in_degree=True)
        self.act = nn.ReLU()

    def forward(self, hg, feats):
        feats = F.dropout(feats, self.dropout, training=self.training)
        res = self.conv(hg, feats)
        res = self.act(res)

        return res
'''In the experiments, some relation type is more beneficial for a 
certain downstream task than others. Therefore, we can adopt the 
attention mechanism'''
class Attention(nn.Module):
    def __init__(self, hid_units, num_mps, num_ndoes):
        super(Attention, self).__init__()
        self.num_mps = num_mps
        self.hid_units = hid_units
        self.num_nodes = num_ndoes
        self.A = nn.ModuleList([nn.Linear(hid_units, 1) for _ in range(num_mps)])
        self.weight_init()
    def weight_init(self):
        for i in range(self.num_mps):
            nn.init.xavier_normal_(self.A[i].weight)
            self.A[i].bias.data.fill_(0.0)
    def forward(self, feat_pos, feat_neg, summary):
        feat_pos, feat_pos_attn = self.attn_feature(feat_pos)
        feat_neg, feat_neg_attn = self.attn_feature(feat_neg)
        summary, summary_attn = self.attn_summary(summary)

        return feat_pos, feat_neg, summary

    def attn_feature(self, features):
        features_attn = []
        for i in range(self.num_mps):
            features_attn.append((self.A[i](features[i].squeeze())))

        features_attn = F.softmax(torch.cat(features_attn, 1), -1)

        features = torch.cat(features,1).squeeze(0)

        features_attn_reshaped = features_attn.transpose(1, 0).contiguous().view(-1, 1)

        features = features * features_attn_reshaped.expand_as(features)

        features = features.view(self.num_mps, self.num_nodes, self.hid_units).sum(0).unsqueeze(0)
        return features, features_attn
    def attn_summary(self, features):
        features_attn = []
        for i in range(self.num_mps):
            features_attn.append((self.A[i](features[i].squeeze())))
        features_attn = F.softmax(torch.cat(features_attn), dim=-1).unsqueeze(1)
        features = torch.stack(features, 0)
        features_attn_expanded = features_attn.expand_as(features)
        features = (features * features_attn_expanded).sum(0).unsqueeze(0)

        return features, features_attn

'''
D is a discriminator that scores patchsummary representation pairs. 
In this paper, we apply a simple bilinear scoring function as it 
empirically performs the best in our experiments:'''
class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k_bilinear = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = c.expand_as(h_pl)

        sc_1 = torch.squeeze(self.f_k_bilinear(h_pl, c_x), 1) # sc_1 = 1 x nb_nodes
        sc_2 = torch.squeeze(self.f_k_bilinear(h_mi, c_x), 1) # sc_2 = 1 x nb_nodes

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2
        logits = torch.cat((sc_1, sc_2), 0)

        return logits

'''considering the efficiency of the method, we simply employ average pooling'''
class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()
    def forward(self, seq):
        return torch.mean(seq, 0)

'''logreg'''
class LogReg(nn.Module):
    # 64 hid    label.shape[1]
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret



