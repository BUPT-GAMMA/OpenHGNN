import dgl
from . import register_model, BaseModel
import torch.nn as nn
import numpy as np

import dgl.nn.pytorch as dglnn
import torch
import torch.nn.functional as F

@register_model('DMGI')
class DMGI(BaseModel):
    r"""
    **Title:** Unsupervised Attributed Multiplex Network Embedding

    **Authors:** Chanyoung Park, Donghyun Kim, Jiawei Han, Hwanjo Yu

    DMGI was introduced in `[paper] <https://ojs.aaai.org//index.php/AAAI/article/view/5985>`_
    and parameters are defined as follows:

    Input
    ------

        meta_paths : dict
            Extract metapaths from graph
        sc : int
            Introducing a weight to self-connections
        category : string
            The category of the nodes to be classificated
        in_size : int
            Input feature size
        hidden_dim : int
            Hidden units size
        dropout : float
            Dropout rate on feature. Defaults: ``0.5``.
        num_nodes : int
            The number of all nodes of category in graph
        num_classes : int
            The numbers of category's types
        isBias :bool
            If True, adds a learnable bias to the output.Defaults: ``False``.
        isAttn : bool
            If True, adopt the attention mechanism to calculate loss . Defaults: ``False``.
        isSemi : bool
            If True, add isSemi's loss to calculate loss
        nheads : int
            the num of attention head

    Parameters
    ----------
        H : torch.FloatTensor
            The learnable weight tensor.

        gcn : The encoder is a single-layer GCN:

            .. math::
              \begin{equation}
                \mathbf{H}^{(r)}=g_{r}\left(\mathbf{X}, \mathbf{A}^{(r)} \mid \mathbf{W}^{(r)}\right)=\sigma\left(\hat{\mathbf{D}}_{r}^{-\frac{1}{2}} \hat{\mathbf{A}}^{(r)} \hat{\mathbf{D}}_{r}^{-\frac{1}{2}} \mathbf{X} \mathbf{W}^{(r)}\right)
              \end{equation}

            where :math:`\hat{\mathbf{A}}^{(r)}=\mathbf{A}^{(r)}+w \mathbf{I}_{n}` ,
            :math:`\hat{D}_{i i}=\sum_{j} \hat{A}_{i j}`


    """
    @classmethod
    def build_model_from_args(cls, args, hg):
        etypes = hg.canonical_etypes
        mps = []

        for etype in etypes:
            if etype[0] == args.category:
                for dst_e in etypes:
                    if etype[0] == dst_e[2] and etype[2] == dst_e[0]:
                        if etype[0] != etype[2]:
                            mps.append([etype, dst_e])
        num_nodes = hg.num_nodes(args.category)

        return cls(meta_paths=mps, sc=args.sc,
                   category=args.category, in_size=args.in_dim,
                   hidden_dim=args.hidden_dim, nheads=args.num_heads,dropout=args.dropout,
                   num_nodes=num_nodes, num_classes=args.num_classes,
                   isSemi=args.isSemi,isAttn=args.isAttn, isBias=args.isBias)

    def __init__(self, meta_paths, sc, category, in_size, hidden_dim, nheads,
                 dropout, num_nodes, num_classes, isBias, isAttn, isSemi):
        super(DMGI, self).__init__()
        self.category = category
        # self.layers = nn.ModuleList()
        self.hidden_dim = hidden_dim
        self.meta_paths = meta_paths
        self.nheads = nheads
        self.isAttn = isAttn
        self.isSemi = isSemi
        self.sc = sc
        self.gcn = nn.ModuleList([dglnn.GraphConv(in_feats=in_size,
                                                  out_feats=hidden_dim,
                                                  activation=nn.ReLU(),
                                                  bias=isBias,
                                                  allow_zero_in_degree=True) for _ in range(len(meta_paths))])

        self.disc = Discriminator(hidden_dim)
        self.readout = AvgReadout()
        self.readout_act_func = nn.Sigmoid()
        self.dropout = dropout
        self.num_nodes = num_nodes
        # num_head = 1
        self.H = nn.Parameter(torch.FloatTensor(1, num_nodes, hidden_dim))

        self.logistic = LogReg(hidden_dim, num_classes)

        if self.isAttn:
            self.attn = nn.ModuleList(Attention(hid_units=hidden_dim,
                                                num_mps=len(meta_paths),
                                                num_ndoes=num_nodes) for _ in range(nheads))
            # self.attn = Attention(hid_units=hid_unit, num_mps=len(meta_paths), num_ndoes=num_nodes)

        self.init_weight()
        print("category:{}, category's classes:{}, isBias:{},"
              " isAttn:{}, isSemi:{}".format(category, num_classes,isBias,isAttn,isSemi))

    def init_weight(self):
        nn.init.xavier_normal_(self.H)
    # samp_bias1, samp_bias2  default  None

    def forward(self, hg, samp_bias1=None, samp_bias2=None):
        r"""

        The forward part of DMGI

        Parameters
        ----------
        hg : object
            the dgl heterogeneous graph

        Returns
        -------
        dict
            The predicted logit, and reg_loss and semi_loss

        The formula to compute the relation-type specific cross entropy :math:`\mathcal{L}^{(r)}`

        .. math::
          \begin{equation}
            \mathcal{L}^{(r)}=\sum_{v_{i} \in \mathcal{V}}^{n} \log \mathcal{D}\left(\mathbf{h}_{i}^{(r)}, \mathbf{s}^{(r)}\right)+\sum_{j=1}^{n} \log \left(1-\mathcal{D}\left(\tilde{\mathbf{h}}_{j}^{(r)}, \mathbf{s}^{(r)}\right)\right)
          \end{equation}

        where :math:`h_{i}^{(r)}`  is calculate by :math:`\mathbf{h}_{i}=\sigma\left(\sum_{j \in N(i)} \frac{1}{c_{i j}} \mathbf{x}_{j} \mathbf{W}\right)` ,
        :math:`s^{(r)}` is :math:`\mathbf{s}^{(r)}=\operatorname{Readout}\left(\mathbf{H}^{(r)}\right)=\sigma\left(\frac{1}{n} \sum_{i=1}^{n} \mathbf{h}_{i}^{(r)}\right)` .
        :math:`\mathcal{D}` is a discriminator that scores patchsummary representation pairs
        :math:`\tilde{\mathbf{h}}_{j}^{(r)}` corrupt the original attribute matrix by shuffling it.




        """
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

            feats[idx] = F.dropout(feats[idx], self.dropout, training=self.training)
            shuf_feats[idx] = F.dropout(shuf_feats[idx], self.dropout, training=self.training)

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
            r"""
                .. math::
                  \begin{equation}
                    \mathbf{h}_{i}=\mathcal{Q}\left(\left\{\mathbf{h}^{(r)} \mid r \in \mathcal{R}\right\}\right)=\sum_{r \in \mathcal{R}} a_{i}^{(r)} \mathbf{h}^{(r)}
                  \end{equation}

                where :math:`a_{i}^{(r)}` denotes the importance of relationr in generating the final embedding of node videfined as:

                .. math::
                  \begin{equation}
                    a_{i}^{(r)}=\frac{\exp \left(\mathbf{q}^{(r)} \cdot \mathbf{h}_{i}^{(r)}\right)}{\sum_{r^{\prime} \in \mathcal{R}} \exp \left(\mathbf{q}^{\left(r^{\prime}\right)} \cdot \mathbf{h}_{i}^{r^{\prime}}\right)}
                  \end{equation}
                """

            h_1_all_lst = [];h_2_all_lst = [];c_all_lst = []
            for h_idx in range(self.nheads):
                h_1_all_, h_2_all_, c_all_ = self.attn[h_idx](h_1_all, h_2_all, c_all)
                h_1_all_lst.append(h_1_all_);h_2_all_lst.append(h_2_all_); c_all_lst.append(c_all_)

            h_1_all = torch.mean(torch.cat(h_1_all_lst, 0), 0).unsqueeze(0)
            h_2_all = torch.mean(torch.cat(h_2_all_lst, 0), 0).unsqueeze(0)

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
            r"""
            Extension to Semi-Supervised Learning

            .. math::
              \begin{equation}
                \ell_{\text {sup }}=-\frac{1}{\left|\mathcal{Y}_{L}\right|} \sum_{l \in \mathcal{Y}_{L}} \sum_{i=1}^{c} Y_{l i} \ln \hat{Y}_{l i}
              \end{equation}

            Where :math:`mathcal{Y}_{L}` is the set of node indices with labels
            """
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


'''In the experiments, some relation type is more beneficial for a 
certain downstream task than others. Therefore, we can adopt the 
attention mechanism'''
class Attention(nn.Module):
    def __init__(self, hidden_dim, num_mps, num_ndoes):
        super(Attention, self).__init__()
        self.num_mps = num_mps
        self.hidden_dim = hidden_dim
        self.num_nodes = num_ndoes
        self.A = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(num_mps)])
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

    r"""
    The discriminator

    .. math::
      \begin{equation}
        \mathcal{D}\left(\mathbf{h}_{i}^{(r)}, \mathbf{s}^{(r)}\right)=\sigma\left(\mathbf{h}_{i}^{(r) T} \mathbf{M}^{(r)} \mathbf{s}^{(r)}\right)
      \end{equation}

    where :math:`M^{(r)}` is a trainable scoring matrix.
    """

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

    r"""
    Considering the efficiency of the method, we simply employ average pooling, computing the average of the set of embedding matrices

    .. math::
      \begin{equation}
        \mathbf{H}=\mathcal{Q}\left(\left\{\mathbf{H}^{(r)} \mid r \in \mathcal{R}\right\}\right)=\frac{1}{|\mathcal{R}|} \sum_{r \in \mathcal{R}} \mathbf{H}^{(r)}
      \end{equation}
    """

    def __init__(self):
        super(AvgReadout, self).__init__()
    def forward(self, seq):
        return torch.mean(seq, 0)

'''logreg'''
class LogReg(nn.Module):
    r"""
    Parameters
    ----------
    ft_in : int
        Size of hid_units
    nb_class : int
        The number of category's types
    """
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



