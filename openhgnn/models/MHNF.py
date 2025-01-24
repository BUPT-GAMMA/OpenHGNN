import dgl
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv, EdgeWeightNorm
from ..utils import transform_relation_graph_list
from . import BaseModel, register_model


@register_model('MHNF')
class MHNF(BaseModel):
    r"""
        MHNF from paper `Multi-hop Heterogeneous Neighborhood information Fusion graph representation learning
        <https://arxiv.org/pdf/2106.09289.pdf>`__.

        Given a heterogeneous graph :math:`G` and its edge relation type set :math:`\mathcal{R}`.Then we can extract l-hops hybrid adjacency matrix list
        in HMAE model. The hybrid adjacency matrix list can be used in HLHIA model to generate l-hops representations. Then HSAF
        model use attention mechanism to aggregate l-hops representations and because of multi-channel conv, the
        HSAF model also  aggregates different channels l-hops representations to generate a final representation.
        You can see detail operation in correspond model.

        Parameters
        ----------
        num_edge_type : int
            Number of relations.
        num_channels : int
            Number of conv channels.
        in_dim : int
            The dimension of input feature.
        hidden_dim : int
            The dimension of hidden layer.
        num_class : int
            Number of classification type.
        num_layers : int
            Length of hybrid metapath.
        category : string
            Type of predicted nodes.
        norm : bool
            If True, the adjacency matrix will be normalized.
        identity : bool
            If True, the identity matrix will be added to relation matrix set.

    """
    @classmethod
    def build_model_from_args(cls, args, hg):
        if args.identity == True:
            num_edge_type = len(hg.canonical_etypes) + 1
        else:
            num_edge_type = len(hg.canonical_etypes)
        # add self-loop edge
        return cls(num_edge_type=num_edge_type, num_channels=args.num_channels,
                    in_dim=args.hidden_dim, hidden_dim=args.hidden_dim, num_class=args.out_dim,
                    num_layers=args.num_layers, category=args.category, norm=args.norm_emd_flag, identity=args.identity)

    def __init__(self, num_edge_type, num_channels, in_dim, hidden_dim, num_class, num_layers, category, norm, identity):
        super(MHNF, self).__init__()
        self.num_edge_type = num_edge_type
        self.num_channels = num_channels
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_class = num_class
        self.num_layers = num_layers
        self.is_norm = norm
        self.category = category
        self.identity = identity

        self.HSAF = HSAF(num_edge_type, self.num_channels, self.num_layers, self.in_dim, self.hidden_dim)
        self.linear = nn.Linear(self.hidden_dim, self.num_class)
        self.category_idx = None
        self.A = None
        self.h = None

    def forward(self, hg, h=None):
        with hg.local_scope():
            #Ws = []
            hg.ndata['h'] = h
            # * =============== Extract edges in original graph ================
            if self.category_idx is None:
                self.A, h, self.category_idx = transform_relation_graph_list(hg, category=self.category,
                                                        identity=self.identity)
            else:
                g = dgl.to_homogeneous(hg, ndata='h')
                h = g.ndata['h']
            # g = dgl.to_homogeneous(hg, ndata='h')
            #X_ = self.gcn(g, self.h)
            A = self.A
            final_representation = self.HSAF(A, h)
            y = self.linear(final_representation)
            return {self.category: y[self.category_idx]}


class HSAF(nn.Module):
    r'''
        HSAF: Hierarchical Semantic Attention Fusion

        The HSAF model use two level attention mechanism to generate final representation

        * Hop-level attention

          .. math::
              \alpha_{i, l}^{\Phi_{p}}=\sigma\left[\delta^{\Phi_{p}} \tanh \left(W^{\Phi_{p}} Z_{i, l}^{\Phi_{p}}\right)\right]

          In which, :math:`\alpha_{i, l}^{\Phi_{p}}` is the importance of the information :math:`\left(Z_{i, l}^{\Phi_{p}}\right)`
          of the l-th-hop neighbors of node i under the path :math:`\Phi_{p}`, and :math:`\delta^{\Phi_{p}}` represents the learnable matrix.

          Then normalize :math:`\alpha_{i, l}^{\Phi_{p}}`

          .. math::
              \beta_{i, l}^{\Phi_{p}}=\frac{\exp \left(\alpha_{i, l}^{\Phi_{p}}\right)}{\sum_{j=1}^{L} \exp \left(\alpha_{i, j}^{\Phi_{p}}\right)}

          Finally, we get hop-level attention representation in one hybrid metapath.

          .. math::
              Z_{i}^{\Phi_{p}}=\sum_{l=1}^{L} \beta_{l}^{\Phi_{p}} Z_{l}^{\Phi_{p}}

        * Channel-level attention

          It also can be seen as multi-head attention mechanism.

          .. math::
              \alpha_{i, \Phi_{p}}=\sigma\left[\delta \tanh \left(W Z_{i}^{\Phi_{p}}\right)\right.

          Then normalize :math:`\alpha_{i, \Phi_{p}}`

          .. math::
              \beta_{i, \Phi_{p}}=\frac{\exp \left(\alpha_{i, \Phi_{p}}\right)}{\sum_{p^{\prime} \in P} \exp \left(\alpha_{\Phi_{p^{\prime}}}\right)}

          Finally, we get final representation of every nodes.

          .. math::
              Z_{i}=\sum_{p \in P} \beta_{i, \Phi_{p}} Z_{i, \Phi_{p}}

    '''
    def __init__(self, num_edge_type, num_channels, num_layers, in_dim, hidden_dim):
        super(HSAF, self).__init__()
        self.num_channels = num_channels
        self.num_layers = num_layers
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.HLHIA_layer = HLHIA(num_edge_type, self.num_channels, self.num_layers, self.in_dim, self.hidden_dim)
        # * =============== channel attention operation ================
        self.channel_attention = nn.Sequential(
            nn.Linear(self.hidden_dim, 1),
            nn.Tanh(),
            nn.Linear(1, 1, bias=False),
            nn.ReLU()
        )
        # * =============== layers attention operation ================
        self.layers_attention = nn.ModuleList()
        for i in range(num_channels):
            self.layers_attention.append(nn.Sequential(
                nn.Linear(self.hidden_dim, 1),
                nn.Tanh(),
                nn.Linear(1, 1, bias=False),
                nn.ReLU()
            ))

    def forward(self, A, h):
        attention_list = self.HLHIA_layer(A, h)
        channel_attention_list = []
        for i in range(self.num_channels):
            layer_level_feature_list = attention_list[i]
            layer_attention = self.layers_attention[i]
            for j in range(self.num_layers + 1):
                layer_level_feature = layer_level_feature_list[j]
                if j == 0:
                    layer_level_alpha = layer_attention(layer_level_feature)
                else:
                    layer_level_alpha = th.cat((layer_level_alpha, layer_attention(layer_level_feature)), dim=-1)
            layer_level_beta = th.softmax(layer_level_alpha, dim=-1)
            channel_attention_list.append(
                th.bmm(th.stack(layer_level_feature_list, dim=-1), layer_level_beta.unsqueeze(-1)).squeeze(-1))

        for i in range(self.num_channels):
            channel_level_feature = channel_attention_list[i]
            if i == 0:
                channel_level_alpha = self.channel_attention(channel_level_feature)
            else:
                channel_level_alpha = th.cat((channel_level_alpha, self.channel_attention(channel_level_feature)),
                                             dim=-1)
        channel_level_beta = th.softmax(channel_level_alpha, dim=-1)
        channel_attention = th.bmm(th.stack(channel_attention_list, dim=-1), channel_level_beta.unsqueeze(-1)).squeeze(
            -1)
        return channel_attention


class HLHIA(nn.Module):
    r"""
        HLHIA: The Hop-Level Heterogeneous Information Aggregation

        The l-hop representation :math:`Z_{l}` is generated by the original node feature through a graph conv

        .. math::
           Z_{l}^{\Phi_{p}} = \sigma\left[\left(D_{(l)}^{\Phi_{p}}\right)^{-1} A_{(l)}^{\Phi_{p}} h W^{\Phi_{p}}\right]

        where :math:`\Phi_{p}` is the hybrid l-hop metapath and `\mathcal{h}` is the original node feature.

    """
    def __init__(self, num_edge_type, num_channels, num_layers, in_dim, hidden_dim):
        super(HLHIA, self).__init__()
        self.num_channels = num_channels
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(HMAELayer(num_edge_type, num_channels, first=True))
            else:
                layers.append(HMAELayer(num_edge_type, num_channels, first=False))
        self.layers = nn.ModuleList(layers)
        self.gcn_list = nn.ModuleList()
        for i in range(num_channels):
            self.gcn_list.append(GraphConv(in_feats=self.in_dim, out_feats=hidden_dim, norm='none', activation=F.relu))
        self.norm = EdgeWeightNorm(norm='right')

    def forward(self, A, h):
        layer_list = []
        for i in range(len(self.layers)):
            if i == 0:
                H, W, first_adj = self.layers[i](A)
                layer_list.append(first_adj)
                layer_list.append(H)
            else:
                H, W, first_adj = self.layers[i](A, H)
                layer_list.append(H)
        # * =============== GCN Encoder ================
        channel_attention_list = []
        for i in range(self.num_channels):
            gcn = self.gcn_list[i]
            layer_attention_list = []
            for j in range(len(layer_list)):
                layer = layer_list[j][i]
                layer = dgl.remove_self_loop(layer)
                edge_weight = layer.edata['w_sum']
                layer = dgl.add_self_loop(layer)
                edge_weight = th.cat((edge_weight, th.full((layer.number_of_nodes(),), 1, device=layer.device)))
                edge_weight = self.norm(layer, edge_weight)
                layer_attention_list.append(gcn(layer, h, edge_weight=edge_weight))
            channel_attention_list.append(layer_attention_list)
        return channel_attention_list


class HMAELayer(nn.Module):
    r"""
        HMAE: Hybrid Metapath Autonomous Extraction

        The method to generate l-hop hybrid adjacency matrix

        .. math::
            A_{(l)}=\Pi_{i=1}^{l} A_{i}
    """
    def __init__(self, in_channels, out_channels, first=True):
        super(HMAELayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.first = first
        self.norm = EdgeWeightNorm(norm='right')
        if self.first == True:
            self.conv1 = GTConv(in_channels, out_channels, softmax_flag=False)
            self.conv2 = GTConv(in_channels, out_channels, softmax_flag=False)
        else:
            self.conv1 = GTConv(in_channels, out_channels, softmax_flag=False)

    def softmax_norm(self, H):
        norm_H = []
        for i in range(len(H)):
            g = H[i]
            g.edata['w_sum'] = self.norm(g, th.exp(g.edata['w_sum'])) # normalize the hybrid relationship matrix
            norm_H.append(g)
        return norm_H

    def forward(self, A, H_=None):
        if self.first == True:
            result_A = self.softmax_norm(self.conv1(A))
            result_B = self.softmax_norm(self.conv2(A))
            W = [self.conv1.weight.detach(), self.conv2.weight.detach()]
        else:
            result_A = H_
            result_B = self.conv1(A)
            W = [self.conv1.weight.detach().detach()]
        H = []
        for i in range(len(result_A)):
            g = dgl.adj_product_graph(result_A[i], result_B[i], 'w_sum')
            H.append(g)
        return H, W, result_A


class GTConv(nn.Module):
    r"""
        We conv each sub adjacency matrix :math:`A_{R_{i}}` to a combination adjacency matrix :math:`A_{1}`:

        .. math::
            A_{1} = conv\left(A ; W_{c}\right)=\sum_{R_{i} \in R} w_{R_{i}} A_{R_{i}}

        where :math:`R_i \subseteq \mathcal{R}` and :math:`W_{c}` is the weight of each relation matrix
    """

    def __init__(self, in_channels, out_channels, softmax_flag=True):
        super(GTConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(th.Tensor(out_channels, in_channels))
        self.softmax_flag = softmax_flag
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight, std=0.01)

    def forward(self, A):
        if self.softmax_flag:
            Filter = F.softmax(self.weight, dim=1)
        else:
            Filter = self.weight
        num_channels = Filter.shape[0]
        results = []
        for i in range(num_channels):
            for j, g in enumerate(A):
                A[j].edata['w_sum'] = g.edata['w'] * Filter[i][j]
            sum_g = dgl.adj_sum_graph(A, 'w_sum')
            results.append(sum_g)
        return results
