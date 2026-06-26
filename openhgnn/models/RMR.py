from openhgnn.models import HAN, register_model
from openhgnn.models.base_model import BaseModel
import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv
from dgl.nn.pytorch.softmax import edge_softmax
import torch.nn.init as init

def mask(x, mask_rate=0.5):
    num_nodes = x.size(0)
    perm = torch.randperm(num_nodes, device=x.device)
    num_mask_nodes = int(mask_rate * num_nodes)
    mask_nodes = perm[: num_mask_nodes]
    return mask_nodes

def sce_loss(x, y, alpha=1):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
    return loss.mean()
class CustomGATLayer(nn.Module):
    def __init__(self, in_dim, dropout=0.0 , bias=True):
        super(CustomGATLayer, self).__init__()
        # self.fc = nn.Linear(in_dim, in_dim, bias=False)
        self.attn_l = nn.Parameter(torch.Tensor(1, in_dim))
        self.attn_r = nn.Parameter(torch.Tensor(1, in_dim))
        self.dropout = nn.Dropout(dropout)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(in_dim))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        # nn.init.xavier_normal_(self.fc.weight)
        nn.init.xavier_normal_(self.attn_l)
        nn.init.xavier_normal_(self.attn_r)
        init.zeros_(self.bias)

    def edge_attention(self, edges):
        score = (edges.src['z'] * self.attn_l).sum(dim=-1) + (edges.dst['z'] * self.attn_r).sum(dim=-1)
        return {'e': F.leaky_relu(score)}

    # def message_func(self, g, edges):
    #     alpha = edge_softmax(g, g.edata['e'])
    #     alpha = F.dropout(alpha, p=self.dropout, training=self.training)
    #     return {'z': edges.src['z']*alpha.unsqueeze(-1)}

    def message_func(self, edges):
        return {'z': edges.src['z'] * edges.data['alpha'].unsqueeze(-1)}

    # g 为根据边划分的子图
    def forward(self, x, g):
        with g.local_scope():
            h_src, h_dst = x
            # feat_src, feat_dst = x
            # h_src = self.fc(feat_src)
            # h_dst = self.fc(feat_dst) if feat_dst is not None else h_src
            g.srcdata['z'] = h_src
            g.dstdata['z'] = h_dst

            g.apply_edges(self.edge_attention)

            # 在这里应用 softmax，避免在 message_func 里使用 g
            g.edata['alpha'] = edge_softmax(g, g.edata['e'])
            g.edata['alpha'] = self.dropout(g.edata['alpha'])

            g.update_all(message_func=self.message_func, reduce_func=fn.sum("z", "h_N"))


            h_N = g.ndata["h_N"]
            h_N = h_N[g.canonical_etypes[0][2]]
            if self.bias is not None:
                h_N += self.bias

            return h_N


@register_model('RMR')
class RMR(BaseModel):
    @classmethod
    def build_model_from_args(cls, args, hg):
        return cls(
            g=hg,  # 输入的heterograph
            dataset_name=args.dataset_name,
            hidden_dim=args.hidden_dim,  # 隐藏特征维度
            feat_drop=args.feat,  # 特征dropout
            att_drop1=args.attr1,  # 第一个注意力dropout
            att_drop2=args.attr2,  # 第二个注意力dropout
            r1=args.r1,  # mask参数r1
            r2=args.r2,  # mask参数r2
            r3=args.r3  # mask参数r3
        )

    def __init__(self, dataset_name ,g, hidden_dim, feat_drop, att_drop1, att_drop2, r1, r2, r3):
        super(RMR, self).__init__()
        self.g = g
        self.hidden_dim = hidden_dim
        self.r1 = r1
        self.r2 = r2
        self.r3 = r3
        self.node_types = g.ntypes
        self.edge_types = g.etypes
        self.dataset_name = dataset_name

        self.fc = nn.ModuleDict({
            n_type: nn.Linear(g.nodes[n_type].data['x'].shape[1], hidden_dim,bias=True)
            for n_type in self.node_types
        })

        self.feat_drop = nn.Dropout(feat_drop)

        self.enc_mask_token = nn.Parameter(torch.zeros(1, hidden_dim))

        self.intra = nn.ModuleList([
            CustomGATLayer(hidden_dim, att_drop1)
            for _ in range(len(self.g.graph_data['schema_dict']))
        ])
        self.action = nn.ModuleList([
            nn.PReLU() for _ in range(len(self.g.graph_data['schema_dict']))
        ])
        self.act = nn.ModuleDict({
            n_type: nn.PReLU()
            for n_type in self.node_types
        })
        self.bn = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(len(self.g.graph_data['schema_dict']))
        ])
        self.schema_dict = {s:i for i,s in enumerate(self.g.graph_data['schema_dict'])}

        self.intra_mp = nn.ModuleList([
            CustomGATLayer(hidden_dim, att_drop2)
            for _ in range(len(self.g.graph_data['mp']))
        ])
        self.action_mp = nn.ModuleList([
            nn.PReLU() for _ in range(len(self.g.graph_data['mp']))
        ])
        self.bn_mp = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(len(self.g.graph_data['mp']))
        ])
        self.mp = {s: i for i, s in enumerate(self.g.graph_data['mp'])}
        self.reset_parameter()
    def reset_parameter(self):
        for fc in self.fc.values():
            nn.init.xavier_normal_(fc.weight, gain=1.414)

    def forward(self):
        h = {}
        # 为每类节点都基于单位矩阵生成特征矩阵
        for n_type in self.g.graph_data['use_nodes']:
            h[n_type] = self.act[n_type](
                self.feat_drop(
                    self.fc[n_type](self.g.nodes[n_type].data['x'])
                )
            )

        for n_type in self.mp:
            src, relation , dst = n_type
            x = h[src], h[dst]
            # 邻居节点的消息汇聚之后直接覆盖原节点的值之后返回总特征矩阵
            g_sub = self.g[n_type]
            # embed1 = self.intra_mp[self.mp[n_type]](x, self.g.edges(etype=n_type))
            embed1 = self.intra_mp[self.mp[n_type]](x, g_sub)
            embed1 = self.bn_mp[self.mp[n_type]](embed1)
            h[dst] = self.action_mp[self.mp[n_type]](embed1)

        if self.dataset_name == 'acm4RMR':
            # Reconstruct
            main_node = self.g.graph_data['main_node']
            mask_nodes = mask(h[main_node], mask_rate=self.r1)
            main_h = h[main_node].clone()
            main_h[mask_nodes] = 0.0
            main_h[mask_nodes] += self.enc_mask_token

            sc = ('a', 'ap', 'p')
            # sc = ('actor', 'movie')
            src, _, dst = sc
            x = h[src], main_h
            g_sub = self.g[sc]
            embed1 = self.intra[self.schema_dict[sc]](x, g_sub)
            embed1 = self.bn[self.schema_dict[sc]](embed1)
            embed1 = self.action[self.schema_dict[sc]](embed1)

            sc = ('s', 'sp', 'p')
            # sc = ('director', 'movie')
            src, _, dst = sc
            x = h[src], h[dst]
            g_sub = self.g[sc]
            embed2 = self.intra[self.schema_dict[sc]](x, g_sub)
            embed2 = self.bn[self.schema_dict[sc]](embed2)
            embed2 = self.action[self.schema_dict[sc]](embed2)
            loss1 = sce_loss(embed1[mask_nodes], embed2[mask_nodes].detach())
            #############################################################

            main_node = self.g.graph_data['main_node']
            mask_nodes = mask(h[main_node], mask_rate=self.r2)
            main_h = h[main_node].clone()
            main_h[mask_nodes] = 0.0
            main_h[mask_nodes] += self.enc_mask_token

            sc = ('s', 'sp', 'p')
            # sc = ('director', 'movie')
            src, _, dst = sc
            x = h[src], main_h
            g_sub = self.g[sc]
            # edge_index, edge_mask = dropout_edge(data[sc].edge_index, 0.05)
            embed1 = self.intra[self.schema_dict[sc]](x, g_sub)
            embed1 = self.bn[self.schema_dict[sc]](embed1)
            embed1 = self.action[self.schema_dict[sc]](embed1)

            sc = ('a', 'ap','p')
            # sc = ('actor', 'movie')
            src, _, dst = sc
            x = h[src], h[dst]
            g_sub = self.g[sc]
            embed2 = self.intra[self.schema_dict[sc]](x, g_sub)
            embed2 = self.bn[self.schema_dict[sc]](embed2)
            embed2 = self.action[self.schema_dict[sc]](embed2)
            loss2 = sce_loss(embed1[mask_nodes], embed2[mask_nodes].detach())
            return loss1 + loss2
        elif self.dataset_name == 'imdb4RMR':
            ########################################################################
            # Reconstruct
            main_node = self.g.graph_data['main_node']
            mask_node = mask(h[main_node], mask_rate=self.r1)
            main_h = h[main_node].clone()
            main_h[mask_node] = 0.0
            main_h[mask_node] += self.enc_mask_token

            # sc = ('a', 'p')
            sc = ('actor', 'am','movie')
            src, _,dst = sc
            x = h[src], main_h
            g_sub = self.g[sc]
            # edge_index, edge_mask = dropout_edge(data[sc].edge_index, 0.1)
            embed1 = self.intra[self.schema_dict[sc]](x, g_sub)
            embed1 = self.bn[self.schema_dict[sc]](embed1)
            embed1 = self.action[self.schema_dict[sc]](embed1)

            # sc = ('s', 'p')
            sc = ('director', 'dm','movie')
            src, _, dst = sc
            x = h[src], h[dst]
            g_sub = self.g[sc]
            embed2 = self.intra[self.schema_dict[sc]](x, g_sub)
            embed2 = self.bn[self.schema_dict[sc]](embed2)
            embed2 = self.action[self.schema_dict[sc]](embed2)
            loss1 = sce_loss(embed1[mask_node], embed2[mask_node].detach())
            ###########################################################################

            mask_node = mask(h[main_node], mask_rate=self.r2)
            main_h = h[main_node].clone()
            main_h[mask_node] = 0.0
            main_h[mask_node] += self.enc_mask_token

            # sc = ('s', 'p')
            sc = ('director', 'dm','movie')
            src, _, dst = sc
            x = h[src], main_h
            g_sub = self.g[sc]
            # edge_index, edge_mask = dropout_edge(data[sc].edge_index, 0.05)
            embed1 = self.intra[self.schema_dict[sc]](x, g_sub)
            embed1 = self.bn[self.schema_dict[sc]](embed1)
            embed1 = self.action[self.schema_dict[sc]](embed1)

            # sc = ('a', 'p')
            sc = ('actor', 'am','movie')
            src, _, dst = sc
            x = h[src], h[dst]
            g_sub = self.g[sc]
            embed2 = self.intra[self.schema_dict[sc]](x, g_sub)
            embed2 = self.bn[self.schema_dict[sc]](embed2)
            embed2 = self.action[self.schema_dict[sc]](embed2)
            loss2 = sce_loss(embed1[mask_node], embed2[mask_node].detach())
            return loss1 + loss2

        elif self.dataset_name == 'aminer4RMR':
            ########################################################################
            # Aminer
            # 接下来是 将main_node随机mask后的相关特征重建

            # mask_node代表需要被mask的节点index
            main_node = self.g.graph_data['main_node']
            mask_node = mask(h[main_node], mask_rate=self.r1)
            # 将数据中的main_node特征矩阵clone出来，不共享内存，但是会之间在原tensor上累积梯度
            main_h = h[main_node].clone()
            # 将需要被mask的节点对应特征值全部置为0
            main_h[mask_node] = 0.0

            h1 = 0
            # 遍历关系模式字典1 schema_dict1={(A, P)=None,(R, P)=None,}
            # 对应论文公式里的eq5
            for n_type in self.g.graph_data['schema_dict1']:
                # 这里dst为main_node
                src, _, dst = n_type
                x = h[src], h[dst]
                # embed1返回的是P的邻居节点传递消息给P的聚合特征矩阵
                g_sub = self.g[n_type]
                embed1 = self.intra[self.schema_dict[n_type]](x, g_sub)
                embed1 = self.bn[self.schema_dict[n_type]](embed1)
                # h1代表利用邻居节点特征重构的main_node特征矩阵 在算损失的时候将作为ans对照
                h1 += self.action[self.schema_dict[n_type]](embed1)

            # 对应论文中公式eq3
            sc = ('C', 'CP', 'P')
            src, _, dst = sc
            x = h[src], main_h
            # 利用关系['C'->'P'] 仅使用'C'节点的相关特征消息重构'P'节点消息
            # 对应论文中公式eq4
            g_sub = self.g[sc]
            embed2 = self.intra[self.schema_dict[sc]](x, g_sub)
            embed2 = self.bn[self.schema_dict[sc]](embed2)
            embed2 = self.action[self.schema_dict[sc]](embed2)

            # 对应仅仅使用关系['C'->'P']重构的P节点特征 和 利用除了关系['C'->'P']之后其他全部关系重构的P节点特征 之间的损失
            # 对应论文公式eq6
            loss1 = sce_loss(embed2[mask_node], h1[mask_node].detach())
            ##########################################################################

            mask_node = mask(h[main_node], mask_rate=self.r2)
            main_h = h[main_node].clone()
            main_h[mask_node] = 0.0
            h1 = 0
            for n_type in self.g.graph_data['schema_dict2']:
                src, _, dst = n_type
                x = h[src], h[dst]
                g_sub = self.g[n_type]
                embed1 = self.intra[self.schema_dict[n_type]](x, g_sub)
                embed1 = self.bn[self.schema_dict[n_type]](embed1)
                h1 += self.action[self.schema_dict[n_type]](embed1)
            sc = ('R', 'RP', 'P')
            src, _, dst = sc
            x = h[src], main_h
            g_sub = self.g[sc]
            embed2 = self.intra[self.schema_dict[sc]](x, g_sub)
            embed2 = self.bn[self.schema_dict[sc]](embed2)
            embed2 = self.action[self.schema_dict[sc]](embed2)
            loss2 = sce_loss(embed2[mask_node], h1[mask_node].detach())
            ##########################################################################
            mask_node = mask(h[main_node], mask_rate=self.r3)
            main_h = h[main_node].clone()
            main_h[mask_node] = 0.0
            h1 = 0
            for n_type in self.g.graph_data['schema_dict3']:
                src, _, dst = n_type
                x = h[src], h[dst]
                g_sub = self.g[n_type]
                embed1 = self.intra[self.schema_dict[n_type]](x, g_sub)
                embed1 = self.bn[self.schema_dict[n_type]](embed1)
                h1 += self.action[self.schema_dict[n_type]](embed1)
            sc = ('A', 'AP', 'P')
            src, _, dst = sc
            x = h[src], main_h
            g_sub = self.g[sc]
            embed2 = self.intra[self.schema_dict[sc]](x, g_sub)
            embed2 = self.bn[self.schema_dict[sc]](embed2)
            embed2 = self.action[self.schema_dict[sc]](embed2)
            loss3 = sce_loss(embed2[mask_node], h1[mask_node].detach())

            # 将不同关系子图下重构的损失率相加代表总损失率
            return loss1 + loss2 + loss3





    def get_embed(self):
        r"""
        Return the embedding of a model for further analysis.

        Returns
        -------
        numpy.array
        """
        h = {}
        for n_type in self.node_types:
            h[n_type] = self.act[n_type](
                self.feat_drop(
                    self.fc[n_type](self.g.nodes[n_type].data['x'])
                )
            )

        embed_sum = 0.0
        for sc, _ in self.g.graph_data['mp'].items():
            src, _, dst = sc
            x = h[src], h[dst]
            g_sub = self.g[sc]
            updated_feat = self.intra_mp[self.mp[sc]](x, g_sub)
            updated_feat = self.bn_mp[self.mp[sc]](updated_feat)
            h[dst] = self.action_mp[self.mp[sc]](updated_feat)


        for sc, _ in self.g.graph_data['schema_dict'].items():
            src, _, dst = sc
            x = h[src], h[dst]
            g_sub = self.g[sc]
            updated_feat = self.intra[self.schema_dict[sc]](x, g_sub)
            updated_feat = self.bn[self.schema_dict[sc]](updated_feat)
            embed_sum += self.action[self.schema_dict[sc]](updated_feat)
        return embed_sum.detach()

    def get_C(self, data):
        h = {}
        for n_type in data.use_nodes:
            h[n_type] = self.act[n_type](
                self.feat_drop(
                    self.fc[n_type](data[n_type].x)
                )
            )
        return h['C'].detach()




