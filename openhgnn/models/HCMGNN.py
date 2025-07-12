import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.ops import edge_softmax
from ..utils.HCMGNN_utils import *
from . import BaseModel, register_model


@register_model('HCMGNN')
class HCMGNN(BaseModel):
    @classmethod
    def build_model_from_args(cls, meta_paths, test_data, in_size, hidden_size, num_heads, dropout, etypes):
        # This method creates an instance of HCMGNN with the provided arguments.
        return cls(meta_paths=meta_paths, test_data=test_data, in_size=in_size,
                   hidden_size=hidden_size, num_heads=num_heads, dropout=dropout, etypes=etypes)

    def __init__(self, meta_paths, test_data, in_size, hidden_size, num_heads, dropout, etypes):
        super(HCMGNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.fc_g = nn.Linear(in_size['g'], hidden_size)
        self.fc_m = nn.Linear(in_size['m'], hidden_size)
        self.fc_d = nn.Linear(in_size['d'], hidden_size)
        self.predict = nn.Sequential(
            nn.Linear(self.hidden_size * self.num_heads * 3, self.hidden_size * self.num_heads),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size * self.num_heads, self.hidden_size * self.num_heads // 4),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size * self.num_heads // 4, self.hidden_size),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, 1)
        )
        r_vec = nn.Parameter(torch.empty(size=(3, self.hidden_size // 2, 2)))
        self.layers1 = HCMGNN_Layer(meta_paths, test_data, hidden_size, r_vec, num_heads, dropout, etypes,
                                    name=['g', 'm', 'd'])
        self.predict.apply(self.weights_init)
        nn.init.xavier_normal_(self.fc_g.weight, gain=1.414)
        nn.init.xavier_normal_(self.fc_m.weight, gain=1.414)
        nn.init.xavier_normal_(self.fc_d.weight, gain=1.414)

        self._node_embeddings = None

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_normal_(m.weight, gain=1.414)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0.0)

    def get_embed_map(self, features, embed_features, data):
        stack_embedding = {'g': list(), 'm': list(), 'd': list()}
        for i in range(len(embed_features.keys())):
            for j in range(len(data)):
                node_id = int(data[j, i].item())
                if node_id in range(len(embed_features[list(embed_features.keys())[i]])):
                    stack_embedding[list(stack_embedding.keys())[i]].append(
                        embed_features[list(embed_features.keys())[i]][node_id])
                else:
                    id = int(data[j][i])
                    stack_embedding[list(stack_embedding.keys())[i]].append(
                        torch.hstack([features[list(embed_features.keys())[i]][id]] * 8))
            stack_embedding[list(stack_embedding.keys())[i]] = torch.stack(
                stack_embedding[list(stack_embedding.keys())[i]], dim=0)
        embedding_concat = torch.cat((stack_embedding['g'], stack_embedding['m'], stack_embedding['d']),
                                     dim=1)
        return embedding_concat

    def forward(self, g, inputs, data):
        h_trans = {}
        h_trans['g'] = self.fc_g(inputs['g']).view(-1, self.hidden_size)
        h_trans['m'] = self.fc_m(inputs['m']).view(-1, self.hidden_size)
        h_trans['d'] = self.fc_d(inputs['d']).view(-1, self.hidden_size)
        h_trans_embed = self.layers1(g, h_trans)
        self._node_embeddings = h_trans_embed
        h_concat = self.get_embed_map(h_trans, h_trans_embed, data)
        predict_score = torch.sigmoid(self.predict(h_concat))
        return predict_score

    def extra_loss(self):
        """
        Returns any additional loss components for the model.
        For HCMGNN, no explicit extra loss is defined beyond the main prediction loss.
        """
        return torch.tensor(0.0, device=self.fc_g.weight.device)  # Return 0.0 if no extra loss

    def get_emb(self):
        """
        Return the learned node embeddings of the model.
        Returns:
            dict[str, torch.Tensor]: A dictionary of learned embeddings for each node type.
        """
        if self._node_embeddings is None:
            # You might want to run a dummy forward pass or raise an error
            # if embeddings are requested before a forward pass has occurred.
            print("Warning: Node embeddings requested before a forward pass. Returning None.")
            return None
        return self._node_embeddings


class MessageAggregator(nn.Module):
    def __init__(self, num_heads, hidden_size, attn_drop, alpha, name):
        super(MessageAggregator, self).__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.leaky_relu = nn.LeakyReLU(alpha)
        self.softmax = edge_softmax
        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x: x
        self.attn1 = nn.Linear(self.hidden_size, self.num_heads, bias=False)
        nn.init.xavier_normal_(self.attn1.weight, gain=1.414)
        self.attn2 = nn.Parameter(torch.empty(size=(1, self.num_heads, self.hidden_size)))
        nn.init.xavier_normal_(self.attn2.data, gain=1.414)
        self.name = name

    def forward(self, nodes, metapath_instances, metapath_embedding, features):
        h_ = []
        for i in range(len(nodes)):
            index = metapath_instances[metapath_instances[self.name] == nodes[i]].index.tolist()
            if index != []:
                node_metapath_embedding = metapath_embedding[index]  # (E,64)
                node_metapath_embedding = torch.cat([node_metapath_embedding] * self.num_heads, dim=1)  # (E,64*8)
                node_metapath_embedding = node_metapath_embedding.unsqueeze(dim=0)  # ( 1, E ,64*8)
                eft = node_metapath_embedding.permute(1, 0, 2).view(-1, self.num_heads, self.hidden_size)  # (E, 8,64)
                device = metapath_embedding.device
                node_embedding = torch.vstack([features[i]] * len(index)).to(device)
                a1 = self.attn1(node_embedding)
                a2 = (eft * self.attn2).sum(dim=-1)
                a = (a1 + a2).unsqueeze(dim=-1)
                a = self.leaky_relu(a)
                attention = F.softmax(a, dim=0) # F 通常是 torch.nn.functional 的别名
                attention = self.attn_drop(attention)
                h = F.elu((attention * eft).sum(dim=0)).view(-1, self.hidden_size * self.num_heads)
                h_.append(h[0])
            else:
                device = features[i].device if hasattr(features[i], 'device') else 'cpu'
                node_embedding = torch.zeros(self.hidden_size * self.num_heads, device=device)
                h_.append(node_embedding)
        return torch.stack(h_, dim=0)


class Subgraph_Fusion(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(Subgraph_Fusion, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_normal_(m.weight, gain=1.414)

    def forward(self, z):
        w = self.project(z).mean(0)
        beta_ = torch.softmax(w, dim=0)
        beta = beta_.expand((z.shape[0],) + beta_.shape)
        return (beta * z).sum(1), beta_


class SemanticEncoder(nn.Module):
    def __init__(self, layer_num_heads, hidden_size, r_vec, etypes):
        super(SemanticEncoder, self).__init__()
        self.num_heads = layer_num_heads
        self.hidden_size = hidden_size
        self.r_vec = r_vec
        self.etypes = etypes

    def forward(self, edata):
        edata = edata.reshape(edata.shape[0], edata.shape[1], edata.shape[2] // 2, 2)
        final_r_vec = torch.zeros([edata.shape[1], self.hidden_size // 2, 2], device=edata.device)
        r_vec = F.normalize(self.r_vec, p=2, dim=2)
        r_vec = torch.stack((r_vec, r_vec), dim=1)
        r_vec[:, 1, :, 1] = -r_vec[:, 1, :, 1]
        r_vec = r_vec.reshape(self.r_vec.shape[0] * 2, self.r_vec.shape[1], 2)
        final_r_vec[-1, :, 0] = 1
        for i in range(final_r_vec.shape[0] - 2, -1, -1):
            if self.etypes[i] is not None:
                final_r_vec[i, :, 0] = final_r_vec[i + 1, :, 0].clone() * r_vec[self.etypes[i], :, 0] - \
                                       final_r_vec[i + 1, :, 1].clone() * r_vec[self.etypes[i], :, 1]
                final_r_vec[i, :, 1] = final_r_vec[i + 1, :, 0].clone() * r_vec[self.etypes[i], :, 1] + \
                                       final_r_vec[i + 1, :, 1].clone() * r_vec[self.etypes[i], :, 0]
            else:
                final_r_vec[i, :, 0] = final_r_vec[i + 1, :, 0].clone()
                final_r_vec[i, :, 1] = final_r_vec[i + 1, :, 1].clone()
        for i in range(edata.shape[1] - 1):
            temp1 = edata[:, i, :, 0].clone() * final_r_vec[i, :, 0] - \
                    edata[:, i, :, 1].clone() * final_r_vec[i, :, 1]
            temp2 = edata[:, i, :, 0].clone() * final_r_vec[i, :, 1] + \
                    edata[:, i, :, 1].clone() * final_r_vec[i, :, 0]
            edata[:, i, :, 0] = temp1
            edata[:, i, :, 1] = temp2
        edata = edata.reshape(edata.shape[0], edata.shape[1], -1)
        metapath_embedding = torch.mean(edata, dim=1)
        return metapath_embedding


class HCMGNN_Layer(nn.Module):
    def __init__(self, meta_paths, test_data, hidden_size, r_vec, layer_num_heads, dropout, etypes, name):
        super(HCMGNN_Layer, self).__init__()
        self.num_heads = layer_num_heads
        self.meta_paths = list(tuple(meta_path) for meta_path in meta_paths)
        self._cached_graph = None
        self._cached_coalesced_graph = {}
        self.r_vec = r_vec
        self.etypes = etypes
        self.message_aggregator_layer = nn.ModuleList()
        self.semantic_encoder_layer = nn.ModuleList()
        self.hidden_size = hidden_size
        self.test_data = test_data
        for i in range(len(meta_paths)):
            self.semantic_encoder_layer.append(
                SemanticEncoder(self.num_heads, self.hidden_size, self.r_vec, self.etypes[i]))
        for i in name:
            self.message_aggregator_layer.append(
                MessageAggregator(self.num_heads, self.hidden_size, attn_drop=dropout, alpha=0.01, name=i))
        self.subgraph_fusion = Subgraph_Fusion(in_size=self.hidden_size * self.num_heads)
        self.separate_metapath_subgraph = Separate_subgraph()
        self.exclude_test = Prevent_leakage(self.test_data)

    def stack_embedding(self, embeddings):
        subgraph_num_nodes = [embeddings[i].size()[0] for i in range(len(embeddings))]
        if subgraph_num_nodes.count(subgraph_num_nodes[0]) == len(subgraph_num_nodes):
            embeddings = torch.stack(embeddings, dim=1)
        else:
            for i in range(0, len(embeddings)):
                index = max(subgraph_num_nodes) - subgraph_num_nodes[i]
                if index != 0:
                    device = embeddings[i].device
                    h_ = torch.zeros(index, self.hidden_size * self.num_heads, device=device)
                    embeddings[i] = torch.cat((embeddings[i], h_), dim=0)
            embeddings = torch.stack(embeddings, dim=1)
        return embeddings

    def generate_metapath_instances(self, g, meta_path):
        edges = [g.edges(etype=f"{meta_path[j]}_{meta_path[j + 1]}") for j in range(len(meta_path) - 1)]
        edges = [[edges[i][j].tolist() for j in range(len(edges[i]))] for i in range(len(edges))]
        df_0 = pd.DataFrame(edges[0], index=list(meta_path)[:2]).T
        df_1 = pd.DataFrame(edges[1], index=list(meta_path)[-2:]).T
        metapath_instances = pd.merge(df_0, df_1, how='inner')
        filt_metapath_instances = metapath_instances[['g', 'm', 'd']]
        filt_metapath_instances = self.exclude_test(filt_metapath_instances)
        metapath_instances = filt_metapath_instances[list(meta_path)]
        return metapath_instances

    def forward(self, g, h):
        if self._cached_graph is None or self._cached_graph is not g:
            self._cached_graph = g
            self._cached_coalesced_graph.clear()
            for meta_path in self.meta_paths:
                self._cached_coalesced_graph[meta_path] = self.separate_metapath_subgraph(g, meta_path)
        semantic_embeddings = {'g': [], 'm': [], 'd': []}
        nodes_embeddings = {}
        for i, meta_path in enumerate(self.meta_paths):
            edata_list = []
            new_g = self._cached_coalesced_graph[meta_path]
            metapath_instances = self.generate_metapath_instances(new_g, meta_path)
            for j in range(len(meta_path)):
                weight = h[list(meta_path)[j]]
                index_tensor = torch.tensor(metapath_instances.iloc[:, j].values, dtype=torch.long, device=weight.device)
                edata_list.append(
                    F.embedding(index_tensor, weight).unsqueeze(1))
            edata = torch.hstack(edata_list)
            metapathembedding = self.semantic_encoder_layer[i](edata)
            semantic_embeddings['g'].append(
                self.message_aggregator_layer[0](new_g.nodes('g').tolist(), metapath_instances, metapathembedding,
                                                 h['g']))
            semantic_embeddings['m'].append(
                self.message_aggregator_layer[1](new_g.nodes('m').tolist(), metapath_instances, metapathembedding,
                                                 h['m']))
            semantic_embeddings['d'].append(
                self.message_aggregator_layer[2](new_g.nodes('d').tolist(), metapath_instances, metapathembedding,
                                                 h['d']))

        for ntype in semantic_embeddings.keys():
            if ntype == 'g':
                semantic_embeddings[ntype] = self.stack_embedding(semantic_embeddings[ntype])
                nodes_embeddings[ntype], g_beta = self.subgraph_fusion(semantic_embeddings[ntype])
            elif ntype == 'm' and semantic_embeddings[ntype]:
                semantic_embeddings[ntype] = self.stack_embedding(semantic_embeddings[ntype])
                nodes_embeddings[ntype], m_beta = self.subgraph_fusion(semantic_embeddings[ntype])
            elif ntype == 'd' and semantic_embeddings[ntype]:
                semantic_embeddings[ntype] = self.stack_embedding(semantic_embeddings[ntype])
                nodes_embeddings[ntype], d_beta = self.subgraph_fusion(semantic_embeddings[ntype])
        return nodes_embeddings