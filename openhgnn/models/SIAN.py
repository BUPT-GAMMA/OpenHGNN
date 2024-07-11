import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from functools import reduce
from . import BaseModel,register_model

@register_model('SIAN')
class SIAN(BaseModel):

    @classmethod
    def build_model_from_args(cls, args, dataset):
        return SIAN(args.user_num, args.item_num, dataset.user_profile, dataset.item_profile, dataset.u_items, dataset.i_users, dataset.social_relation,
                    args.emb_size, args.profile_size, args.device)

    def __init__(self, user_num, item_num,
                 user_profile, item_profile,
                 user_items, item_users, social_rel,
                 emb_size=64, profile_size=64, device='cpu'):
        super(SIAN, self).__init__()

        self.emb_size = emb_size
        self.profile_size = profile_size
        self.device = device
        self.user_embedding = nn.Embedding(user_num, self.emb_size).to(self.device)
        self.item_embedding = nn.Embedding(item_num, self.emb_size).to(self.device)
        self.user_profile = user_profile.to(self.device)
        self.item_profile = item_profile.to(self.device)
        self.user_items = user_items
        self.item_users = item_users
        self.social_rel = social_rel

        self.fusion = Fusion(self.emb_size, self.profile_size).to(self.device)
        self.att = Attention(self.emb_size).to(self.device)
        # -------- Yelp data
        self.item_feat_agg = FeatureAgg(self.item_embedding, {'user': self.user_embedding},
                                        self.item_profile, {'user': self.user_profile},
                                        {'user': self.item_users}, ['user'], self.emb_size, self.device,
                                        self.fusion, self.att).to(self.device)
        self.user_feat_social_agg = FeatureAgg(self.user_embedding,
                                               {'item': self.item_embedding, 'user': self.user_embedding},
                                               self.user_profile,
                                               {'item': self.item_profile, 'user': self.user_profile},
                                               {'item': self.user_items, 'user': self.social_rel},
                                               ['item', 'user'], self.emb_size, self.device, self.fusion, self.att).to(self.device)

        self.social_inf_prop = InfluenceProp(self.social_rel, self.emb_size,
                                             self.user_embedding, self.user_profile,
                                             self.fusion, self.att).to(self.device)

        self.w_u1 = nn.Linear(self.emb_size, self.emb_size).to(self.device)
        self.w_u2 = nn.Linear(self.emb_size, self.emb_size).to(self.device)
        self.w_f1 = nn.Linear(self.emb_size, self.emb_size).to(self.device)
        self.w_f2 = nn.Linear(self.emb_size, self.emb_size).to(self.device)
        self.w_i1 = nn.Linear(self.emb_size, self.emb_size).to(self.device)
        self.w_i2 = nn.Linear(self.emb_size, self.emb_size).to(self.device)
        self.w_ui1 = nn.Linear(self.emb_size * 2, self.emb_size).to(self.device)
        self.w_ui2 = nn.Linear(self.emb_size, 16).to(self.device)
        self.w_ui3 = nn.Linear(16, 1).to(self.device)
        self.w_ufi1 = nn.Linear(self.emb_size * 3, self.emb_size).to(self.device)
        self.w_ufi2 = nn.Linear(self.emb_size, 16).to(self.device)
        self.w_ufi3 = nn.Linear(16, 1).to(self.device)
        self.bn1 = nn.BatchNorm1d(self.emb_size, momentum=0.5).to(self.device)
        self.bn2 = nn.BatchNorm1d(self.emb_size, momentum=0.5).to(self.device)
        self.bn3 = nn.BatchNorm1d(self.emb_size, momentum=0.5).to(self.device)
        self.bnf = nn.BatchNorm1d(self.emb_size, momentum=0.5).to(self.device)
        self.bn4 = nn.BatchNorm1d(16, momentum=0.5).to(self.device)
        self.item_fea_att_analysis = []
        self.user_fea_att_analysis = []
        self.inf_att_analysis = []

    def forward(self, users, items, act_users):

        i_embs, item_fea_att_list = self.item_feat_agg(items)
        u_embs, user_fea_att_list = self.user_feat_social_agg(users)
        u_inf, inf_att_list = self.social_inf_prop(users, u_embs, items, i_embs, act_users)

        x_u = F.relu(self.bn1(self.w_u1(u_embs)))
        x_u = F.dropout(x_u, training=self.training)
        x_u = self.w_u2(x_u)

        x_f = F.relu(self.bnf(self.w_f1(u_inf)))
        x_f = F.dropout(x_f, training=self.training)
        x_f = self.w_f2(x_f)

        x_i = F.relu(self.bn2(self.w_i1(i_embs)))
        x_i = F.dropout(x_i, training=self.training)
        x_i = self.w_i2(x_i)

        x_ufi = torch.cat((x_u, x_f, x_i), 1)
        x = F.relu(self.bn3(self.w_ufi1(x_ufi)))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.bn4(self.w_ufi2(x)))
        x = F.dropout(x, training=self.training)
        scores = torch.sigmoid(self.w_ufi3(x))

        return scores.squeeze()


class Attention(nn.Module):
    def __init__(self, embedding_dims):
        super(Attention, self).__init__()
        self.embed_dim = embedding_dims
        self.att1 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.att2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.att3 = nn.Linear(self.embed_dim, 1)

    def forward(self, node, neighs, num_neighs):
        node = node.repeat(num_neighs, 1)  # (num_neighs, dim)
        x = torch.cat((neighs, node), 1)  # (num_neighs, 2*dim)
        x = F.relu(self.att1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.att2(x))
        x = F.dropout(x, training=self.training)
        x = self.att3(x)

        att = F.softmax(x, dim=0)
        return att


class Fusion(nn.Module):
    def __init__(self, embedding_dims, profile_dim):
        super(Fusion, self).__init__()
        self.embed_dim = embedding_dims
        self.profile_dim = profile_dim
        self.w_1 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.w_2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.linear = nn.Linear(profile_dim, self.embed_dim)

    def forward(self, embedding, profile):

        return embedding


class InfluenceProp(nn.Module):
    def __init__(self, social_rel, emb_size, user_embs, user_profiles, fusion, att):
        super(InfluenceProp, self).__init__()
        self.social_rel = social_rel
        self.emb_size = emb_size
        self.user_embs = user_embs
        self.user_profiles = user_profiles
        self.linear = nn.Linear(2 * self.emb_size, self.emb_size)
        self.fusion = fusion
        self.att = att
        self.w_c1 = nn.Linear(self.emb_size * 2, self.emb_size)
        self.w_c2 = nn.Linear(self.emb_size, self.emb_size)

    def forward(self, users, u_embs, items, i_embs, act_users):

        batch_size = len(users)
        act_u_fusion = list(map(lambda x: self.fusion(self.user_embs.weight[x], self.user_profiles[x]), act_users))

        coupling_fea = list(map(lambda idx: torch.cat((act_u_fusion[idx],
                                                       i_embs[idx].repeat(len(act_u_fusion[idx]), 1)), 1),
                                range(batch_size)))

        coupling_fea = list(map(lambda idx: F.relu(self.w_c1(coupling_fea[idx])), range(batch_size)))
        coupling_fea = list(map(lambda idx: F.relu(self.w_c2(coupling_fea[idx])), range(batch_size)))
        attention_list = list(map(lambda idx: self.att(u_embs[idx], coupling_fea[idx], len(act_users[idx])),
                                  range(batch_size)))
        neigh_feature_matrix_coupling = torch.stack(
            [torch.mm(coupling_fea[idx].t(), attention_list[idx]) for idx in range(batch_size)]
        ).reshape(batch_size, self.emb_size).to(users.device)

        combined_feature = neigh_feature_matrix_coupling

        return combined_feature, attention_list


class FeatureAgg(nn.Module):
    def __init__(self, nodes_embeddings, neighbors_embeddings,
                 nodes_profiles, neighbors_profiles,
                 nodes_neighbors, neighbor_types,
                 emb_size, device, fusion, att):
        super(FeatureAgg, self).__init__()
        self.emb_size = emb_size
        self.device = device
        self.nodes_embeddings = nodes_embeddings.to(self.device)  # Embedding
        self.neighbors_embeddings_dict = neighbors_embeddings  # {type: Embedding}
        self.nodes_profiles = nodes_profiles.to(self.device)  # Embedding
        self.neighbors_profiles_dict = neighbors_profiles  # {type: Embedding}
        self.nodes_neighbors_dict = nodes_neighbors  # {type: {node: [neigh]}}
        self.neighbor_types_set = set(neighbor_types)  # [type]
        self.num_neigh_type = len(self.neighbor_types_set)
        self.linear_1 = nn.Linear(self.emb_size, self.emb_size)
        self.linear_2 = nn.Linear(self.emb_size, self.emb_size)
        self.linear = nn.Linear(self.emb_size * 2, self.emb_size)
        self.fusion = fusion
        self.att = att

        self.w_type_att = nn.Linear(self.emb_size * self.num_neigh_type, self.num_neigh_type, bias=False)

    def forward(self, nodes):
        batch_size = len(nodes)
        nodes_emb = self.nodes_embeddings.weight[nodes]  # (#node, dim)
        nodes = nodes.to(self.device)
        nodes_profile = self.nodes_profiles[nodes]
        nodes_fusion = self.fusion(nodes_emb, nodes_profile)

        neigh_type_agg = []  # (len(neigh_type_set), batch, dim)
        for idx, neigh_type in enumerate(self.neighbor_types_set):
            node_neighs = self.nodes_neighbors_dict[neigh_type]
            neighs = list(map(lambda y: list(y), map(lambda x: node_neighs[int(x)], nodes)))
            neigh_agg = self.aggregation(nodes_fusion, neighs, neigh_type, batch_size) # (batch, dim)
            neigh_type_agg.append(neigh_agg)

        type_agg = torch.Tensor(reduce(lambda x, y: torch.cat((x, y), 1), neigh_type_agg).cpu()).to(
            nodes.device)  # (batch, len(neigh_type_set)*dim)
        map_type_agg = self.w_type_att(type_agg)  # (batch, #type)
        att = F.softmax(map_type_agg, dim=1).view(batch_size, self.num_neigh_type, 1)  # (b, #type)
        neigh_agg_final = torch.matmul(torch.transpose(type_agg.view(batch_size, -1, self.emb_size), dim0=1, dim1=2),
                                       att).squeeze()  # (batch, dim)
        neigh_agg_final = F.relu(self.linear_2(neigh_agg_final))

        combined_feature = torch.cat([nodes_fusion, neigh_agg_final], dim=1)
        combined_feature = F.relu(self.linear(combined_feature))

        return combined_feature, att

    def aggregation(self, nodes_fusion, nodes_neighbors, neigh_type, batch_size):
        neighbors_embeddings = self.neighbors_embeddings_dict[neigh_type].to(self.device)
        neighbors_profiles = self.neighbors_profiles_dict[neigh_type].to(self.device)

        neighs_fusion = list(map(lambda x: self.fusion(neighbors_embeddings.weight[x], neighbors_profiles[x]),
                                 nodes_neighbors))

        attention_list = list(
            map(lambda idx: self.att(nodes_fusion[idx], neighs_fusion[idx], len(nodes_neighbors[idx])),
                range(batch_size)))
        neigh_feature_matrix = torch.stack(
            [torch.mm(neighs_fusion[idx].t().to(self.device), attention_list[idx].to(self.device)) for idx in range(batch_size)]
        ).reshape(batch_size, self.emb_size).to(nodes_fusion.device)

        combined_feature = F.relu(self.linear_1(neigh_feature_matrix))

        return combined_feature  # (#node, emb_size)
