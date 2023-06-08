import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl import function as fn
from . import BaseModel,register_model


@register_model('KGAT')
class KGAT(BaseModel):
    r"""
    This model KGAT was introduced in `KGAT <https://arxiv.org/pdf/1905.07854v2.pdf>`__.

    It involves some careful design.

    Embedding Layer:

    Knowledge graph embedding is an effective way to parameterize entities and relations as vector representations.
    KGAT employ TransR, a widely used method, on CKG.

    Attentive Embedding Propagation Layers:

    A single attentive embedding propagation layer consists of three components: information propagation, knowledge-aware attention, and information aggregation.

    1. Information Propagation:

    Considering an entity :math:`h`, we use :math:`\mathcal{N}_{h} = {(h,r,t)|(h,r,t)\in\mathcal{G}}` to denote the set of triplets where :math:`h` is the head entity, termed ego-network.
    To characterize the first-order connectivity structure of entity :math:`h`, we compute the linear combination of :math:`h`’s ego-network:

    :math:`e_{\mathcal{N}_{h}}=\sum_{(h,r,t)\in\mathcal{N}_{h}}\pi(h,r,t)e_{t}`

    where :math:`\pi(h,r,t)` controls the decay factor on each propagation on edge :math:`(h,r,t)`, indicating how much information being propagated from :math:`t` to :math:`h` conditioned to relation :math:`r`.

    2. Knowledge-aware Attention:

    GAT implement :math:`\pi(h,r,t)` via relational attention mechanism, which is formulated as follows:

    :math:`\pi(h,r,t)=(\mathcal{N}_{r}\mathcal{e}_{t})^{\mathsf{T}}tanh((\mathcal{W}_{r}\mathcal{e}_{h}+e_{r}))`

    This makes the attention score dependent on the distance between :math:`e_h` and :math:`e_t` in the relation :math:`r`’s space
    Hereafter, GAT normalize the coefficients across all triplets connected with :math:`h` by adopting the softmax function:

    :math:`\pi(h,r,t)=\frac{exp(\pi(h,r,t))}{\sum_{(h,r',t')\in\mathcal{N}_{h}}exp(\pi(h,r',t'))}`

    As a result, the final attention score is capable of suggesting which neighbor nodes should be given more attention to capture collaborative signals.

    3. Information Aggregation:
    The final phase is to aggregate the entity representation :math:`e_h` and its ego-network representations :math:`e_{\mathcal{N}_h}` as the new representation of entity :math:`h`

    GCN Aggregator : :math:`\mathcal{f}_{GCN}=LeakyReLU(\mathcal{W}(e_h+e_{\mathcal{N}_h}))`

    GraphSage Aggregator : :math:`\mathcal{f}_{GraphSage}=LeakyReLU(\mathcal{W}(e_h||e_{\mathcal{N}_h}))`

    Bi-Interaction Aggregator : :math:`\mathcal{f}_{Bi-Interaction}=LeakyReLU(\mathcal{W}(e_h+e_{\mathcal{N}_h})+LeakyReLU(\mathcal{W}(e_h\odote_{\mathcal{N}_h})`

    High-order Propagation:
    We can further stack more propagation layers to explore the high-order connectivity information, gathering the information propagated from the higher-hop neighbors.

    :math:`e_{h}^{(l)}=\mathcal{f}(e_{h}^{(l-1)}_{\mathcal{N}_h})`

    Model Prediction：

    After performing :math:`L` layers, we obtain multiple representations for user node :math:`u`, namely :math:`{e_{u}^{(1)},...,{e_{u}^{(L)}}`; Analogous to item node i, :math:`{e_{i}^{(1)},...,{e_{i}^{(L)}}` are obtained.
    GAT hence adopt the layer-aggregation mechanism to concatenate the representations at each step into a single vector, as follows:

    :math:`e^*_u=e_u^{(0)}||...||e_u^{(L)},e^*_i=e_i^{(0)}||...||e_i^{(L)}`

    Finally, we conduct inner product of user and item representations, so as to predict their matching score:

    :math:`\check{\mathcal{y}}(u,i)=e^*_u`\mathsf{T}e^*_i`

    Parameters
    ----------
    entity_dim ：User / entity Embedding size
    relation_dim ： Relation Embedding size
    aggregation_type ： Specify the type of the aggregation layer from {gcn, graphsage, bi-interaction}
    conv_dim_list ： Output sizes of every aggregation layer
    mess_dropout ： Dropout probability w.r.t. message dropout for each deep layer. 0: no dropout
    """
    @classmethod
    def build_model_from_args(cls,args,hg):
        return cls(args=args)

    def __init__(self,args):
        super(KGAT, self).__init__()
        if args.aggregation_type not in ['bi-interaction','gcn','graphsage']:
            raise KeyError('Aggregator type {} not supported.'.format(args.aggregation_type))
        self.use_pretrain = args.use_pretrain
        self.entity_dim = args.entity_dim
        self.relation_dim = args.relation_dim
        self.aggregation_type = args.aggregation_type
        self.conv_dim_list = [args.entity_dim] + eval(args.conv_dim_list)
        self.mess_dropout = eval(args.mess_dropout)
        self.n_layers = len(eval(args.conv_dim_list))

        self.kg_l2loss_lambda = args.kg_l2loss_lambda
        self.cf_l2loss_lambda = args.cf_l2loss_lambda

        self.aggregator_layers = nn.ModuleList()
        for k in range(self.n_layers):
            self.aggregator_layers.append(KGAT_Aggregator(self.conv_dim_list[k], self.conv_dim_list[k + 1], self.mess_dropout[k], self.aggregation_type))

    def set_parameters(self, n_users, n_entities, n_relations, user_pre_embed=None, item_pre_embed=None):

        self.n_users = n_users
        self.n_entities = n_entities
        self.n_relations = n_relations

        # Embedding
        self.relation_embed = nn.Embedding(self.n_relations, self.relation_dim)
        self.entity_user_embed = nn.Embedding(self.n_entities + self.n_users, self.entity_dim)
        if (self.use_pretrain == 1) and (user_pre_embed is not None) and (item_pre_embed is not None):
            other_entity_embed = nn.Parameter(torch.Tensor(self.n_entities - item_pre_embed.shape[0], self.entity_dim))
            nn.init.xavier_uniform_(other_entity_embed, gain=nn.init.calculate_gain('relu'))
            entity_user_embed = torch.cat([item_pre_embed, other_entity_embed, user_pre_embed], dim=0)
            self.entity_user_embed.weight = nn.Parameter(entity_user_embed)

        self.W_R = nn.Parameter(torch.Tensor(self.n_relations, self.entity_dim, self.relation_dim))
        nn.init.xavier_uniform_(self.W_R, gain=nn.init.calculate_gain('relu'))

    '''
    # DGL: dgl-cu101(0.4.3)
    # We will get different results when using the function `fn.sum`, and the randomness is due to `atomicAdd`.
    # Use custom function to ensure deterministic behavior.
    '''

    def edge_softmax_fix(self,graph,score):

        def reduce_sum(nodes):
            accum = torch.sum(nodes.mailbox['temp'], 1)
            return {'out_sum': accum}

        graph = graph.local_var()
        graph.edata['out'] = score
        graph.edata['out'] = torch.exp(graph.edata['out'])
        graph.update_all(fn.copy_e('out', 'temp'), reduce_sum)
        graph.apply_edges(fn.e_div_v('out', 'out_sum', 'out'))
        out = graph.edata['out']
        return out

    def att_score(self, edges):
        # Equation (4)
        r_mul_t = torch.matmul(self.entity_user_embed(edges.src['id']), self.W_r)                       # (n_edge, relation_dim)
        r_mul_h = torch.matmul(self.entity_user_embed(edges.dst['id']), self.W_r)                       # (n_edge, relation_dim)
        r_embed = self.relation_embed(edges.data['type'])                                               # (1, relation_dim)
        att = torch.bmm(r_mul_t.unsqueeze(1), torch.tanh(r_mul_h + r_embed).unsqueeze(2)).squeeze(-1)   # (n_edge, 1)
        return {'att': att}

    def compute_attention(self, g):
        g = g.local_var()
        for i in range(self.n_relations):
            edge_idxs = g.filter_edges(lambda edge: edge.data['type'] == i)
            self.W_r = self.W_R[i]  # [entity_dim, relation_dim]
            g.apply_edges(self.att_score, edge_idxs)

        # Equation (5)
        g.edata['att'] = self.edge_softmax_fix(g, g.edata.pop('att'))
        return g.edata.pop('att')

    def _L2_loss_mean(self,x):
        return torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.)

    def calc_kg_loss(self, h, r, pos_t, neg_t):   #f
        """
        h:      (kg_batch_size)
        r:      (kg_batch_size)
        pos_t:  (kg_batch_size)
        neg_t:  (kg_batch_size)
        """
        r_embed = self.relation_embed(r)                 # (kg_batch_size, relation_dim)
        W_r = self.W_R[r]                                # (kg_batch_size, entity_dim, relation_dim)

        h_embed = self.entity_user_embed(h)              # (kg_batch_size, entity_dim)
        pos_t_embed = self.entity_user_embed(pos_t)      # (kg_batch_size, entity_dim)
        neg_t_embed = self.entity_user_embed(neg_t)      # (kg_batch_size, entity_dim)

        r_mul_h = torch.bmm(h_embed.unsqueeze(1), W_r).squeeze(1)             # (kg_batch_size, relation_dim)
        r_mul_pos_t = torch.bmm(pos_t_embed.unsqueeze(1), W_r).squeeze(1)     # (kg_batch_size, relation_dim)
        r_mul_neg_t = torch.bmm(neg_t_embed.unsqueeze(1), W_r).squeeze(1)     # (kg_batch_size, relation_dim)

        # Equation (1)
        pos_score = torch.sum(torch.pow(r_mul_h + r_embed - r_mul_pos_t, 2), dim=1)     # (kg_batch_size)
        neg_score = torch.sum(torch.pow(r_mul_h + r_embed - r_mul_neg_t, 2), dim=1)     # (kg_batch_size)

        # Equation (2)
        kg_loss = (-1.0) * F.logsigmoid(neg_score - pos_score)
        kg_loss = torch.mean(kg_loss)

        l2_loss = self._L2_loss_mean(r_mul_h) + self._L2_loss_mean(r_embed) + self._L2_loss_mean(r_mul_pos_t) + self._L2_loss_mean(r_mul_neg_t)
        loss = kg_loss + self.kg_l2loss_lambda * l2_loss
        return loss

    def cf_embedding(self, mode, g):
        g = g.local_var()
        ego_embed = self.entity_user_embed(g.ndata['id'])
        all_embed = [ego_embed]

        for i, layer in enumerate(self.aggregator_layers):
            ego_embed = layer(mode, g, ego_embed)
            norm_embed = F.normalize(ego_embed, p=2, dim=1)
            all_embed.append(norm_embed)

        # Equation (11)
        all_embed = torch.cat(all_embed, dim=1)         # (n_users + n_entities, cf_concat_dim)
        return all_embed

    def cf_score(self, mode, g, user_ids, item_ids):
        """
        user_ids:   number of users to evaluate   (n_eval_users)
        item_ids:   number of items to evaluate   (n_eval_items)
        """
        all_embed = self.cf_embedding(mode, g)          # (n_users + n_entities, cf_concat_dim)
        user_embed = all_embed[user_ids]                # (n_eval_users, cf_concat_dim)
        item_embed = all_embed[item_ids]                # (n_eval_items, cf_concat_dim)

        # Equation (12)
        cf_score = torch.matmul(user_embed, item_embed.transpose(0, 1))    # (n_eval_users, n_eval_items)
        return cf_score

    def calc_cf_loss(self, mode, g, user_ids, item_pos_ids, item_neg_ids):
        """
        user_ids:       (cf_batch_size)
        item_pos_ids:   (cf_batch_size)
        item_neg_ids:   (cf_batch_size)
        """
        all_embed = self.cf_embedding(mode, g)                      # (n_users + n_entities, cf_concat_dim)
        user_embed = all_embed[user_ids]                            # (cf_batch_size, cf_concat_dim)
        item_pos_embed = all_embed[item_pos_ids]                    # (cf_batch_size, cf_concat_dim)
        item_neg_embed = all_embed[item_neg_ids]                    # (cf_batch_size, cf_concat_dim)

        # Equation (12)
        pos_score = torch.sum(user_embed * item_pos_embed, dim=1)   # (cf_batch_size)
        neg_score = torch.sum(user_embed * item_neg_embed, dim=1)   # (cf_batch_size)

        # Equation (13)
        cf_loss = (-1.0) * F.logsigmoid(pos_score - neg_score)
        cf_loss = torch.mean(cf_loss)

        l2_loss = self._L2_loss_mean(user_embed) + self._L2_loss_mean(item_pos_embed) + self._L2_loss_mean(item_neg_embed)
        loss = cf_loss + self.cf_l2loss_lambda * l2_loss
        return loss

    def forward(self, mode, *input):
        if mode == 'calc_att':
            return self.compute_attention(*input)
        if mode == 'calc_cf_loss':
            return self.calc_cf_loss(mode, *input)
        if mode == 'calc_kg_loss':
            return self.calc_kg_loss(*input)
        if mode == 'predict':
            return self.cf_score(mode, *input)


class KGAT_Aggregator(nn.Module):

    def __init__(self, in_dim, out_dim, dropout, aggregator_type):
        super(KGAT_Aggregator, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.aggregator_type = aggregator_type

        self.message_dropout = nn.Dropout(dropout)

        if aggregator_type == 'gcn':
            self.W = nn.Linear(self.in_dim, self.out_dim)       # W in Equation (6)
        elif aggregator_type == 'graphsage':
            self.W = nn.Linear(self.in_dim * 2, self.out_dim)   # W in Equation (7)
        elif aggregator_type == 'bi-interaction':
            self.W1 = nn.Linear(self.in_dim, self.out_dim)      # W1 in Equation (8)
            self.W2 = nn.Linear(self.in_dim, self.out_dim)      # W2 in Equation (8)
        else:
            raise NotImplementedError

        self.activation = nn.LeakyReLU()

    def forward(self, mode, g, entity_embed):
        g = g.local_var()
        g.ndata['node'] = entity_embed

        # Equation (3) & (10)
        # DGL: dgl-cu10.1(0.5.3)
        # Get different results when using `dgl.function.sum`, and the randomness is due to `atomicAdd`
        # Use `dgl.function.sum` when training model to speed up
        # Use custom function to ensure deterministic behavior when predicting
        if mode == 'predict':
            g.update_all(dgl.function.u_mul_e('node', 'att', 'side'), lambda nodes: {'N_h': torch.sum(nodes.mailbox['side'], 1)})
        else:
            g.update_all(dgl.function.u_mul_e('node', 'att', 'side'), dgl.function.sum('side', 'N_h'))

        if self.aggregator_type == 'gcn':
            # Equation (6) & (9)
            out = self.activation(self.W(g.ndata['node'] + g.ndata['N_h']))                    # (n_users + n_entities, out_dim)

        elif self.aggregator_type == 'graphsage':
            # Equation (7) & (9)
            out = self.activation(self.W(torch.cat([g.ndata['node'], g.ndata['N_h']], dim=1))) # (n_users + n_entities, out_dim)

        elif self.aggregator_type == 'bi-interaction':
            # Equation (8) & (9)
            out1 = self.activation(self.W1(g.ndata['node'] + g.ndata['N_h']))                  # (n_users + n_entities, out_dim)
            out2 = self.activation(self.W2(g.ndata['node'] * g.ndata['N_h']))                  # (n_users + n_entities, out_dim)
            out = out1 + out2
        else:
            raise NotImplementedError

        out = self.message_dropout(out)
        return out