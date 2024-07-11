import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from itertools import product
from . import BaseModel, register_model


@register_model('ExpressGNN')
class ExpressGNN(BaseModel):
    @classmethod
    def build_model_from_args(cls, args, hg):
        return cls(args=args,
                   latent_dim=args.embedding_size - args.gcn_free_size,
                   free_dim=args.gcn_free_size,
                   device=args.device,
                   load_method=args.load_method,
                   rule_list=args.rule_list,
                   rule_weights_learning=args.rule_weights_learning,
                   graph=hg,
                   PRED_DICT=args.PRED_DICT,
                   slice_dim=args.slice_dim,
                   transductive=(args.trans == 1))

    def __init__(self, args, graph, latent_dim, free_dim, device, load_method, rule_list, rule_weights_learning,
                 PRED_DICT,
                 num_hops=5, num_layers=2, slice_dim=5, transductive=True):
        """

        Parameters
        ----------
        graph: knowledge graph
        latent_dim: embedding_size - gcn_free_size
        free_dim: gcn_free_size
        device: device
        load_method: Factorized Posterior's load method, use args to get
        rule_list: MLN's rules, should come from dataset
        rule_weights_learning: MLN's args, should come from args
        num_hops: number of hops of GCN
        num_layers: number of layers of GCN
        slice_dim: Used by Factorized Posterior
        transductive: Used by GCN
        """
        # GCN's setting
        super(ExpressGNN, self).__init__()
        self.graph = graph
        self.latent_dim = latent_dim
        self.free_dim = free_dim
        self.num_hops = num_hops
        self.num_layers = num_layers
        self.PRED_DICT = PRED_DICT
        self.args = args
        self.num_ents = graph.num_ents
        self.num_rels = graph.num_rels
        self.num_nodes = graph.num_nodes
        self.num_edges = graph.num_edges
        self.num_edge_types = len(graph.edge_type2idx)

        # Factorized Posterior's loss function
        self.xent_loss = F.binary_cross_entropy_with_logits

        # Factorized Posterior's
        self.load_method = load_method
        self.num_rels = graph.num_rels
        self.ent2idx = graph.ent2idx
        self.rel2idx = graph.rel2idx
        self.idx2rel = graph.idx2rel

        # Trainable Embedding
        self.num_ents = self.graph.num_ents
        self.ent_embeds = nn.Embedding(self.num_ents, self.args.embedding_size)
        self.ents = torch.arange(self.num_ents).to(args.device)

        self.edge2node_in, self.edge2node_out, self.node_degree, \
            self.edge_type_masks, self.edge_direction_masks = self.gen_edge2node_mapping()

        self.node_feat, self.const_nodes = self.prepare_node_feature(graph, transductive=transductive)

        if not transductive:
            self.node_feat_dim = 1 + self.num_rels
        else:
            self.node_feat_dim = self.num_ents + self.num_rels

        self.init_node_linear = nn.Linear(self.node_feat_dim, latent_dim, bias=False)

        for param in self.init_node_linear.parameters():
            param.requires_grad = False

        self.node_feat = self.node_feat.to(device)
        self.const_nodes = self.const_nodes.to(device)
        self.edge2node_in = self.edge2node_in.to(device)
        self.edge2node_out = self.edge2node_out.to(device)
        self.edge_type_masks = [mask.to(device) for mask in self.edge_type_masks]
        self.edge_direction_masks = [mask.to(device) for mask in self.edge_direction_masks]
        self.MLPs = nn.ModuleList()
        for _ in range(self.num_hops):
            self.MLPs.append(MLP(input_size=self.latent_dim, num_layers=self.num_layers,
                                 hidden_size=self.latent_dim, output_size=self.latent_dim))

        self.edge_type_W = nn.ModuleList()
        for _ in range(self.num_edge_types):
            ml_edge_type = nn.ModuleList()
            for _ in range(self.num_hops):
                ml_hop = nn.ModuleList()
                for _ in range(2):  # 2 directions of edges
                    ml_hop.append(nn.Linear(latent_dim, latent_dim, bias=False))
                ml_edge_type.append(ml_hop)
            self.edge_type_W.append(ml_edge_type)
        self.const_nodes_free_params = nn.Parameter(nn.init.kaiming_uniform_(torch.zeros(self.num_ents, free_dim)))
        # load Factorized Posterior
        if load_method == 1:
            self.params_u_R = nn.ModuleList()
            self.params_W_R = nn.ModuleList()
            self.params_V_R = nn.ModuleList()
            for idx in range(self.num_rels):
                rel = self.idx2rel[idx]
                num_args = self.PRED_DICT[rel].num_args
                self.params_W_R.append(
                    nn.Bilinear(num_args * args.embedding_size, num_args * args.embedding_size, slice_dim, bias=False))
                self.params_V_R.append(nn.Linear(num_args * args.embedding_size, slice_dim, bias=True))
                self.params_u_R.append(nn.Linear(slice_dim, 1, bias=False))
        elif load_method == 0:
            self.params_u_R = nn.ParameterList()
            self.params_W_R = nn.ModuleList()
            self.params_V_R = nn.ModuleList()
            self.params_b_R = nn.ParameterList()
            for idx in range(self.num_rels):
                rel = self.idx2rel[idx]
                num_args = self.PRED_DICT[rel].num_args
                self.params_u_R.append(nn.Parameter(nn.init.kaiming_uniform_(torch.zeros(slice_dim, 1)).view(-1)))
                self.params_W_R.append(
                    nn.Bilinear(num_args * args.embedding_size, num_args * args.embedding_size, slice_dim, bias=False))
                self.params_V_R.append(nn.Linear(num_args * args.embedding_size, slice_dim, bias=False))
                self.params_b_R.append(nn.Parameter(nn.init.kaiming_uniform_(torch.zeros(slice_dim, 1)).view(-1)))

        # --- MLN ---

        self.rule_weights_lin = nn.Linear(len(rule_list), 1, bias=False)
        self.num_rules = len(rule_list)
        self.soft_logic = False

        self.alpha_table = nn.Parameter(torch.tensor([10.0 for _ in range(len(self.PRED_DICT))], requires_grad=True))

        self.predname2ind = dict(e for e in zip(self.PRED_DICT.keys(), range(len(self.PRED_DICT))))

        if rule_weights_learning == 0:
            self.rule_weights_lin.weight.data = torch.tensor([[rule.weight for rule in rule_list]],
                                                             dtype=torch.float)
            print('rule weights fixed as pre-defined values\n')
        else:
            self.rule_weights_lin.weight = nn.Parameter(
                torch.tensor([[rule.weight for rule in rule_list]], dtype=torch.float))
            print('rule weights set to pre-defined values, learning weights\n')

    def gcn_forward(self, batch_data):
        if self.args.use_gcn == 0:
            node_embeds = self.ent_embeds(self.ents)
            return node_embeds
        else:
            node_embeds = self.init_node_linear(self.node_feat)

            hop = 0
            hidden = node_embeds
            while hop < self.num_hops:
                node_aggregate = torch.zeros_like(hidden)
                for edge_type in set(self.graph.edge_types):
                    for direction in range(2):
                        W = self.edge_type_W[edge_type][hop][direction]
                        W_nodes = W(hidden)
                        nodes_attached_on_edges_out = torch.gather(W_nodes, 0, self.edge2node_out)
                        nodes_attached_on_edges_out *= self.edge_type_masks[edge_type].view(-1, 1)
                        nodes_attached_on_edges_out *= self.edge_direction_masks[direction].view(-1, 1)
                        node_aggregate.scatter_add_(0, self.edge2node_in, nodes_attached_on_edges_out)

                hidden = self.MLPs[hop](hidden + node_aggregate)
                hop += 1

            read_out_const_nodes_embed = torch.cat((hidden[self.const_nodes], self.const_nodes_free_params), dim=1)

            return read_out_const_nodes_embed

    def posterior_forward(self, latent_vars, node_embeds, batch_mode=False, fast_mode=False, fast_inference_mode=False):
        """
        compute posterior probabilities of specified latent variables

        :param latent_vars:
            list of latent variables (i.e. unobserved facts)
        :param node_embeds:
            node embeddings
        :return:
            n-dim vector, probability of corresponding latent variable being True

        Parameters
        ----------
        fast_inference_mode
        fast_mode
        batch_mode
        """

        # this mode is only for fast inference on Freebase data
        if fast_inference_mode:
            assert self.load_method == 1

            samples = latent_vars
            scores = []

            for ind in range(len(samples)):
                pred_name, pred_sample = samples[ind]

                rel_idx = self.rel2idx[pred_name]

                sample_mat = torch.tensor(pred_sample, dtype=torch.long).to(self.args.device)  # (bsize, 2)

                sample_query = torch.cat([node_embeds[sample_mat[:, 0]], node_embeds[sample_mat[:, 1]]], dim=1)

                sample_score = self.params_u_R[rel_idx](
                    torch.tanh(self.params_W_R[rel_idx](sample_query, sample_query) +
                               self.params_V_R[rel_idx](sample_query))).view(-1)  # (bsize)
                scores.append(torch.sigmoid(sample_score))
            return scores

        # this mode is only for fast training on Freebase data
        elif fast_mode:

            assert self.load_method == 1

            samples, neg_mask, latent_mask, obs_var, neg_var = latent_vars
            scores = []
            obs_probs = []
            neg_probs = []
            a = []
            for pred_mask in neg_mask:
                a.append(pred_mask[1])
            pos_mask_mat = torch.tensor(a)
            pos_mask_mat = pos_mask_mat.to(self.args.device)
            neg_mask_mat = (pos_mask_mat == 0).type(torch.float)
            latent_mask_mat = torch.tensor([pred_mask[1] for pred_mask in latent_mask], dtype=torch.float).to(
                self.args.device)
            obs_mask_mat = (latent_mask_mat == 0).type(torch.float)
            for ind in range(len(samples)):
                pred_name, pred_sample = samples[ind]
                _, obs_sample = obs_var[ind]
                _, neg_sample = neg_var[ind]

                rel_idx = self.rel2idx[pred_name]

                sample_mat = torch.tensor(pred_sample, dtype=torch.long).to(self.args.device)
                obs_mat = torch.tensor(obs_sample, dtype=torch.long).to(self.args.device)
                neg_mat = torch.tensor(neg_sample, dtype=torch.long).to(self.args.device)

                sample_mat = torch.cat([sample_mat, obs_mat, neg_mat], dim=0)

                sample_query = torch.cat([node_embeds[sample_mat[:, 0]], node_embeds[sample_mat[:, 1]]], dim=1)

                sample_score = self.params_u_R[rel_idx](
                    torch.tanh(self.params_W_R[rel_idx](sample_query, sample_query) +
                               self.params_V_R[rel_idx](sample_query))).view(-1)
                var_prob = sample_score[len(pred_sample):]
                obs_prob = var_prob[:len(obs_sample)]
                neg_prob = var_prob[len(obs_sample):]
                sample_score = sample_score[:len(pred_sample)]

                scores.append(sample_score)
                obs_probs.append(obs_prob)
                neg_probs.append(neg_prob)
            score_mat = torch.stack(scores, dim=0)
            score_mat = torch.sigmoid(score_mat)

            pos_score = (1 - score_mat) * pos_mask_mat
            neg_score = score_mat * neg_mask_mat

            potential = 1 - ((pos_score + neg_score) * latent_mask_mat + obs_mask_mat).prod(dim=0)

            obs_mat = torch.cat(obs_probs, dim=0)

            if obs_mat.size(0) == 0:
                obs_loss = 0.0
            else:
                obs_loss = self.xent_loss(obs_mat, torch.ones_like(obs_mat), reduction='sum')

            neg_mat = torch.cat(neg_probs, dim=0)
            if neg_mat.size(0) != 0:
                obs_loss += self.xent_loss(obs_mat, torch.zeros_like(neg_mat), reduction='sum')

            obs_loss /= (obs_mat.size(0) + neg_mat.size(0) + 1e-6)
            return potential, (score_mat * latent_mask_mat).view(-1), obs_loss

        elif batch_mode:
            assert self.load_method == 1

            pred_name, x_mat, invx_mat, sample_mat = latent_vars

            rel_idx = self.rel2idx[pred_name]

            x_mat = torch.tensor(x_mat, dtype=torch.long).to(self.args.device)
            invx_mat = torch.tensor(invx_mat, dtype=torch.long).to(self.args.device)
            sample_mat = torch.tensor(sample_mat, dtype=torch.long).to(self.args.device)

            tail_query = torch.cat([node_embeds[x_mat[:, 0]], node_embeds[x_mat[:, 1]]], dim=1)
            head_query = torch.cat([node_embeds[invx_mat[:, 0]], node_embeds[invx_mat[:, 1]]], dim=1)
            true_query = torch.cat([node_embeds[sample_mat[:, 0]], node_embeds[sample_mat[:, 1]]], dim=1)

            tail_score = self.params_u_R[rel_idx](torch.tanh(self.params_W_R[rel_idx](tail_query, tail_query) +
                                                             self.params_V_R[rel_idx](tail_query))).view(-1)

            head_score = self.params_u_R[rel_idx](torch.tanh(self.params_W_R[rel_idx](head_query, head_query) +
                                                             self.params_V_R[rel_idx](head_query))).view(-1)

            true_score = self.params_u_R[rel_idx](torch.tanh(self.params_W_R[rel_idx](true_query, true_query) +
                                                             self.params_V_R[rel_idx](true_query))).view(-1)

            probas_tail = torch.sigmoid(tail_score)
            probas_head = torch.sigmoid(head_score)
            probas_true = torch.sigmoid(true_score)
            return probas_tail, probas_head, probas_true

        else:
            assert self.load_method == 0

            probas = torch.zeros(len(latent_vars)).to(self.args.device)
            for i in range(len(latent_vars)):
                rel, args = latent_vars[i]
                args_embed = torch.cat([node_embeds[self.ent2idx[arg]] for arg in args], 0)
                rel_idx = self.rel2idx[rel]

                score = self.params_u_R[rel_idx].dot(
                    torch.tanh(self.params_W_R[rel_idx](args_embed, args_embed) +
                               self.params_V_R[rel_idx](args_embed) +
                               self.params_b_R[rel_idx])
                )
                proba = torch.sigmoid(score)
                probas[i] = proba
            return probas

    def mln_forward(self, neg_mask_ls_ls, latent_var_inds_ls_ls, observed_rule_cnts, posterior_prob, flat_list,
                    observed_vars_ls_ls):
        """
        compute the MLN potential given the posterior probability of latent variables
        :param neg_mask_ls_ls:

        :return:

        Parameters
        ----------
        flat_list
        posterior_prob
        observed_vars_ls_ls
        latent_var_inds_ls_ls
        observed_rule_cnts
        """

        scores = torch.zeros(self.num_rules, dtype=torch.float, device=self.args.device)
        pred_ind_flat_list = []
        if self.soft_logic:
            pred_name_ls = [e[0] for e in flat_list]
            pred_ind_flat_list = [self.predname2ind[pred_name] for pred_name in pred_name_ls]

        for i in range(len(neg_mask_ls_ls)):
            neg_mask_ls = neg_mask_ls_ls[i]
            latent_var_inds_ls = latent_var_inds_ls_ls[i]
            observed_vars_ls = observed_vars_ls_ls[i]

            # sum of scores from gnd rules with latent vars
            for j in range(len(neg_mask_ls)):

                latent_neg_mask, observed_neg_mask = neg_mask_ls[j]
                latent_var_inds = latent_var_inds_ls[j]
                observed_vars = observed_vars_ls[j]

                z_probs = posterior_prob[latent_var_inds].unsqueeze(0)

                z_probs = torch.cat([1 - z_probs, z_probs], dim=0)

                cartesian_prod = z_probs[:, 0]
                for j in range(1, z_probs.shape[1]):
                    cartesian_prod = torch.ger(cartesian_prod, z_probs[:, j])
                    cartesian_prod = cartesian_prod.view(-1)

                view_ls = [2 for _ in range(len(latent_neg_mask))]
                cartesian_prod = cartesian_prod.view(*[view_ls])

                if self.soft_logic:

                    # observed alpha
                    obs_vals = [e[0] for e in observed_vars]
                    pred_names = [e[1] for e in observed_vars]
                    pred_inds = [self.predname2ind[pn] for pn in pred_names]
                    alpha = self.alpha_table[pred_inds]  # alphas in this formula
                    act_alpha = torch.sigmoid(alpha)
                    obs_neg_flag = [(1 if observed_vars[i] != observed_neg_mask[i] else 0)
                                    for i in range(len(observed_vars))]
                    tn_obs_neg_flag = torch.tensor(obs_neg_flag, dtype=torch.float)

                    val = torch.abs(1 - torch.tensor(obs_vals, dtype=torch.float) - act_alpha)
                    obs_score = torch.abs(tn_obs_neg_flag - val)

                    # latent alpha
                    inds = product(*[[0, 1] for _ in range(len(latent_neg_mask))])
                    pred_inds = [pred_ind_flat_list[i] for i in latent_var_inds]
                    alpha = self.alpha_table[pred_inds]  # alphas in this formula
                    act_alpha = torch.sigmoid(alpha)
                    tn_latent_neg_mask = torch.tensor(latent_neg_mask, dtype=torch.float)

                    for ind in inds:
                        val = torch.abs(1 - torch.tensor(ind, dtype=torch.float) - act_alpha)
                        val = torch.abs(tn_latent_neg_mask - val)
                        cartesian_prod[tuple(ind)] *= torch.max(torch.cat([val, obs_score], dim=0))

                else:

                    if sum(observed_neg_mask) == 0:
                        cartesian_prod[tuple(latent_neg_mask)] = 0.0

                scores[i] += cartesian_prod.sum()

            # sum of scores from gnd rule with only observed vars
            scores[i] += observed_rule_cnts[i]

        return self.rule_weights_lin(scores)

    def gen_edge2node_mapping(self):
        """
        A GCN's function
        Returns
        -------

        """
        ei = 0  # edge index with direction
        edge_idx = 0  # edge index without direction
        edge2node_in = torch.zeros(self.num_edges * 2, dtype=torch.long)
        edge2node_out = torch.zeros(self.num_edges * 2, dtype=torch.long)
        node_degree = torch.zeros(self.num_nodes)

        edge_type_masks = []
        for _ in range(self.num_edge_types):
            edge_type_masks.append(torch.zeros(self.num_edges * 2))
        edge_direction_masks = []
        for _ in range(2):  # 2 directions of edges
            edge_direction_masks.append(torch.zeros(self.num_edges * 2))

        for ni, nj in torch.as_tensor(self.graph.edge_pairs):
            edge_type = self.graph.edge_types[edge_idx]
            edge_idx += 1

            edge2node_in[ei] = nj
            edge2node_out[ei] = ni
            node_degree[ni] += 1
            edge_type_masks[edge_type][ei] = 1
            edge_direction_masks[0][ei] = 1
            ei += 1

            edge2node_in[ei] = ni
            edge2node_out[ei] = nj
            node_degree[nj] += 1
            edge_type_masks[edge_type][ei] = 1
            edge_direction_masks[1][ei] = 1
            ei += 1

        edge2node_in = edge2node_in.view(-1, 1).expand(-1, self.latent_dim)
        edge2node_out = edge2node_out.view(-1, 1).expand(-1, self.latent_dim)
        node_degree = node_degree.view(-1, 1)
        return edge2node_in, edge2node_out, node_degree, edge_type_masks, edge_direction_masks

    def weight_update(self, neg_mask_ls_ls, latent_var_inds_ls_ls, observed_rule_cnts, posterior_prob, flat_list,
                      observed_vars_ls_ls):
        """
        A MLN's Function
        Parameters
        ----------
        neg_mask_ls_ls
        latent_var_inds_ls_ls
        observed_rule_cnts
        posterior_prob
        flat_list
        observed_vars_ls_ls

        Returns
        -------

        """
        closed_wolrd_potentials = torch.zeros(self.num_rules, dtype=torch.float)
        pred_ind_flat_list = []
        if self.soft_logic:
            pred_name_ls = [e[0] for e in flat_list]
            pred_ind_flat_list = [self.predname2ind[pred_name] for pred_name in pred_name_ls]

        for i in range(len(neg_mask_ls_ls)):
            neg_mask_ls = neg_mask_ls_ls[i]
            latent_var_inds_ls = latent_var_inds_ls_ls[i]
            observed_vars_ls = observed_vars_ls_ls[i]

            # sum of scores from gnd rules with latent vars
            for j in range(len(neg_mask_ls)):

                latent_neg_mask, observed_neg_mask = neg_mask_ls[j]
                latent_var_inds = latent_var_inds_ls[j]
                observed_vars = observed_vars_ls[j]

                has_pos_atom = False
                for val in observed_neg_mask + latent_neg_mask:
                    if val == 1:
                        has_pos_atom = True
                        break

                if has_pos_atom:
                    closed_wolrd_potentials[i] += 1

                z_probs = posterior_prob[latent_var_inds].unsqueeze(0)

                z_probs = torch.cat([1 - z_probs, z_probs], dim=0)

                cartesian_prod = z_probs[:, 0]
                for j in range(1, z_probs.shape[1]):
                    cartesian_prod = torch.ger(cartesian_prod, z_probs[:, j])
                    cartesian_prod = cartesian_prod.view(-1)

                view_ls = [2 for _ in range(len(latent_neg_mask))]
                cartesian_prod = cartesian_prod.view(*[view_ls])

                if self.soft_logic:

                    # observed alpha
                    obs_vals = [e[0] for e in observed_vars]
                    pred_names = [e[1] for e in observed_vars]
                    pred_inds = [self.predname2ind[pn] for pn in pred_names]
                    alpha = self.alpha_table[pred_inds]  # alphas in this formula
                    act_alpha = torch.sigmoid(alpha)
                    obs_neg_flag = [(1 if observed_vars[i] != observed_neg_mask[i] else 0)
                                    for i in range(len(observed_vars))]
                    tn_obs_neg_flag = torch.tensor(obs_neg_flag, dtype=torch.float)

                    val = torch.abs(1 - torch.tensor(obs_vals, dtype=torch.float) - act_alpha)
                    obs_score = torch.abs(tn_obs_neg_flag - val)

                    # latent alpha
                    inds = product(*[[0, 1] for _ in range(len(latent_neg_mask))])
                    pred_inds = [pred_ind_flat_list[i] for i in latent_var_inds]
                    alpha = self.alpha_table[pred_inds]  # alphas in this formula
                    act_alpha = torch.sigmoid(alpha)
                    tn_latent_neg_mask = torch.tensor(latent_neg_mask, dtype=torch.float)

                    for ind in inds:
                        val = torch.abs(1 - torch.tensor(ind, dtype=torch.float) - act_alpha)
                        val = torch.abs(tn_latent_neg_mask - val)
                        cartesian_prod[tuple(ind)] *= torch.max(torch.cat([val, obs_score], dim=0))

                else:

                    if sum(observed_neg_mask) == 0:
                        cartesian_prod[tuple(latent_neg_mask)] = 0.0

            weight_grad = closed_wolrd_potentials

            return weight_grad

    def gen_index(self, facts, predicates, dataset):
        rel2idx = dict()
        idx_rel = 0
        for rel in sorted(predicates.keys()):
            if rel not in rel2idx:
                rel2idx[rel] = idx_rel
                idx_rel += 1
        idx2rel = dict(zip(rel2idx.values(), rel2idx.keys()))

        ent2idx = dict()
        idx_ent = 0
        for type_name in sorted(dataset.const_sort_dict.keys()):
            for const in dataset.const_sort_dict[type_name]:
                ent2idx[const] = idx_ent
                idx_ent += 1
        idx2ent = dict(zip(ent2idx.values(), ent2idx.keys()))

        node2idx = ent2idx.copy()
        idx_node = len(node2idx)
        for rel in sorted(facts.keys()):
            for fact in sorted(list(facts[rel])):
                val, args = fact
                if (rel, args) not in node2idx:
                    node2idx[(rel, args)] = idx_node
                    idx_node += 1
        idx2node = dict(zip(node2idx.values(), node2idx.keys()))

        return ent2idx, idx2ent, rel2idx, idx2rel, node2idx, idx2node

    def gen_edge_type(self):
        edge_type2idx = dict()
        num_args_set = set()
        for rel in self.PRED_DICT:
            num_args = self.PRED_DICT[rel].num_args
            num_args_set.add(num_args)
        idx = 0
        for num_args in sorted(list(num_args_set)):
            for pos_code in product(['0', '1'], repeat=num_args):
                if '1' in pos_code:
                    edge_type2idx[(0, ''.join(pos_code))] = idx
                    idx += 1
                    edge_type2idx[(1, ''.join(pos_code))] = idx
                    idx += 1
        return edge_type2idx

    def gen_graph(self, facts, predicates, dataset):
        """
            generate directed knowledge graph, where each edge is from subject to object
        :param facts:
            dictionary of facts
        :param predicates:
            dictionary of predicates
        :param dataset:
            dataset object
        :return:
            graph object, entity to index, index to entity, relation to index, index to relation
        """

        # build bipartite graph (constant nodes and hyper predicate nodes)
        g = nx.Graph()
        ent2idx, idx2ent, rel2idx, idx2rel, node2idx, idx2node = self.gen_index(facts, predicates, dataset)

        edge_type2idx = self.gen_edge_type()

        for node_idx in idx2node:
            g.add_node(node_idx)

        for rel in facts.keys():
            for fact in facts[rel]:
                val, args = fact
                fact_node_idx = node2idx[(rel, args)]
                for arg in args:
                    pos_code = ''.join(['%d' % (arg == v) for v in args])
                    g.add_edge(fact_node_idx, node2idx[arg],
                               edge_type=edge_type2idx[(val, pos_code)])
        return g, edge_type2idx, ent2idx, idx2ent, rel2idx, idx2rel, node2idx, idx2node

    def prepare_node_feature(self, graph, transductive=True):
        if transductive:
            node_feat = torch.zeros(graph.num_nodes,  # for transductive GCN
                                    graph.num_ents + graph.num_rels)

            const_nodes = []
            for i in graph.idx2node:
                if isinstance(graph.idx2node[i], str):  # const (entity) node
                    const_nodes.append(i)
                    node_feat[i][i] = 1
                elif isinstance(graph.idx2node[i], tuple):  # fact node
                    rel, args = graph.idx2node[i]
                    node_feat[i][graph.num_ents + graph.rel2idx[rel]] = 1
        else:
            node_feat = torch.zeros(graph.num_nodes, 1 + graph.num_rels)  # for inductive GCN
            const_nodes = []
            for i in graph.idx2node:
                if isinstance(graph.idx2node[i], str):  # const (entity) node
                    node_feat[i][0] = 1
                    const_nodes.append(i)
                elif isinstance(graph.idx2node[i], tuple):  # fact node
                    rel, args = graph.idx2node[i]
                    node_feat[i][1 + graph.rel2idx[rel]] = 1

        return node_feat, torch.LongTensor(const_nodes)


class MLP(nn.Module):
    def __init__(self, input_size, num_layers, hidden_size, output_size):
        super(MLP, self).__init__()

        self.input_linear = nn.Linear(input_size, hidden_size)

        self.hidden = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.hidden.append(nn.Linear(hidden_size, hidden_size))

        self.output_linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h = F.relu(self.input_linear(x))

        for layer in self.hidden:
            h = F.relu(layer(h))

        output = self.output_linear(h)

        return output
