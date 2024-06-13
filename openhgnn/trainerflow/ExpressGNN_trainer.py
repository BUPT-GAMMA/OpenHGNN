import torch
import torch.optim as optim
import networkx as nx
from itertools import product
from tqdm import tqdm
from itertools import chain
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import math
from collections import Counter
from . import BaseFlow, register_flow
from ..models import build_model
from ..utils import EarlyStopping


class KnowledgeGraph(object):
    def __init__(self, facts, predicates, dataset):
        self.dataset = dataset
        self.PRED_DICT = predicates
        self.graph, self.edge_type2idx, \
            self.ent2idx, self.idx2ent, self.rel2idx, self.idx2rel, \
            self.node2idx, self.idx2node = self.gen_graph(facts, dataset)

        self.num_ents = len(self.ent2idx)
        self.num_rels = len(self.rel2idx)

        self.num_nodes = len(self.graph.nodes())
        self.num_edges = len(self.graph.edges())

        x, y, v = zip(*sorted(self.graph.edges(data=True), key=lambda t: t[:2]))
        self.edge_types = [d['edge_type'] for d in v]
        self.edge_pairs = np.ndarray(shape=(self.num_edges, 2), dtype=np.int64)
        self.edge_pairs[:, 0] = x
        self.edge_pairs[:, 1] = y

        self.idx2edge = dict()
        idx = 0
        for x, y in self.edge_pairs:
            self.idx2edge[idx] = (self.idx2node[x], self.idx2node[y])
            idx += 1
            self.idx2edge[idx] = (self.idx2node[y], self.idx2node[x])
            idx += 1

    def gen_graph(self, facts, dataset):
        """
            generate directed knowledge graph, where each edge is from subject to object
        :param facts:
            dictionary of facts
        :param dataset:
            dataset object
        :return:
            graph object, entity to index, index to entity, relation to index, index to relation
        """

        # build bipartite graph (constant nodes and hyper predicate nodes)
        g = nx.Graph()
        ent2idx, idx2ent, rel2idx, idx2rel, node2idx, idx2node = self.gen_index(facts, dataset)

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

    def gen_index(self, facts, dataset):
        rel2idx = dict()
        idx_rel = 0
        for rel in sorted(self.PRED_DICT.keys()):
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


@register_flow("ExpressGNN_trainer")
class ExpressGNNTrainer(BaseFlow):

    def __init__(self, args):
        super(ExpressGNNTrainer, self).__init__(args)

        
        self.model_name = args.model
        self.device = args.device
        self.dataset = self.task.dataset
        self.args.rule_list = self.dataset.rule_ls
        self.kg = KnowledgeGraph(self.dataset.fact_dict, self.dataset.PRED_DICT, self.dataset)
        self.args.PRED_DICT = self.dataset.PRED_DICT
        self.model = build_model(self.model).build_model_from_args(self.args, self.kg).to(self.device)

        self.stopper = EarlyStopping(self.args.patience, self._checkpoint)
        self.scheduler = None
        self.pred_aggregated_hid_args = dict()
        self.preprocess()

    def preprocess(self):
        all_params = chain.from_iterable([self.model.parameters()])
        self.optimizer = optim.Adam(all_params, lr=self.args.learning_rate, weight_decay=self.args.l2_coef)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'max', factor=self.args.lr_decay_factor,
                                                              patience=self.args.lr_decay_patience,
                                                              min_lr=self.args.lr_decay_min)
        if self.args.no_train == 1:
            self.args.num_epochs = 0

        # for Freebase data
        if self.args.load_method == 1:
            # prepare data for M-step
            tqdm.write('preparing data for M-step...')
            pred_arg1_set_arg2 = dict()
            pred_arg2_set_arg1 = dict()
            pred_fact_set = dict()
            for pred in self.dataset.fact_dict_2:
                pred_arg1_set_arg2[pred] = dict()
                pred_arg2_set_arg1[pred] = dict()
                pred_fact_set[pred] = set()
                for _, args in self.dataset.fact_dict_2[pred]:
                    if args[0] not in pred_arg1_set_arg2[pred]:
                        pred_arg1_set_arg2[pred][args[0]] = set()
                    if args[1] not in pred_arg2_set_arg1[pred]:
                        pred_arg2_set_arg1[pred][args[1]] = set()
                    pred_arg1_set_arg2[pred][args[0]].add(args[1])
                    pred_arg2_set_arg1[pred][args[1]].add(args[0])
                    pred_fact_set[pred].add(args)
            grounded_rules = []
            for rule_idx, rule in enumerate(self.dataset.rule_ls):
                grounded_rules.append(set())
                body_atoms = []
                head_atom = None
                for atom in rule.atom_ls:
                    if atom.neg:
                        body_atoms.append(atom)
                    elif head_atom is None:
                        head_atom = atom
                # atom in body must be observed
                assert len(body_atoms) <= 2
                if len(body_atoms) > 0:
                    body1 = body_atoms[0]
                    for _, body1_args in self.dataset.fact_dict_2[body1.pred_name]:
                        var2arg = dict()
                        var2arg[body1.var_name_ls[0]] = body1_args[0]
                        var2arg[body1.var_name_ls[1]] = body1_args[1]
                        for body2 in body_atoms[1:]:
                            if body2.var_name_ls[0] in var2arg:
                                if var2arg[body2.var_name_ls[0]] in pred_arg1_set_arg2[body2.pred_name]:
                                    for body2_arg2 in pred_arg1_set_arg2[body2.pred_name][
                                        var2arg[body2.var_name_ls[0]]]:
                                        var2arg[body2.var_name_ls[1]] = body2_arg2
                                        grounded_rules[rule_idx].add(tuple(sorted(var2arg.items())))
                            elif body2.var_name_ls[1] in var2arg:
                                if var2arg[body2.var_name_ls[1]] in pred_arg2_set_arg1[body2.pred_name]:
                                    for body2_arg1 in pred_arg2_set_arg1[body2.pred_name][
                                        var2arg[body2.var_name_ls[1]]]:
                                        var2arg[body2.var_name_ls[0]] = body2_arg1
                                        grounded_rules[rule_idx].add(tuple(sorted(var2arg.items())))
            # Collect head atoms derived by grounded formulas
            self.grounded_obs = dict()
            self.grounded_hid = dict()
            self.grounded_hid_score = dict()
            for rule_idx in range(len(self.dataset.rule_ls)):
                rule = self.dataset.rule_ls[rule_idx]
                for var2arg in grounded_rules[rule_idx]:
                    var2arg = dict(var2arg)
                    head_atom = rule.atom_ls[-1]
                    assert not head_atom.neg  # head atom
                    pred = head_atom.pred_name
                    args = (var2arg[head_atom.var_name_ls[0]], var2arg[head_atom.var_name_ls[1]])
                    if args in pred_fact_set[pred]:
                        if (pred, args) not in self.grounded_obs:
                            self.grounded_obs[(pred, args)] = []
                        self.grounded_obs[(pred, args)].append(rule_idx)
                    else:
                        if (pred, args) not in self.grounded_hid:
                            self.grounded_hid[(pred, args)] = []
                        self.grounded_hid[(pred, args)].append(rule_idx)

            tqdm.write('observed: %d, hidden: %d' % (len(self.grounded_obs), len(self.grounded_hid)))

            # Aggregate atoms by predicates for fast inference
            pred_aggregated_hid = dict()
            for (pred, args) in self.grounded_hid:
                if pred not in pred_aggregated_hid:
                    pred_aggregated_hid[pred] = []
                if pred not in self.pred_aggregated_hid_args:
                    self.pred_aggregated_hid_args[pred] = []
                pred_aggregated_hid[pred].append((self.dataset.const2ind[args[0]], self.dataset.const2ind[args[1]]))
                self.pred_aggregated_hid_args[pred].append(args)
            self.pred_aggregated_hid_list = [[pred, pred_aggregated_hid[pred]] for pred in
                                             sorted(pred_aggregated_hid.keys())]

    def train(self):
        if self.args.load_method == 1:
            for current_epoch in range(self.args.num_epochs):
                num_batches = int(math.ceil(len(self.dataset.test_fact_ls) / self.args.batchsize))
                pbar = tqdm(total=num_batches)
                acc_loss = 0.0
                cur_batch = 0

                for samples_by_r, latent_mask_by_r, neg_mask_by_r, obs_var_by_r, neg_var_by_r in \
                        self.dataset.get_batch_by_q(self.args.batchsize):

                    node_embeds = self.model.gcn_forward(self.dataset)

                    loss = 0.0
                    r_cnt = 0
                    for ind, samples in enumerate(samples_by_r):
                        neg_mask = neg_mask_by_r[ind]
                        latent_mask = latent_mask_by_r[ind]
                        obs_var = obs_var_by_r[ind]
                        neg_var = neg_var_by_r[ind]

                        if sum([len(e[1]) for e in neg_mask]) == 0:
                            continue

                        potential, posterior_prob, obs_xent = self.model.posterior_forward(
                            [samples, neg_mask, latent_mask,
                             obs_var, neg_var],
                            node_embeds, fast_mode=True)
                        if self.args.no_entropy == 1:
                            entropy = 0
                        else:
                            entropy = compute_entropy(posterior_prob) / self.args.entropy_temp

                        loss += - (potential.sum() * self.dataset.rule_ls[ind].weight + entropy) / (
                                potential.size(0) + 1e-6) + obs_xent

                        r_cnt += 1

                    if r_cnt > 0:
                        loss /= r_cnt
                        acc_loss += loss.item()

                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                    pbar.update()
                    cur_batch += 1
                    pbar.set_description(
                        'Epoch %d, train loss: %.4f, lr: %.4g' % (
                            current_epoch, acc_loss / cur_batch, get_lr(self.optimizer)))
                # M-step: optimize the weights of logic rules
                with torch.no_grad():
                    posterior_prob = self.model.posterior_forward(self.pred_aggregated_hid_list, node_embeds,
                                                                  fast_inference_mode=True)
                    for pred_i, (pred, var_ls) in enumerate(self.pred_aggregated_hid_list):
                        for var_i, var in enumerate(var_ls):
                            args = self.pred_aggregated_hid_args[pred][var_i]
                            self.grounded_hid_score[(pred, args)] = posterior_prob[pred_i][var_i]

                    rule_weight_gradient = torch.zeros(len(self.dataset.rule_ls), device=self.args.device)
                    for (pred, args) in self.grounded_obs:
                        for rule_idx in set(self.grounded_obs[(pred, args)]):
                            rule_weight_gradient[rule_idx] += 1.0 - compute_MB_proba(self.dataset.rule_ls,
                                                                                     self.grounded_obs[(pred, args)])
                    for (pred, args) in self.grounded_hid:
                        for rule_idx in set(self.grounded_hid[(pred, args)]):
                            target = self.grounded_hid_score[(pred, args)]
                            rule_weight_gradient[rule_idx] += target - compute_MB_proba(self.dataset.rule_ls,
                                                                                        self.grounded_hid[(pred, args)])

                    for rule_idx, rule in enumerate(self.dataset.rule_ls):
                        rule.weight += self.args.learning_rate_rule_weights * rule_weight_gradient[rule_idx]
                        # print(self.dataset.rule_ls[rule_idx].weight, end=' ')
                pbar.close()
                # validation
                with torch.no_grad():
                    node_embeds = self.model.gcn_forward(self.dataset)

                    valid_loss = 0.0
                    cnt_batch = 0
                    for samples_by_r, latent_mask_by_r, neg_mask_by_r, obs_var_by_r, neg_var_by_r in \
                            self.dataset.get_batch_by_q(self.args.batchsize, validation=True):
                        loss = 0.0
                        r_cnt = 0
                        for ind, samples in enumerate(samples_by_r):
                            neg_mask = neg_mask_by_r[ind]
                            latent_mask = latent_mask_by_r[ind]
                            obs_var = obs_var_by_r[ind]
                            neg_var = neg_var_by_r[ind]

                            if sum([len(e[1]) for e in neg_mask]) == 0:
                                continue

                            valid_potential, valid_prob, valid_obs_xent = self.model.posterior_forward(
                                [samples, neg_mask, latent_mask,
                                 obs_var, neg_var],
                                node_embeds, fast_mode=True)

                            if self.args.no_entropy == 1:
                                valid_entropy = 0
                            else:
                                valid_entropy = compute_entropy(valid_prob) / self.args.entropy_temp

                            loss += - (valid_potential.sum() + valid_entropy) / (
                                    valid_potential.size(0) + 1e-6) + valid_obs_xent

                            r_cnt += 1

                        if r_cnt > 0:
                            loss /= r_cnt
                            valid_loss += loss.item()

                        cnt_batch += 1

                    tqdm.write('Epoch %d, valid loss: %.4f' % (current_epoch, valid_loss / cnt_batch))

                    should_stop = self.stopper.loss_step(valid_loss, self.model)
                    self.scheduler.step(valid_loss)

                    is_current_best = self.stopper.counter == 0
                    if is_current_best:
                        self.stopper.save_model(self.model)

                    should_stop = should_stop or (current_epoch + 1 == self.args.num_epochs)

                    if should_stop:
                        tqdm.write('Early stopping')
                        break

            # ======================= generate rank list =======================
            print("rank_list", current_epoch)
            node_embeds = self.model.gcn_forward(self.dataset)

            pbar = tqdm(total=len(self.dataset.test_fact_ls))
            pbar.write('\n' + '*' * 10 + ' Evaluation ' + '*' * 10)
            rrank = 0.0
            hits = 0.0
            cnt = 0

            rrank_pred = dict([(pred_name, 0.0) for pred_name in self.kg.PRED_DICT])
            hits_pred = dict([(pred_name, 0.0) for pred_name in self.kg.PRED_DICT])
            cnt_pred = dict([(pred_name, 0.0) for pred_name in self.kg.PRED_DICT])
            for pred_name, X, invX, sample in gen_eval_query(self.dataset, const2ind=self.kg.ent2idx):
                x_mat = np.array(X)
                invx_mat = np.array(invX)
                sample_mat = np.array(sample)

                tail_score, head_score, true_score = self.model.posterior_forward(
                    [pred_name, x_mat, invx_mat, sample_mat],
                    node_embeds,
                    batch_mode=True)

                rank = torch.sum(tail_score >= true_score).item() + 1
                rrank += 1.0 / rank
                hits += 1 if rank <= 10 else 0

                rrank_pred[pred_name] += 1.0 / rank
                hits_pred[pred_name] += 1 if rank <= 10 else 0

                rank = torch.sum(head_score >= true_score).item() + 1
                rrank += 1.0 / rank
                hits += 1 if rank <= 10 else 0

                rrank_pred[pred_name] += 1.0 / rank
                hits_pred[pred_name] += 1 if rank <= 10 else 0

                cnt_pred[pred_name] += 2
                cnt += 2

                pbar.update()
            pbar.close()
            self.logger.info('\ncomplete:\n mmr %.4f\n' % (rrank / cnt) + 'hits %.4f\n' % (hits / cnt))
            for pred_name in self.kg.PRED_DICT:
                if cnt_pred[pred_name] == 0:
                    continue
                self.logger.info('mmr %s %.4f\n' % (pred_name, rrank_pred[pred_name] / cnt_pred[pred_name]))
                self.logger.info('hits %s %.4f\n' % (pred_name, hits_pred[pred_name] / cnt_pred[pred_name]))


        # for Kinship / UW-CSE / Cora data
        elif self.args.load_method == 0:
            for current_epoch in range(self.args.num_epochs):
                pbar = tqdm(range(self.args.num_batches))
                acc_loss = 0.0

                for k in pbar:
                    node_embeds = self.model.gcn_forward(self.dataset)

                    batch_neg_mask, flat_list, batch_latent_var_inds, observed_rule_cnts, batch_observed_vars = self.dataset.get_batch_rnd(
                        observed_prob=self.args.observed_prob,
                        filter_latent=self.args.filter_latent == 1,
                        closed_world=self.args.closed_world == 1,
                        filter_observed=1)

                    posterior_prob = self.model.posterior_forward(flat_list, node_embeds)

                    if self.args.no_entropy == 1:
                        entropy = 0
                    else:
                        entropy = compute_entropy(posterior_prob) / self.args.entropy_temp

                    entropy = entropy.to('cpu')
                    posterior_prob = posterior_prob.to('cpu')

                    potential = self.model.mln_forward(batch_neg_mask, batch_latent_var_inds, observed_rule_cnts,
                                                       posterior_prob,
                                                       flat_list, batch_observed_vars)

                    self.optimizer.zero_grad()

                    loss = - (potential + entropy) / self.args.batchsize
                    acc_loss += loss.item()

                    loss.backward()

                    self.optimizer.step()

                    pbar.set_description('train loss: %.4f, lr: %.4g' % (acc_loss / (k + 1), get_lr(self.optimizer)))

                # test
                node_embeds = self.model.gcn_forward(self.dataset)
                with torch.no_grad():

                    posterior_prob = self.model.posterior_forward([(e[1], e[2]) for e in self.dataset.test_fact_ls],
                                                                  node_embeds)
                    posterior_prob = posterior_prob.to('cpu')

                    label = np.array([e[0] for e in self.dataset.test_fact_ls])
                    test_log_prob = float(
                        np.sum(np.log(np.clip(np.abs((1 - label) - posterior_prob.numpy()), 1e-6, 1 - 1e-6))))
                    auc_roc = roc_auc_score(label, posterior_prob.numpy())
                    auc_pr = average_precision_score(label, posterior_prob.numpy())

                    self.logger.info(
                        'Epoch: %d, train loss: %.4f, test auc-roc: %.4f, test auc-pr: %.4f, test log prob: %.4f' % (
                            current_epoch, acc_loss / self.args.num_batches, auc_roc, auc_pr, test_log_prob))
                    # tqdm.write(str(posterior_prob[:10]))

                # validation for early stop
                valid_sample = []
                valid_label = []
                for pred_name in self.dataset.valid_dict_2:
                    for val, consts in self.dataset.valid_dict_2[pred_name]:
                        valid_sample.append((pred_name, consts))
                        valid_label.append(val)
                valid_label = np.array(valid_label)

                valid_prob = self.model.posterior_forward(valid_sample, node_embeds)
                valid_prob = valid_prob.to('cpu')

                valid_log_prob = float(
                    np.sum(np.log(np.clip(np.abs((1 - valid_label) - valid_prob.numpy()), 1e-6, 1 - 1e-6))))

                # tqdm.write('epoch: %d, valid log prob: %.4f' % (current_epoch, valid_log_prob))
                #
                # should_stop = monitor.update(-valid_log_prob)
                # scheduler.step(valid_log_prob)
                #
                # is_current_best = monitor.cnt == 0
                # if is_current_best:
                #   savepath = joinpath(self.args.exp_path, 'saved_model')
                #   os.makedirs(savepath, exist_ok=True)
                #   torch.save(gcn.state_dict(), joinpath(savepath, 'gcn.model'))
                #   torch.save(posterior_model.state_dict(), joinpath(savepath, 'posterior.model'))
                #
                # should_stop = should_stop or (current_epoch + 1 == self.args.num_epochs)
                #
                # if should_stop:
                #   tqdm.write('Early stopping')
                #   break
            self.evaluate()

    def evaluate(self):
        # evaluation after training
        node_embeds = self.model.gcn_forward(self.dataset)
        with torch.no_grad():
            posterior_prob = self.model.posterior_forward([(e[1], e[2]) for e in self.dataset.test_fact_ls],
                                                          node_embeds)
            posterior_prob = posterior_prob.to('cpu')

            label = np.array([e[0] for e in self.dataset.test_fact_ls])
            test_log_prob = float(
                np.sum(np.log(np.clip(np.abs((1 - label) - posterior_prob.numpy()), 1e-6, 1 - 1e-6))))
            auc_roc = roc_auc_score(label, posterior_prob.numpy())
            auc_pr = average_precision_score(label, posterior_prob.numpy())

            self.logger.info(
                'test auc-roc: %.4f, test auc-pr: %.4f, test log prob: %.4f' % (auc_roc, auc_pr, test_log_prob))
        pass


def compute_entropy(posterior_prob):
    eps = 1e-6
    posterior_prob.clamp_(eps, 1 - eps)
    compl_prob = 1 - posterior_prob
    entropy = -(posterior_prob * torch.log(posterior_prob) + compl_prob * torch.log(compl_prob)).sum()
    return entropy


def compute_MB_proba(rule_ls, ls_rule_idx):
    rule_idx_cnt = Counter(ls_rule_idx)
    numerator = 0
    for rule_idx in rule_idx_cnt:
        weight = rule_ls[rule_idx].weight
        cnt = rule_idx_cnt[rule_idx]
        numerator += math.exp(weight * cnt)
    return numerator / (numerator + 1.0)


def get_lr(optimizer):
    return optimizer.state_dict()['param_groups'][0]['lr']


def gen_eval_query(dataset, const2ind=None, pickone=None):
    const_ls = dataset.const_sort_dict['type']

    toindex = lambda x: x
    if const2ind is not None:
        toindex = lambda x: const2ind[x]

    for val, pred_name, consts in dataset.test_fact_ls:
        c1, c2 = toindex(consts[0]), toindex(consts[1])

        if pickone is not None:
            if pred_name != pickone:
                continue

        X, invX = [], []
        for const in const_ls:

            if const not in dataset.ht_dict[pred_name][0][consts[0]]:
                X.append([c1, toindex(const)])
            if const not in dataset.ht_dict[pred_name][1][consts[1]]:
                invX.append([toindex(const), c2])

        yield pred_name, X, invX, [[c1, c2]]


class EarlyStopMonitor:

    def __init__(self, patience):
        self.patience = patience
        self.cnt = 0
        self.cur_best = float('inf')

    def update(self, loss):
        """

    :param loss:
    :return:
        return True if patience exceeded
    """
        if loss < self.cur_best:
            self.cnt = 0
            self.cur_best = loss
        else:
            self.cnt += 1

        if self.cnt >= self.patience:
            return True
        else:
            return False

    def reset(self):
        self.cnt = 0
        self.cur_best = float('inf')
