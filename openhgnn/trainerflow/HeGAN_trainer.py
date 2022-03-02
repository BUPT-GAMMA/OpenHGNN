import argparse
import copy
import dgl
import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F
from openhgnn.models import build_model
from . import BaseFlow, register_flow
from ..tasks import build_task
import random
import copy

class GraphSampler2:
    r"""
    A faster sampler than GraphSampler.
    First load graph data to self.hg_dict, then interate.
    """
    def __init__(self, hg, k):
        self.k = k
        self.ets = hg.canonical_etypes
        self.nt_et = {}
        for et in hg.canonical_etypes:
            if et[0] not in self.nt_et:
                self.nt_et[et[0]] = [et]
            else:
                self.nt_et[et[0]].append(et)

        self.hg_dict = {key: {} for key in hg.ntypes}
        for nt in hg.ntypes:
            for nid in range(hg.num_nodes(nt)):
                if nid not in self.hg_dict[nt]:
                    self.hg_dict[nt][nid] = {}
                for et in self.nt_et[nt]:
                    self.hg_dict[nt][nid][et] = hg.successors(nid, et)

    def sample_graph_for_dis(self):
        r"""
        sample three graphs from original graph.

        Note
        ------------
        pos_hg:
            Sampled graph from true graph distribution, that is from the original graph with real node and real relation.
        neg_hg1:
            Sampled graph with true nodes pair but wrong realtion.
        neg_hg2:
            Sampled graph with true scr nodes and realtion but wrong node embedding.
        """
        pos_dict = {}
        neg_dict1 = {}
        neg_dict2 = {}

        for nt in self.hg_dict.keys():
            for src in self.hg_dict[nt].keys():
                for i in range(self.k):
                    et = random.choice(self.nt_et[nt])
                    dst = random.choice(self.hg_dict[nt][src][et])
                    if et not in pos_dict:
                        pos_dict[et] = ([src], [dst])
                    else:
                        pos_dict[et][0].append(src)
                        pos_dict[et][1].append(dst)

                    wrong_et = random.choice(self.ets)
                    while wrong_et == et:
                        wrong_et = random.choice(self.ets)
                    wrong_et = (et[0], wrong_et[1], et[2])

                    if wrong_et not in neg_dict1:
                        neg_dict1[wrong_et] = ([src], [dst])
                    else:
                        neg_dict1[wrong_et][0].append(src)
                        neg_dict1[wrong_et][1].append(dst)

        pos_hg = dgl.heterograph(pos_dict, {nt: len(self.hg_dict[nt].keys()) for nt in self.hg_dict.keys()})
        pos_hg1 = dgl.heterograph(neg_dict1, {nt: len(self.hg_dict[nt].keys()) for nt in self.hg_dict.keys()})
        pos_hg2 = dgl.heterograph(pos_dict, {nt: len(self.hg_dict[nt].keys()) for nt in self.hg_dict.keys()})

        return pos_hg, pos_hg1, pos_hg2

    def sample_graph_for_gen(self):
        d = {}
        for nt in self.hg_dict.keys():
            for src in self.hg_dict[nt].keys():
                for i in range(self.k):
                    et = self.nt_et[nt][random.randint(0, len(self.nt_et[nt]) - 1)]
                    dst = self.hg_dict[nt][src][et][random.randint(0, len(self.hg_dict[nt][src][et]) - 1)]
                    if et not in d:
                        d[et] = ([src], [dst])
                    else:
                        d[et][0].append(src)
                        d[et][1].append(dst)

        return dgl.heterograph(d, {nt: len(self.hg_dict[nt].keys()) for nt in self.hg_dict.keys()})

class GraphSampler:
    def __init__(self, hg, k):
        self.nt_et = {}
        for et in hg.canonical_etypes:
            if et[0] not in self.nt_et:
                self.nt_et[et[0]] = [et]
            else:
                self.nt_et[et[0]].append(et)

        self.k = k
        self.hg = hg

    def sample_graph_for_dis(self):
        pos_dict = {}
        neg_dict1 = {}
        neg_dict2 = {}

        for nt in self.hg.ntypes:
            for src in self.hg.nodes(nt):
                for i in range(self.k):
                    et = self.nt_et[nt][random.randint(0, len(self.nt_et[nt])-1)]
                    dst = self.hg.successors(src, et)[random.randint(0, len(self.hg.successors(src, et))-1)]
                    if et not in pos_dict:
                        pos_dict[et] = ([src], [dst])
                    else:
                        pos_dict[et][0].append(src)
                        pos_dict[et][1].append(dst)

                    ets = copy.deepcopy(self.hg.etypes)
                    ets.remove(et[1])
                    wrong_et = (et[0], random.choice(ets), et[2])
                    if wrong_et not in neg_dict1:
                        neg_dict1[wrong_et] = ([src], [dst])
                    else:
                        neg_dict1[wrong_et][0].append(src)
                        neg_dict1[wrong_et][1].append(dst)

        neg_dict2 = copy.deepcopy(pos_dict)

        pos_hg = dgl.heterograph(pos_dict, {nt: self.hg.number_of_nodes(nt) for nt in self.hg.ntypes})
        pos_hg1 = dgl.heterograph(neg_dict1, {nt: self.hg.number_of_nodes(nt) for nt in self.hg.ntypes})
        pos_hg2 = dgl.heterograph(neg_dict2, {nt: self.hg.number_of_nodes(nt) for nt in self.hg.ntypes})

        return pos_hg, pos_hg1, pos_hg2

    def sample_graph_for_gen(self):
        d = {}
        for nt in self.hg.ntypes:
            for src in self.hg.nodes(nt):
                for i in range(self.k):
                    et = self.nt_et[nt][random.randint(0, len(self.nt_et[nt]) - 1)]  # random edge type
                    dst = self.hg.successors(src, et)[random.randint(0, len(self.hg.successors(src, et)) - 1)]
                    if et not in d:
                        d[et] = ([src], [dst])
                    else:
                        d[et][0].append(src)
                        d[et][1].append(dst)

        return dgl.heterograph(d, {nt: self.hg.number_of_nodes(nt) for nt in self.hg.ntypes})


@register_flow('HeGAN_trainer')
class HeGANTrainer(BaseFlow):
    """Node classification flows.
    Supported Model: HeGAN
    Supported Dataset：yelp
    The task is to classify the nodes of HIN(Heterogeneous Information Network).
    Note: If the output dim is not equal the number of classes, a MLP will follow the gnn model.
    """
    def __init__(self, args):
        super().__init__(args)

        self.num_classes = self.task.dataset.num_classes
        self.category = self.task.dataset.category

        self.hg = self.task.get_graph()
        self.model = build_model(self.model_name).build_model_from_args(self.args, self.hg)
        self.model = self.model.to(self.device)

        self.evaluator = self.task.evaluator.classification
        self.evaluate_interval = 1
        self.loss_fn = torch.nn.BCEWithLogitsLoss(reduction='sum')
        self.optim_dis = torch.optim.Adam(self.model.discriminator.parameters(), lr=args.lr_dis, weight_decay=args.wd_dis)
        self.optim_gen = torch.optim.Adam(self.model.generator.parameters(), lr=args.lr_gen, weight_decay=args.wd_gen)
        self.train_idx, self.val_idx, self.test_idx = self.task.get_idx()
        self.labels = self.task.get_labels().to(self.device)
        self.sampler = GraphSampler(self.hg, self.args.n_sample)
        self.sampler2 = GraphSampler2(self.hg, self.args.n_sample)

    def train(self):
        epoch_iter = tqdm(range(self.args.max_epoch))
        for epoch in epoch_iter:
            if self.args.mini_batch_flag:
                dis_loss, gen_loss = self._mini_train_step()
            else:
                dis_loss, gen_loss = self._full_train_step()

            dis_score, gen_score = self._test_step()

            print(epoch)
            print("discriminator:\n\tloss:{:.4f}\n\tmicro_f1: {:.4f},\n\tmacro_f1: {:.4f}".format(dis_loss, dis_score[0], dis_score[1]))
            print("generator:\n\tloss:{:.4f}\n\tmicro_f1: {:.4f},\n\tmacro_f1: {:.4f}".format(gen_loss, gen_score[0], gen_score[1]))

    def _mini_train_step(self):

        dis_loss, gen_loss = None, None
        return dis_loss, gen_loss

    def _full_train_step(self):
        r"""
        Note
        ----
        pos_loss:
            positive graph loss.
        neg_loss1:
            negative graph loss with wrong realtions.
        neg_loss2:
            negativa graph loss with wrong nodes embedding.
        """
        self.model.train()

        gen_loss = None
        dis_loss = None

        # discriminator step
        for _ in range(self.args.epoch_dis):
            # pos_hg, pos_hg1, pos_hg2 = self.sampler.sample_graph_for_dis()
            pos_hg, pos_hg1, pos_hg2 = self.sampler2.sample_graph_for_dis()
            pos_hg = pos_hg.to(self.device)
            pos_hg1 = pos_hg1.to(self.device)
            pos_hg2 = pos_hg2.to(self.device)
            noise_emb = {
                et: torch.tensor(np.random.normal(0.0, self.args.sigma, (pos_hg2.num_edges(et), self.args.emb_size)).astype('float32')).to(self.device)
                for et in pos_hg2.canonical_etypes
            }

            self.model.generator.assign_node_data(pos_hg2, None)
            self.model.generator.assign_edge_data(pos_hg2, None)
            generate_neighbor_emb = self.model.generator.generate_neighbor_emb(pos_hg2, noise_emb)
            pos_score, neg_score1, neg_score2 = self.model.discriminator(pos_hg, pos_hg1, pos_hg2, generate_neighbor_emb)
            pos_loss = self.loss_fn(pos_score, torch.ones_like(pos_score))
            neg_loss1 = self.loss_fn(neg_score1, torch.zeros_like(neg_score1))
            neg_loss2 = self.loss_fn(neg_score2, torch.zeros_like(neg_score2))
            dis_loss = pos_loss + neg_loss2 + neg_loss1

            self.optim_dis.zero_grad()
            dis_loss.backward()
            self.optim_dis.step()

        # generator step
        dis_node_emb, dis_relation_matrix = self.model.discriminator.get_parameters()
        for _ in range(self.args.epoch_gen):
            # gen_hg = self.sampler.sample_graph_for_gen()
            gen_hg = self.sampler2.sample_graph_for_gen()
            noise_emb = {
                # et: torch.tensor(np.random.normal(0.0, self.args.sigma, self.gen_hg.edata['e'][et].shape))
                et: torch.tensor(np.random.normal(0.0, self.args.sigma, (gen_hg.num_edges(et), self.args.emb_size)).astype('float32')).to(self.device)
                for et in gen_hg.canonical_etypes
            }
            gen_hg = gen_hg.to(self.device)
            score = self.model.generator(gen_hg, dis_node_emb, dis_relation_matrix, noise_emb)
            gen_loss = self.loss_fn(score, torch.ones_like(score))
            self.optim_gen.zero_grad()
            gen_loss.backward()
            self.optim_gen.step()

        return dis_loss.item(), gen_loss.item()

    def _test_step(self, split=None, logits=None):
        self.model.eval()
        self.model.generator.eval()
        self.model.discriminator.eval()

        with torch.no_grad():
            dis_emb = self.model.discriminator.nodes_embedding[self.category]
            gen_emb = self.model.generator.nodes_embedding[self.category]

            dis_metric = self.evaluator(dis_emb.cpu(), self.labels.cpu())
            gen_metric = self.evaluator(gen_emb.cpu(), self.labels.cpu())

            return dis_metric, gen_metric


