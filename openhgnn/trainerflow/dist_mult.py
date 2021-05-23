import copy
import dgl
import numpy as np
import torch as th
from tqdm import tqdm
import torch.nn as nn
from openhgnn.models import build_model
import torch.nn.functional as F
from . import BaseFlow, register_flow
from ..tasks import build_task
from ..utils import extract_embed
from collections.abc import Mapping


class NegativeSampler(object):
    def __init__(self, g, k):
        # caches the probability distribution
        self.weights = {
            etype: g.in_degrees(etype=etype).float() ** 0.75
            for etype in g.canonical_etypes
        }
        self.k = k

    def __call__(self, g, eids_dict):
        result_dict = {}
        for etype, eids in eids_dict.items():
            src, _ = g.find_edges(eids, etype=etype)
            src = src.repeat_interleave(self.k)
            dst = self.weights[etype].multinomial(len(src), replacement=True)
            result_dict[etype] = (src, dst)
        return result_dict

@register_flow("distmult")
class DistMult(BaseFlow):
    """Node classification flows."""

    def __init__(self, args):
        super(DistMult, self).__init__(args)

        self.args = args
        self.model_name = args.model
        self.device = args.device

        self.task = build_task(args)
        self.hg = self.task.get_graph().to(self.device)
        self.category = self.task.dataset.category
        self.loss_fn = self.task.get_loss_fn()
        self.model = build_model(self.model_name).build_model_from_args(self.args, self.hg)
        self.model = self.model.to(self.device)
        self.evaluator = self.task.get_evaluator('mrr')
        self.num_rels = self.task.dataset.num_rels
        if hasattr(self.model, 'r_embedding'):
            para = self.model.parameters()
            self.r_embedding = self.model.r_embedding[:self.num_rels]
        else:
            self.r_embedding = nn.Parameter(th.Tensor(self.num_rels, self.args.out_dim).to(self.device))
            nn.init.uniform_(self.r_embedding,a=-1,b=1)
            para = [{'params': self.model.parameters()}, {'params': self.r_embedding}]
        self.optimizer = (
            th.optim.Adam(para, lr=args.lr, weight_decay=args.weight_decay)
        )
        self.patience = args.patience
        self.max_epoch = args.max_epoch
        if self.args.mini_batch_flag:
            self.hg = self.hg.to('cpu')
            train_eid_dict = {
                etype: self.hg.edges(etype=etype, form='eid')
                for etype in self.hg.canonical_etypes}

            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(self.args.n_layers)
            self.dataloader = dgl.dataloading.EdgeDataLoader(
                self.hg, train_eid_dict, sampler, device=self.device,
                    negative_sampler=NegativeSampler(self.hg, 1), batch_size=100,
                shuffle=True, drop_last=False, num_workers=0
            )
        else:
            self.train_eid_dict = {
                etype: self.hg.edges(etype=etype, form='eid')
                for etype in self.hg.canonical_etypes}
            self.negative_sampler = NegativeSampler(self.hg, 10)

    def preprocess(self):
        self.test_dataset = self.task.dataset.get_triples('test_mask').to(self.device)
        self.val_dataset = self.task.dataset.get_triples('val_mask').to(self.device)
        self.train_dataset = self.task.dataset.get_triples('train_mask').to(self.device)
        return

    def train(self):
        self.preprocess()
        epoch_iter = tqdm(range(self.max_epoch))
        patience = 0
        best_score = 0
        best_model = copy.deepcopy(self.model)

        for epoch in tqdm(range(self.max_epoch), ncols=80):
            if self.args.mini_batch_flag:
                loss = self._mini_train_step()
            else:
                loss = self._full_train_setp()
            if epoch % 2 == 0:
                metric= self._test_step(split='train')
                epoch_iter.set_description(
                    f"Epoch: {epoch:03d}, Val-mrr: {metric:.4f}, Loss:{loss:.4f}"
                )
                if metric >= best_score:
                    best_score = metric
                    best_model = copy.deepcopy(self.model)
                    patience = 0
                else:
                    patience += 1
                    if patience == self.patience:
                        epoch_iter.close()
                        break
        print(f"Valid mrr = {best_score: .4f}")
        self.model = best_model
        test_mrr = self._test_step(split="test")
        val_mrr = self._test_step(split="val")
        print(f"Test mrr = {test_mrr:.4f}")
        return dict(Test_mrr=test_mrr, ValMrr=val_mrr)

    def _mini_train_step(self,):
        self.model.train()
        all_loss = 0
        for input_nodes, positive_graph, negative_graph, blocks in self.dataloader:
            blocks = [b.to(self.device) for b in blocks]
            positive_graph = positive_graph.to(self.device)
            negative_graph = negative_graph.to(self.device)
            if type(input_nodes) == th.Tensor:
                input_nodes = {self.category: input_nodes}
            input_features = extract_embed(self.model.embed_layer(), input_nodes)
            logits = self.model(blocks, input_features)[self.category]
            loss = self.loss_calculation(positive_graph, negative_graph, logits)
            all_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return all_loss

    def loss_calculation(self, positive_graph, negative_graph, logits):
        p_score = self.ScorePredictor(positive_graph, logits)
        p_label = th.ones(len(p_score), device=self.device)
        n_score = self.ScorePredictor(negative_graph, logits)
        n_label = th.zeros(len(n_score), device=self.device)
        loss = F.binary_cross_entropy_with_logits(th.cat((p_score, n_score)), th.cat((p_label, n_label)))
        return loss

    def ScorePredictor(self, edge_subgraph, x):
        score_list = []
        with edge_subgraph.local_scope():
            edge_subgraph.ndata['x'] = x
            for etype in edge_subgraph.canonical_etypes:
                e = self.r_embedding[int(etype[1])]
                n = edge_subgraph.num_edges(etype)
                edge_subgraph.edata['e'] = {etype: e.expand(n, -1)}
                edge_subgraph.apply_edges(
                    dgl.function.u_mul_e('x', 'e', 's'), etype=etype)
                edge_subgraph.apply_edges(
                    dgl.function.e_mul_v('s', 'x', 'score'), etype=etype)
                score = th.sum(edge_subgraph.edata['score'].pop(etype), dim=1)
                #score = th.sum(th.mul(edge_subgraph.edata['score'].pop(etype), e), dim=1)
                score_list.append(score)
            return th.cat(score_list)

    def ScorePredictor_(self, edge_subgraph, x):
        score_list = []
        with edge_subgraph.local_scope():
            edge_subgraph.ndata['x'] = x
            for etype in edge_subgraph.canonical_etypes:
                e = self.r_embedding[int(etype[1])]
                n = edge_subgraph.num_edges(etype)
                edge_subgraph.edata['e'] = {etype: e.expand(n, -1)}
                edge_subgraph.apply_edges(
                    dgl.function.u_add_e('x', 'e', 's'), etype=etype)
                edge_subgraph.apply_edges(
                    dgl.function.e_sub_v('s', 'x', 'score'), etype=etype)


                score = -th.norm(edge_subgraph.edata['score'].pop(etype), p=1, dim=1)
                #score = th.sum(th.mul(edge_subgraph.edata['score'].pop(etype), e), dim=1)
                score_list.append(score)
            return th.cat(score_list)

    def regularization_loss(self, embedding):
        return th.mean(embedding.pow(2)) + th.mean(self.r_embedding.pow(2))

    def construct_negative_graph(self,):

        neg_srcdst = self.negative_sampler(self.hg, self.train_eid_dict)
        if not isinstance(neg_srcdst, Mapping):
            assert len(self.hg.etypes) == 1, \
                'graph has multiple or no edge types; '\
                'please return a dict in negative sampler.'
            neg_srcdst = {self.hg.canonical_etypes[0]: neg_srcdst}
        # Get dtype from a tuple of tensors
        #dtype = F.dtype(list(neg_srcdst.values())[0][0])
        neg_edges = {
            etype: neg_srcdst.get(etype, (th.tensor([]), th.tensor([])))
            for etype in self.hg.canonical_etypes}
        neg_pair_graph = dgl.heterograph(
            neg_edges, {ntype: self.hg.number_of_nodes(ntype) for ntype in self.hg.ntypes})
        return neg_pair_graph

    def _full_train_setp(self):
        self.model.train()
        negative_graph = self.construct_negative_graph()
        #for _ in range(2000):
        logits = self.model(self.hg)[self.category]
        #reg_loss = self.regularization_loss(logits)
        loss = self.loss_calculation(self.hg, negative_graph, logits)
        self.optimizer.zero_grad()
        loss.backward()
        #th.nn.utils.clip_grad_norm_(list(self.model.parameters()) + [self.r_embedding], 1.0)
        th.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        print(loss.item())
        return loss.item()

    def _test_step(self, split=None, logits=None):
        self.model.eval()
        with th.no_grad():
            logits = logits if logits else self.model(self.hg)[self.category]
            metric = self.evaluator(logits, self.r_embedding, self.train_dataset, self.val_dataset, self.train_dataset, hits=[1], eval_p='raw')
        # if split == 'val':
        #     metric = self.evaluator(logits, self.r_embedding, self.val_dataset, hits=[1, 3, 10], eval_p='raw')
        # elif split == 'test':
        #     metric = self.evaluator(logits, self.r_embedding, self.test_dataset, hits=[1, 3, 10], eval_p='raw')
        # elif split == 'train':
        #     metric = self.evaluator(logits, self.r_embedding, self.train_dataset, hits=[1], eval_p='raw')
        return metric
