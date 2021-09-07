import copy
import dgl
import numpy as np
import torch as th
from tqdm import tqdm
import torch.nn as nn
import torch
import torch.nn.functional as F
from . import BaseFlow, register_flow
from ..tasks import build_task
from ..utils import extract_embed, get_nodes_dict
from collections.abc import Mapping
from ..models import build_model
from ..layers.EmbedLayer import HeteroEmbedLayer
from ..layers.HeteroLinear import HeteroFeature
from ..sampler.negative_sampler import Uniform_exclusive

@register_flow("recommendation")
class Recommendation(BaseFlow):
    """
    Recommendation flows.
    """

    def __init__(self, args):
        super(Recommendation, self).__init__(args)

        self.target_link = self.task.dataset.target_link
        self.loss_fn = self.task.get_loss_fn()
        self.args.has_feature = self.task.dataset.has_feature

        self.model = build_model(self.model_name).build_model_from_args(self.args, self.hg)
        self.model = self.model.to(self.device)

        self.evaluator = self.task.get_evaluator('mrr')

        self.optimizer = (
            th.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        )

    def preprocess(self):
        self.train_hg, self.val_hg, self.test_hg = self.task.get_idx()
        self.train_hg = self.train_hg.to(self.device)
        self.val_hg = self.val_hg.to(self.device)
        self.test_hg = self.test_hg.to(self.device)
        self.negative_graph = self.task.dataset.construct_negative_graph(self.train_hg).to(self.device)
        self.preprocess_feature()

    def train(self):
        self.preprocess()
        epoch_iter = tqdm(range(self.max_epoch))
        patience = 0
        best_score = 100
        best_model = copy.deepcopy(self.model)

        for epoch in tqdm(range(self.max_epoch), ncols=80):
            if self.args.mini_batch_flag:
                loss = self._mini_train_step()
            else:
                loss = self._full_train_setp()
            if epoch % 2 == 0:
                metric = self._test_step()
                epoch_iter.set_description(
                    f"Epoch: {epoch:03d}, NDCG: {metric:.4f}, Loss:{loss:.4f}"
                )
                if metric <= best_score:
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

    def loss_calculation(self, positive_graph, negative_graph, embedding):
        p_score = self.ScorePredictor(positive_graph, embedding).repeat_interleave(99)
        n_score = self.ScorePredictor(negative_graph, embedding)
        label = th.ones(len(n_score), device=self.device)
        loss = F.binary_cross_entropy_with_logits(p_score - n_score, label)
        return loss

    def ScorePredictor(self, hg, x):
        edge_subgraph = hg[self.target_link]
        with edge_subgraph.local_scope():
            for ntype in edge_subgraph.ntypes:
                edge_subgraph.nodes[ntype].data['x'] = x[ntype]
            edge_subgraph.apply_edges(
                dgl.function.u_dot_v('x', 'x', 'score'))
            score = edge_subgraph.edata['score']
            return score.squeeze()

    def regularization_loss(self, embedding):
        return th.mean(embedding.pow(2)) + th.mean(self.r_embedding.pow(2))

    def _full_train_setp(self):
        self.model.train()
        h_dict = self.input_feature()
        embedding = self.model(self.train_hg, h_dict)
        loss = self.loss_calculation(self.train_hg, self.negative_graph, embedding)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        #print(loss.item())
        return loss.item()

    def _test_step(self, split=None, logits=None):
        self.model.eval()
        with th.no_grad():
            h_dict = self.input_feature()
            embedding = self.model(self.train_hg, h_dict)

            p_score = self.ScorePredictor(self.pos_test_graph, embedding).unsqueeze(0)
            n_score = self.ScorePredictor(self.neg_test_graph, embedding)
            n_score = th.reshape(n_score, (99, -1))
            matrix = th.cat((p_score, n_score), 0).t().cpu().numpy()

            y_true = np.zeros_like(matrix)
            y_true[:, 0] = 1
            # _, indices = torch.sort(matrix, dim=0, descending=True)
            # rank = th.nonzero(indices == 0, as_tuple=True)[0]
            from sklearn.metrics import ndcg_score
            metric = ndcg_score(y_true, matrix, k=10)

        return metric
