import dgl
import torch as th
import torch.nn.functional as F
from dgl.dataloading.negative_sampler import Uniform
from . import BaseTask, register_task
from ..dataset import build_dataset
from ..utils import Evaluator


@register_task("link_prediction")
class LinkPrediction(BaseTask):
    r"""
    Link prediction tasks.

    Attributes
    -----------
    dataset : NodeClassificationDataset
        Task-related dataset

    evaluator : Evaluator
        offer evaluation metric

    Methods
    ---------
    get_graph :
        return a graph
    get_loss_fn :
        return a loss function
    """
    def __init__(self, args):
        super(LinkPrediction, self).__init__()
        self.n_dataset = args.dataset
        self.dataset = build_dataset(args.dataset, 'link_prediction')
        # self.evaluator = Evaluator()
        self.train_hg, self.val_hg, self.test_hg = self.dataset.get_idx()
        self.val_hg = self.val_hg.to(args.device)
        self.test_hg = self.test_hg.to(args.device)
        self.evaluator = Evaluator(args.seed)
        if not hasattr(args, 'score_fn'):
            self.score_fn = 'distmult'
        else:
            self.score_fn = args.score_fn
        self.negative_sampler = Uniform(1)

    def get_graph(self):
        return self.dataset.g

    def get_loss_fn(self):
        return F.binary_cross_entropy_with_logits

    def get_evaluator(self, name):
        if name == 'acc':
            return self.evaluator.author_link_prediction
        elif name == 'mrr':
            return self.evaluator.mrr_
        elif name == 'academic_lp':
            return self.evaluator.author_link_prediction
        elif name == 'roc_auc':
            return self.evaluator.cal_roc_auc

    def evaluate(self, name, n_embedding, r_embedding=None, mode='test'):
        r"""

        Parameters
        ----------
        name: str
            the name of evaluation metric
        n_embedding: th.Tensor
            the embedding of nodes
        r_embedding: th.Tensor
            the embedding of relation types
        mode: str
            the evaluation mode, train/valid/test
        Returns
        -------

        """
        if name == 'acc':
            return self.evaluator.author_link_prediction
        elif name == 'mrr':

            return self.evaluator.mrr_(n_embedding['_N'], self.dict2emd(r_embedding), self.dataset.train_triplets,
                                       self.dataset.valid_triplets, self.dataset.test_triplets,
                                       hits=[1, 3, 10], eval_bz=100)
        elif name == 'academic_lp':
            return self.evaluator.author_link_prediction(n_embedding, self.dataset.train_batch, self.dataset.test_batch)
        elif name == 'roc_auc':
            if mode == 'test':
                eval_hg = self.test_hg
            elif mode == 'valid':
                eval_hg = self.val_hg
            else:
                raise ValueError('Mode error, supported test and valid.')
            negative_graph = self.construct_negative_graph(eval_hg)
            p_score = th.sigmoid(self.ScorePredictor(eval_hg, n_embedding, r_embedding))
            n_score = th.sigmoid(self.ScorePredictor(negative_graph, n_embedding, r_embedding))
            p_label = th.ones(len(p_score))
            n_label = th.zeros(len(n_score))
            return self.evaluator.cal_roc_auc(th.cat((p_label, n_label)).cpu(), th.cat((p_score, n_score)).cpu())
        else:
            return self.evaluator.link_prediction

    def get_batch(self):
        return self.dataset.train_batch, self.dataset.test_batch

    def get_train(self):
        return self.train_hg

    def get_labels(self):
        return self.dataset.get_labels()

    def dict2emd(self, r_embedding):
        r_emd = []
        for i in range(self.dataset.num_rels):
            r_emd.append(r_embedding[str(i)])
        return th.stack(r_emd).squeeze()

    def construct_negative_graph(self, hg):
        e_dict = {
            etype: hg.edges(etype=etype, form='eid')
            for etype in hg.canonical_etypes}
        neg_srcdst = self.negative_sampler(hg, e_dict)
        neg_pair_graph = dgl.heterograph(neg_srcdst,
                                         {ntype: hg.number_of_nodes(ntype) for ntype in hg.ntypes})
        return neg_pair_graph

    def ScorePredictor(self, g, n_embedding, r_embedding=None):
        if self.score_fn == 'dot-product':
            score = self.dot_product(g, n_embedding)
        elif self.score_fn == 'distmult':
            score = self.distmult(g, n_embedding, r_embedding)
        return score

    def dot_product(self, edge_subgraph, x):
        """

        Parameters
        ----------
        edge_subgraph: dgl.Heterograph
            the prediction graph
        x: dict[th.Tensor]
            the embedding dict

        Returns
        -------
        score: th.Tensor
            the prediction of the edges in edge_subgraph
        """

        with edge_subgraph.local_scope():
            for ntype in edge_subgraph.ntypes:
                edge_subgraph.nodes[ntype].data['x'] = x[ntype]
            for etype in edge_subgraph.canonical_etypes:
                edge_subgraph.apply_edges(
                    dgl.function.u_dot_v('x', 'x', 'score'), etype=etype)
            score = edge_subgraph.edata['score']
            if isinstance(score, dict):
                result = []
                for _, value in score.items():
                    result.append(value)
                score = th.cat(result)
            return score.squeeze()

    def distmult(self, edge_subgraph, x, r_embedding):
        """
        DistMult factorization (Yang et al. 2014) as the scoring function,
        which is known to perform well on standard link prediction benchmarks when used on its own.

        In DistMult, every relation r is associated with a diagonal matrix :math:`R_{r} \in \mathbb{R}^{d \times d}`
        and a triple (s, r, o) is scored as

        .. math::
            f(s, r, o)=e_{s}^{T} R_{r} e_{o}

        Parameters
        ----------
        edge_subgraph: dgl.Heterograph
            the prediction graph
        x: dict[th.Tensor]
            the embedding dict
        r_embedding: th.Tensor

        Returns
        -------
        score: th.Tensor
            the prediction of the edges in edge_subgraph
        """
        score_list = []
        with edge_subgraph.local_scope():
            for ntype in edge_subgraph.ntypes:
                edge_subgraph.nodes[ntype].data['x'] = x[ntype]
            for etype in edge_subgraph.canonical_etypes:
                e = r_embedding[etype[1]]
                n = edge_subgraph.num_edges(etype)
                if 1 == len(edge_subgraph.canonical_etypes):
                    edge_subgraph.edata['e'] = e.expand(n, -1)
                else:
                    edge_subgraph.edata['e'] = {etype: e.expand(n, -1)}
                edge_subgraph.apply_edges(
                    dgl.function.u_mul_e('x', 'e', 's'), etype=etype)
                edge_subgraph.apply_edges(
                    dgl.function.e_mul_v('s', 'x', 'score'), etype=etype)
                if 1 == len(edge_subgraph.canonical_etypes):
                    score = th.sum(edge_subgraph.edata['score'], dim=1)
                else:
                    score = th.sum(edge_subgraph.edata['score'].pop(etype), dim=1)
                # score = th.sum(th.mul(edge_subgraph.edata['score'].pop(etype), e), dim=1)
                score_list.append(score)
            return th.cat(score_list)

