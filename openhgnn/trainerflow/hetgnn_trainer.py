import dgl
import torch as th
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader
from ..models import build_model
from . import BaseFlow, register_flow
from ..sampler.HetGNN_sampler import SkipGramBatchSampler, HetGNNCollator, NeighborSampler, hetgnn_graph
from ..utils import EarlyStopping


@register_flow("hetgnntrainer")
class HetGNNTrainer(BaseFlow):
    """SkipGram flows.

    Supported Model: HetGNN
    Supported Dataset：Academic4HetGNN
    Dataset description can be found in HetGNN paper.
    The trainerflow supports node classification and author link prediction.

    """

    def __init__(self, args):
        super(HetGNNTrainer, self).__init__(args)

        self.category = self.task.dataset.category
        self.model = build_model(self.model).build_model_from_args(self.args, self.hg).to(self.device)

        self.optimizer = (
            th.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        )

    def preprocess(self):

        if self.args.mini_batch_flag:
            if self.args.model == 'HetGNN':
                hetg = hetgnn_graph(self.hg, self.args.dataset)
                self.hg = self.hg.to('cpu')
                self.het_graph = hetg.get_hetgnn_graph(self.args.rw_length, self.args.rw_walks, self.args.rwr_prob).to('cpu')

                batch_sampler = SkipGramBatchSampler(self.hg, self.args.batch_size, self.args.window_size)
                neighbor_sampler = NeighborSampler(self.het_graph, self.hg.ntypes, batch_sampler.num_nodes, self.args.device)
                collator = HetGNNCollator(neighbor_sampler, self.hg)
                dataloader = DataLoader(
                    batch_sampler,
                    collate_fn=collator.collate_train,
                    num_workers=self.args.num_workers)
                self.dataloader_it = iter(dataloader)
                self.hg = self.hg.to(self.args.device)
                self.het_graph = self.het_graph.to(self.args.device)

        return

    def train(self):
        self.preprocess()
        stopper = EarlyStopping(self.args.patience, self._checkpoint)
        epoch_iter = tqdm(range(self.max_epoch))
        for epoch in epoch_iter:
            if self.args.mini_batch_flag:
                loss = self._mini_train_step()
            else:
                loss = self._full_train_setp()
                
            if epoch % self.evaluate_interval == 0:
                early_stop = stopper.loss_step(loss, self.model)
                metrics = self._test_step()
                self.logger.train_info(f"Epoch{epoch}, train loss:{loss:.4f}" + self.logger.metric2str(metrics))
                if early_stop:
                    self.logger.train_info(f"Early Stop!\tEpoch:{epoch}")
                    break
                
        stopper.load_model(self.model)
        metrics = self._test_step()
        return metrics

    def _full_train_setp(self):
        self.model.train()
        negative_graph = self.construct_negative_graph()
        x = self.model(self.het_graph)[self.category]
        loss = self.loss_calculation(self.ScorePredictor(self.hg, x), self.ScorePredictor(negative_graph, x))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _mini_train_step(self, ):
        self.model.train()
        all_loss = 0
        for batch_id in range(self.args.batches_per_epoch):
            positive_graph, negative_graph, blocks = next(self.dataloader_it)
            blocks = [b.to(self.device) for b in blocks]
            positive_graph = positive_graph.to(self.device)
            negative_graph = negative_graph.to(self.device)
            # we need extract multi-feature
            x = self.model(blocks[0])
            loss = self.loss_calculation(self.ScorePredictor(positive_graph, x), self.ScorePredictor(negative_graph, x))
            all_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return all_loss/self.args.batches_per_epoch

    def loss_calculation(self, pos_score, neg_score):
        # an example hinge loss
        loss = []
        for i in pos_score:
            loss.append(F.logsigmoid(pos_score[i]))
            loss.append(F.logsigmoid(-neg_score[i]))
        loss = th.cat(loss)
        return -loss.mean()

    def ScorePredictor(self, edge_subgraph, x):
        with edge_subgraph.local_scope():
            edge_subgraph.ndata['x'] = x
            for etype in edge_subgraph.canonical_etypes:
                edge_subgraph.apply_edges(
                    dgl.function.u_dot_v('x', 'x', 'score'), etype=etype)
            return edge_subgraph.edata['score']

    def _test_step(self, logits=None):
        self.model.eval()
        with th.no_grad():
            h = self.model.extract_feature(self.hg, self.hg.ntypes)
            logits = logits if logits else self.model(self.het_graph, h)
            logits = logits[self.category].to('cpu')
            if self.args.task == 'node_classification':
                metric = self.task.downstream_evaluate(logits, 'f1_lr')
                return {'test': metric}
            elif self.args.task == 'link_prediction':
                metric = self.task.downstream_evaluate(logits, 'academic_lp')
                return {'test': metric}



