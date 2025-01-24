import torch as th
from tqdm import tqdm
from . import BaseFlow, register_flow
from ..models import build_model
from ..utils import EarlyStopping
from ..sampler.TransX_sampler import TransX_Sampler


@register_flow("TransX_trainer")
class TransXTrainer(BaseFlow):
    """TransX flows."""

    def __init__(self, args):
        super(TransXTrainer, self).__init__(args)

        self.args = args
        self.model_name = args.model
        self.device = args.device
        self.batch_size = args.batch_size
        self.neg_size = args.neg_size
        self.score_size = args.batch_size * (args.neg_size * 2 + 1)
        self.max_epoch = args.max_epoch
        self.margin = args.margin

        self.train_hg = self.task.get_train()
        self.model = build_model(self.model).build_model_from_args(self.args, self.hg)
        self.model = self.model.to(self.device)

        self.optimizer = self.candidate_optimizer[args.optimizer](self.model.parameters(),
                                                                  lr=args.lr, weight_decay=args.weight_decay)

        self.stopper = EarlyStopping(args.patience, self._checkpoint)
        self.task.ScorePredictor = self.model.forward  # new score prdictor here

        self.num_nodes = getattr(self.task.dataset, 'num_nodes', self.hg.num_nodes())
        self.num_rels = getattr(self.task.dataset, 'num_rels', self.hg.num_edges())

    def preprocess(self):
        self.load_from_pretrained()
        self.train_hg.to(self.device)
        self.train_sampler = TransX_Sampler(self.train_hg, self.args)
        self.node_range = th.arange(0, self.num_nodes).to(self.device)
        self.rel_range = th.arange(0, self.num_rels).to(self.device)

        if self.args.score_fn == 'transr':  # load transe data when training transr
            transe_state_dict = th.load(self.stopper.save_path.replace("TransR", "TransE"))
            self.model.n_emb.weight.data = transe_state_dict['n_emb.weight']
            self.model.r_emb.weight.data = transe_state_dict['r_emb.weight']
            print("load")

    def train(self):
        self.preprocess()
        epoch = self._train()
        self.stopper.load_model(self.model)
        if self.args.test_flag:
            test_matrix = self._test()
            return dict(metric=test_matrix, epoch=epoch)
        if self.args.prediction_flag:
            return self._pred_step()

    def _train(self):
        batch_num = self.train_sampler.batch_num
        for epoch in range(self.max_epoch):
            self.logger.info(f"[Train Info] epoch {epoch:03d}")
            self.model.train()
            loss_sum = 0
            iter_range = tqdm(range(batch_num), ncols=100)
            for iter in iter_range:
                self.optimizer.zero_grad()
                pos_g = self.train_sampler.get_pos()
                neg_g = self.train_sampler.get_neg()
                h_emb, r_emb, t_emb = th.cat((pos_g[0], neg_g[0]), -1), th.cat((pos_g[1], neg_g[1]), -1), th.cat(
                    (pos_g[2], neg_g[2]), -1)
                loss = self.loss_calculation(h_emb, r_emb, t_emb)
                loss.backward()
                self.optimizer.step()
                loss_sum += loss.item()

            self.logger.info(f"[Train Info] epoch {epoch:03d} loss: {loss_sum}")

            if epoch % self.evaluate_interval == 0:
                val_metric = self._test_step('valid')
                self.logger.info("[Evaluation metric] " + str(val_metric))  # out test result
                early_stop = self.stopper.loss_step(val_metric['valid']['MR'], self.model)  # less is better
                if early_stop:
                    self.logger.train_info(f'Early Stop!\tEpoch:{epoch:03d}.')
                    break
        if self.max_epoch == 0:
            return 0
        else:
            return epoch

    def loss_calculation(self, h_emb, r_emb, t_emb):
        score = self.task.ScorePredictor(h_emb, r_emb, t_emb)

        if score.size(dim=0) == self.score_size:
            batch_size = self.batch_size
        else:  # last batch of a round
            batch_size = score.size(dim=0) // (self.neg_size * 2 + 1)

        p_score = score[:batch_size]
        p_score = p_score.view(batch_size, 1)

        n_score = score[batch_size:]
        n_score_split = th.chunk(n_score, 2, dim=0)
        n_score_tail = n_score_split[0].view(batch_size, self.neg_size)  # change tail
        n_score_head = n_score_split[1].view(batch_size, self.neg_size)  # change head
        n_score = th.cat((n_score_head, n_score_tail), dim=1)

        loss = th.clamp(p_score - n_score + self.margin, min=0.0).mean()
        return loss

    def _test(self):
        test_metric = self._test_step('test')
        self.logger.info("[Test Info] " + str(test_metric))  # out test result
        return test_metric  # dict

    def _test_step(self, mode):
        self.model.eval()
        with th.no_grad():
            n_emb = th.arange(self.num_nodes)
            r_emb_pre = th.arange(self.num_rels)
            # n_emb, r_emb_pre, _ = self.model(self.node_range, self.rel_range, th.tensor(0))
            r_emb = {}
            for i in range(self.num_rels):
                r_emb[i] = r_emb_pre[i]
            return {mode: self.task.evaluate(n_emb, r_emb, mode)}

    def _pred_step(self):
        self.model.eval()
        with th.no_grad():
            return self.task.tranX_predict()
