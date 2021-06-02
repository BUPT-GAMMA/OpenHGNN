import dgl
import torch as th
from tqdm import tqdm
import torch.nn.functional as F
from ..models import build_model
from . import BaseFlow, register_flow
from ..tasks import build_task
from ..utils import EarlyStopping
from openhgnn.utils.sampler import get_epoch_samples

@register_flow("nshetrainer")
class NSHETrainer(BaseFlow):
    """NSHE flows.

    Supported Model: NSHE
    Supported Dataset：acm4NSHE
    The trainerflow supports node classification.

    """

    def __init__(self, args):
        super(NSHETrainer, self).__init__(args)

        self.args = args
        self.model_name = args.model
        self.device = args.device
        self.task = build_task(args)

        self.hg = self.task.get_graph().to(self.device)

        self.model = build_model(self.model_name).build_model_from_args(self.args, self.hg)

        self.optimizer = th.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        self.model = self.model.to(self.device)
        self.patience = args.patience
        self.max_epoch = args.max_epoch

    def preprocess(self):
        self.g = dgl.to_homogeneous(self.hg)
        self.pos_edges = self.g.edges()
        self.category = self.task.dataset.category
        return

    def train(self):
        self.preprocess()
        stopper = EarlyStopping(self.args.patience, self._checkpoint)
        epoch_iter = tqdm(range(self.max_epoch))
        for epoch in epoch_iter:
            if self.args.mini_batch_flag:
                loss = self._mini_train_step()
            else:
                neg_edges, ns_samples = get_epoch_samples(self.hg, epoch, self.args.dataset, self.args.num_ns_neg, self.args.device)
                loss = self._full_train_setp(neg_edges, ns_samples)
            print('Epoch{}: Loss:{:.4f}'.format(epoch, loss))
            self._test_step()
            early_stop = stopper.loss_step(loss, self.model)
            if early_stop:
                print('Early Stop!\tEpoch:' + str(epoch))
                break
        stopper.load_model(self.model)
        metrics = self._test_step()
        return dict(metrics=metrics)

    def _full_train_setp(self, neg_edges, ns_samples):
        self.model.train()

        self.optimizer.zero_grad()
        h = self.hg.ndata['h']
        node_emb, homo_h = self.model(self.hg, h)
        h_context = self.model.context_encoder(node_emb)
        p_list = self.pre_ns(ns_samples, node_emb, h_context, self.hg.ntypes)
        ns_prediction = th.sigmoid(th.cat([p for p in p_list])).flatten()
        # compute loss

        loss = self.loss_calculation(homo_h, neg_edges, ns_samples, ns_prediction)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _mini_train_step(self, ):
        pass

    def loss_calculation(self, homo_h, neg_edges, ns_samples, ns_prediction):
        pairwise_loss = self.cal_node_pairwise_loss(homo_h, self.pos_edges, neg_edges)

        ns_label = th.cat([ns['label'] for ns in ns_samples]).type(th.float32).to(self.args.device)
        BCE_loss = th.nn.BCELoss()
        cla_loss = BCE_loss(ns_prediction, ns_label)
        loss = pairwise_loss + cla_loss * self.args.beta
        return loss

    def cal_node_pairwise_loss(self, node_emd, edge, neg_edge):
        # cross entropy loss from LINE
        # pos loss
        inner_product = self.cal_inner_product(node_emd, edge)
        pos_loss = - th.mean(F.logsigmoid(inner_product))
        # neg loss
        inner_product = self.cal_inner_product(node_emd, neg_edge)
        neg_loss = - th.mean(F.logsigmoid(-1 * inner_product))
        loss = pos_loss + neg_loss
        return loss

    def pre_ns(self, ns_samples, h, h_context, ntypes):
        p_list = []
        for ns_type in ns_samples:
            target = ns_type['target_type']
            index_h = ns_type[target]
            h_tar_type = h[target]
            h_tar = h_tar_type[index_h]
            for type in ntypes:
                if type != target:
                    index_h = ns_type[type]
                    h_con_type = h_context[type]
                    h_con = h_con_type[index_h]
                    h_tar = th.cat((h_tar, h_con), dim=1)
            p = self.model.linear_classifier(target, h_tar)
            p_list.append(p)
        return p_list

    def cal_inner_product(self, node_emd, edge):
        emb_u_i = node_emd[edge[0]]
        emb_u_j = node_emd[edge[1]]
        inner_product = th.sum(emb_u_i * emb_u_j, dim=1)
        return inner_product

    def _test_step(self, logits=None):
        self.model.eval()
        with th.no_grad():
            h = self.hg.ndata['h']
            node_emb, homo_h = self.model(self.hg, h)
            logits = logits if logits else node_emb
            logits = logits[self.category].to('cpu')
            if self.args.task == 'node_classification':
                metric = self.task.evaluate(logits, 'f1_lr')
                return metric
            # elif self.args.task == 'link_prediction':
            #     metric = self.task.evaluate(logits, 'academic_lp')
            #     return metric




