import torch as th
from torch import nn
from tqdm import tqdm
import torch
from . import BaseFlow, register_flow
from ..models import build_model
from ..utils import extract_embed, EarlyStopping, get_nodes_dict
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from ..sampler import hde_sampler

@register_flow("hde_trainer")
class hde_trainer(BaseFlow):
    """
    HDE trainer flow.
    Supported Model: HDE
    Supported Dataset：imdb4hde
    The trainerflow supports link prediction task. It first calculates HDE for every node in the graph,
    and uses the HDE as a part of the initial feature for each node.
    And then it performs standard message passing and link prediction operations.
    Please notice that for different target node set, the HDE for each node can be different.
    For more details, please refer to the original paper: http://www.shichuan.org/doc/116.pdf
    """

    def __init__(self, args):
        super(hde_trainer, self).__init__(args)
        self.target_link = self.task.dataset.target_link
        self.loss_fn = self.task.get_loss_fn()
        self.args.out_node_type = self.task.dataset.out_ntypes
        self.type2idx = {'A': 0, 'B': 1}
        self.node_type = len(self.type2idx)
        self.num_fea = (self.node_type * (args.max_dist + 1)) * 2 + self.node_type
        self.sample_size = self.num_fea * (1 + args.num_neighbor + args.num_neighbor * args.num_neighbor)
        self.args.patience = 10
        args.input_dim = self.num_fea
        args.output_dim = args.emb_dim // 2

        self.model = build_model(self.model).build_model_from_args(self.args, self.hg).to(self.device)
        # initialization
        for m in self.model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
        self.loss_fn = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=1e-2)
        self.evaluator = roc_auc_score
        self.HDE_sampler = hde_sampler.HDESampler(self)
        self.HDE_sampler.dgl2nx()

        self.data_A_train, self.data_B_train, self.data_y_train, \
        self.val_batch_A_fea, self.val_batch_B_fea, self.val_batch_y, \
        self.test_batch_A_fea, self.test_batch_B_fea, self.test_batch_y = self.HDE_sampler.compute_hde(args)

    def train(self):
        epoch_iter = tqdm(range(self.max_epoch))
        stopper = EarlyStopping(self.args.patience, self._checkpoint)
        for epoch in tqdm(range(self.max_epoch), ncols=80):
            loss = self._mini_train_step()
            if epoch % 2 == 0:
                val_metric = self._test_step('valid')
                epoch_iter.set_description(
                    f"Epoch: {epoch:03d}, roc_auc: {val_metric:.4f}, Loss:{loss:.4f}"
                )
                early_stop = stopper.step_score(val_metric, self.model)
                if early_stop:
                    print('Early Stop!\tEpoch:' + str(epoch))
                    break
        print(f"Valid_score_ = {stopper.best_score: .4f}")
        stopper.load_model(self.model)

        test_auc = self._test_step(split="test")
        val_auc = self._test_step(split="valid")
        print(f"Test roc_auc = {test_auc:.4f}")
        return dict(Test_mrr=test_auc, Val_mrr=val_auc)

    def _mini_train_step(self,):
        self.model.train()
        all_loss = 0
        for (train_batch_A_fea, train_batch_B_fea, train_batch_y) in zip(self.data_A_train, self.data_B_train, self.data_y_train):
            # train
            self.model.train()
            train_batch_A_fea = torch.FloatTensor(train_batch_A_fea).to(self.device)
            train_batch_B_fea = torch.FloatTensor(train_batch_B_fea).to(self.device)
            train_batch_y = torch.LongTensor(train_batch_y).to(self.device)
            logits = self.model(train_batch_A_fea, train_batch_B_fea)
            loss = self.loss_fn(logits, train_batch_y.squeeze())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            all_loss += loss.item()
        return all_loss

    def _test_step(self, split=None, logits=None):
        self.model.eval()
        with th.no_grad():
            if split == 'valid':
                data_A_eval = self.val_batch_A_fea
                data_B_eval = self.val_batch_B_fea
                data_y_eval = self.val_batch_y
            elif split == 'test':
                data_A_eval = self.test_batch_A_fea
                data_B_eval = self.test_batch_B_fea
                data_y_eval = self.test_batch_y
            logits = self.model(data_A_eval, data_B_eval)
            pred = logits.argmax(dim=1)
        metric = self.evaluator(data_y_eval.cpu().numpy(), pred.cpu().numpy())

        return metric
