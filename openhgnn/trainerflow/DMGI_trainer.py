import torch
from sklearn.metrics import f1_score
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import torch.nn as nn
from openhgnn.models import build_model
from openhgnn.models.DMGI import LogReg
from openhgnn.trainerflow import register_flow, BaseFlow
from openhgnn.utils import EarlyStopping


@register_flow("DMGI_trainer")
class DMGI_trainer(BaseFlow):

    def __init__(self, args):
        super(DMGI_trainer, self).__init__(args)

        # get category
        self.args.category = self.task.dataset.category
        self.category = self.args.category

        if hasattr(self.task.dataset, 'in_dim'):
            self.args.in_dim = self.task.dataset.in_dim
        else:
            self.args.in_dim = self.hg.ndata['h'][self.category].shape[1]
        # get category num_classes
        self.num_classes = self.task.dataset.num_classes
        self.args.num_classes = self.task.dataset.num_classes
        self.model = build_model(self.model_name).build_model_from_args(self.args, self.hg)
        self.model = self.model.to(self.device)
        
        self.optimizer = self.candidate_optimizer[args.optimizer](self.model.parameters(),
                                                                  lr=args.lr, weight_decay=args.weight_decay)

        self.train_idx, self.val_idx, self.test_idx = self.task.get_split()
        # get label
        self.labels = self.task.get_labels().to(self.device)
        # get category's numbers
        self.num_nodes = self.hg.num_nodes(self.category)
        self.isSemi = args.isSemi
        # a coefficient to Calculate semi
        self.sup_coef = args.sup_coef

    def preprocess(self):
        pass

    def train(self):
        stopper = EarlyStopping(self.patience)
        model = self.model
        epoch_iter = tqdm(range(self.max_epoch))
        for epoch in epoch_iter:
            '''use earlyStopping'''
            loss = self._full_train_setp()
            early_stop = stopper.loss_step(loss, model)
            self.logger.train_info(f"Epoch: {epoch:03d}, Loss: {loss:.4f}")

            if early_stop:
                self.logger.train_info(f'Early Stop!\tEpoch:{epoch}')
                break
        # Evaluation
        stopper.load_model(self.model)
        model.eval()
        self.evaluate(model.H.data.detach(),)

    def _full_train_setp(self):

        self.model.train()
        self.optimizer.zero_grad()
        lbl_1 = torch.ones(1, self.num_nodes)
        lbl_2 = torch.zeros(1, self.num_nodes)
        lbl = torch.cat((lbl_1, lbl_2), 1).to(self.args.device)

        result = self.model(self.hg)

        loss = self.calculate_J(result, lbl)

        loss.backward()
        self.optimizer.step()
        loss = loss.cpu()
        loss = loss.detach().numpy()
        return loss

    def _test_step(self, split=None, logits=None):
        pass

    def _mini_train_step(self, ):
        pass

    def loss_calculation(self, positive_graph, negative_graph, embedding):
        pass

    def calculate_J(self, result, lbl):
        r"""
            Two formulas to calculate the final objective :math:`\mathcal{J}`
            If isSemi = Ture, introduce a semi-supervised module into our framework that predicts the labels of labeled nodes from
            the consensus embedding Z. More precisely, we minimize the cross-entropy error over the labeled nodes:

            .. math::
              \begin{equation}
                \mathcal{J}_{\text {semi }}=\sum_{r \in \mathcal{R}} \mathcal{L}^{(r)}+\alpha \ell_{\mathrm{cs}}+\beta\|\Theta\|+\gamma \ell_{\text {sup }}
              \end{equation}

            Where :math:`\gamma` is  the coefficient of the semi-supervised module, the way to calculate :math:`\ell_{\text {sup }}` :

            .. math::
              \begin{equation}
                \ell_{\text {sup }}=-\frac{1}{\left|\mathcal{Y}_{L}\right|} \sum_{l \in \mathcal{Y}_{L}} \sum_{i=1}^{c} Y_{l i} \ln \hat{Y}_{l i}
              \end{equation}

            If isSemi = False:

            .. math::
              \begin{equation}
                \mathcal{J}=\sum_{r \in \mathcal{R}} \mathcal{L}^{(r)}+\alpha \ell_{\mathrm{cs}}+\beta\|\Theta\|^{2}
              \end{equation}

            Where :math:`\alpha` controls the importance of the consensus regularization,
            :math:`mathcal{L}^{(r)}`  is cross entropy.
            """
        logits = result['logits']

        xent = nn.CrossEntropyLoss()
        b_xent = nn.BCEWithLogitsLoss()
        xent_loss = None

        for idx, logit in enumerate(logits):
            logit = logit.unsqueeze(0)
            if xent_loss is None:
                xent_loss = b_xent(logit, lbl)
            else:
                xent_loss += b_xent(logit, lbl)

        loss = xent_loss
        reg_loss = result['reg_loss']
        loss += self.args.reg_coef * reg_loss

        if self.isSemi:
            sup = result['semi']
            semi_loss = xent(sup[self.train_idx], self.labels[self.train_idx])
            loss += self.sup_coef * semi_loss
        return loss

    def evaluate(self, embeds):
        hid_units = embeds.shape[2]

        xent = F.cross_entropy

        train_embs = embeds[0, self.train_idx]
        val_embs = embeds[0, self.val_idx]
        test_embs = embeds[0, self.test_idx]

        train_lbls = self.labels[self.train_idx]
        val_lbls = self.labels[self.val_idx]
        test_lbls = self.labels[self.test_idx]

        val_accs = [];test_accs = []
        val_micro_f1s = [];test_micro_f1s = []
        val_macro_f1s = [];test_macro_f1s = []
        for _ in range(50):
            log = LogReg(hid_units, self.num_classes)
            opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.0)
            log.to(self.device)

            accs = []
            micro_f1s = []
            macro_f1s = []
            macro_f1s_val = []  ##
            for iter_ in range(50):
                # train
                log.train()
                opt.zero_grad()

                logits = log(train_embs)
                loss = xent(logits, train_lbls)

                loss.backward()
                opt.step()

                # val
                logits = log(val_embs)
                preds = torch.argmax(logits, dim=1)

                val_acc = torch.sum(preds == val_lbls).float() / val_lbls.shape[0]
                val_f1_macro = f1_score(val_lbls.cpu(), preds.cpu(), average='macro')
                val_f1_micro = f1_score(val_lbls.cpu(), preds.cpu(), average='micro')

                val_accs.append(val_acc.item())
                val_macro_f1s.append(val_f1_macro)
                val_micro_f1s.append(val_f1_micro)

                # test
                logits = log(test_embs)
                preds = torch.argmax(logits, dim=1)

                test_acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
                test_f1_macro = f1_score(test_lbls.cpu(), preds.cpu(), average='macro')
                test_f1_micro = f1_score(test_lbls.cpu(), preds.cpu(), average='micro')

                test_accs.append(test_acc.item())
                test_macro_f1s.append(test_f1_macro)
                test_micro_f1s.append(test_f1_micro)

            max_iter = val_accs.index(max(val_accs))
            accs.append(test_accs[max_iter])

            max_iter = val_macro_f1s.index(max(val_macro_f1s))
            macro_f1s.append(test_macro_f1s[max_iter])
            macro_f1s_val.append(val_macro_f1s[max_iter]) ###

            max_iter = val_micro_f1s.index(max(val_micro_f1s))
            micro_f1s.append(test_micro_f1s[max_iter])

        self.logger.train_info("\t[Classification] Macro-F1: {:.4f} ({:.4f}) | Micro-F1: {:.4f} ({:.4f})".format(np.mean(macro_f1s),
                                                                                                    np.std(macro_f1s),
                                                                                                    np.mean(micro_f1s),
                                                                                                    np.std(micro_f1s)))






