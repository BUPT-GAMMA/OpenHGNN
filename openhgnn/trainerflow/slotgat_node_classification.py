import torch
from tqdm import tqdm

from ..models import build_model
from ..utils import EarlyStopping
from . import BaseFlow, register_flow


@register_flow("slotgat_node_classification")
class SlotGATNodeClassification(BaseFlow):
    def __init__(self, args):
        if args.task == 'slotgat_node_classification':
            args.task = 'node_classification'
        super(SlotGATNodeClassification, self).__init__(args)
        self.args.category = self.task.dataset.category
        self.category = self.args.category
        self.num_classes = self.task.dataset.num_classes
        self.args.num_classes = self.num_classes
        self.args.out_dim = self.num_classes

        self.features_list = [feature.to(self.device) for feature in self.task.dataset.features_list]
        self.e_feat = self.task.dataset.e_feat.to(self.device)
        self.hg = self.hg.to(self.device)

        self.args.in_dim = [feature.shape[1] for feature in self.features_list]
        self.args.num_ntype = len(self.features_list)
        self.args.num_etypes = int(self.e_feat.max().item()) + 1 if self.e_feat.numel() > 0 else 0

        self.model = build_model(self.model).build_model_from_args(self.args, self.hg).to(self.device)
        self.optimizer = self.candidate_optimizer[args.optimizer](
            self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

        self.train_idx, self.val_idx, self.test_idx = self.task.get_split()
        self.train_idx = self.train_idx.to(self.device)
        self.val_idx = self.val_idx.to(self.device)
        self.test_idx = self.test_idx.to(self.device)
        self.labels = self.task.get_labels().to(self.device)

    def train(self):
        stopper = EarlyStopping(self.args.patience, self._checkpoint)
        epoch_iter = tqdm(range(self.max_epoch), ncols=120)
        epoch = 0
        for epoch in epoch_iter:
            train_loss = self._full_train_step()
            if epoch % self.evaluate_interval == 0:
                modes = ['train', 'valid']
                if self.args.test_flag:
                    modes.append('test')
                metric_dict, losses = self._full_test_step(modes=modes)
                val_loss = losses['valid']
                self.logger.train_info(
                    f"Epoch: {epoch}, Train loss: {train_loss:.4f}, Valid loss: {val_loss:.4f}. "
                    + self.logger.metric2str(metric_dict)
                )
                if stopper.loss_step(val_loss, self.model):
                    self.logger.train_info('Early Stop!\tEpoch:' + str(epoch))
                    break

        best_model = stopper.load_model(self.model)
        if best_model is not None:
            self.model = best_model

        if self.args.test_flag:
            metric_dict, _ = self._full_test_step(modes=['valid', 'test'])
            self.logger.train_info('[Test Info]' + self.logger.metric2str(metric_dict))
            self.logger.info('trainerflow finished')
            return dict(metric=metric_dict, epoch=epoch)
        return None

    def _full_train_step(self):
        self.model.train()
        logits, _ = self.model(self.hg, self.features_list, self.e_feat)
        loss = self.loss_fn(logits[self.train_idx], self.labels[self.train_idx])
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _full_test_step(self, modes, logits=None):
        self.model.eval()
        with torch.no_grad():
            logits = logits if logits is not None else self.model(self.hg, self.features_list, self.e_feat)[0]
            masks = {}
            for mode in modes:
                if mode == 'train':
                    masks[mode] = self.train_idx
                elif mode == 'valid':
                    masks[mode] = self.val_idx
                elif mode == 'test':
                    masks[mode] = self.test_idx

            metric_dict = {mode: self.task.evaluate(logits, mode=mode) for mode in masks}
            loss_dict = {
                mode: self.loss_fn(logits[mask], self.labels[mask]).item()
                for mode, mask in masks.items()
            }
            return metric_dict, loss_dict

    def _mini_train_step(self):
        raise NotImplementedError('SlotGAT currently supports full-batch training only.')

    def _mini_test_step(self):
        raise NotImplementedError('SlotGAT currently supports full-batch training only.')
