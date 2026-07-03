import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from . import BaseFlow, register_flow
from ..models import build_model
from ..utils import EarlyStopping


@register_flow("slotgat_link_prediction")
class SlotGATLinkPrediction(BaseFlow):
    def __init__(self, args):
        if args.task == 'slotgat_link_prediction':
            args.task = 'link_prediction'
        super(SlotGATLinkPrediction, self).__init__(args)
        self.slotgat_dataset = self.task.dataset
        self.train_hg = self.slotgat_dataset.g
        self.args.in_dim = self.slotgat_dataset.in_dim
        self.args.num_ntype = self.slotgat_dataset.num_ntype
        self.args.num_etypes = self.slotgat_dataset.num_etypes
        self.args.num_classes = self.slotgat_dataset.num_classes
        self.args.out_dim = self.slotgat_dataset.num_classes
        self.model = build_model(self.model).build_model_from_args(self.args, self.train_hg).to(self.device)
        self.optimizer = self.candidate_optimizer[args.optimizer](
            self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        self.patience = args.patience
        self.max_epoch = args.max_epoch
        self.writer = SummaryWriter(f'./openhgnn/output/{self.model_name}/')

    def train(self):
        dataset = self.slotgat_dataset
        stopper = EarlyStopping(self.patience, self._checkpoint)
        best_epoch = 0
        for epoch in tqdm(range(self.max_epoch)):
            loss = self._full_train_step()
            if epoch % self.evaluate_interval == 0:
                val_metric = self._full_test_step('valid')
                self.logger.train_info(
                    f"Epoch: {epoch:03d}, train loss: {loss:.4f}, "
                    f"valid roc_auc: {val_metric['roc_auc']:.4f}, "
                    f"valid MRR: {val_metric['MRR']:.4f}"
                )
                self.writer.add_scalar('train_loss', loss, global_step=epoch)
                self.writer.add_scalar('val_roc_auc', val_metric['roc_auc'], global_step=epoch)
                early_stop = stopper.loss_step(val_metric['loss'], self.model)
                best_epoch = epoch
                if early_stop:
                    self.logger.train_info(f'Early Stop!\tEpoch:{epoch:03d}.')
                    break
        stopper.load_model(self.model)
        self.writer.close()
        test_metric = self._full_test_step('test') if self.args.test_flag else self._full_test_step('valid')
        self.logger.train_info(
            f"SlotGAT LP result: roc_auc: {test_metric['roc_auc']:.4f}, "
            f"MRR: {test_metric['MRR']:.4f}, loss: {test_metric['loss']:.4f}"
        )
        return dict(metric=test_metric, epoch=best_epoch, dataset=dataset.name)

    def _full_train_step(self):
        dataset = self.slotgat_dataset
        self.model.train()
        train_neg = dataset.sample_negative(dataset.train_pos)
        pos_head, pos_tail, neg_head, neg_tail, rel_ids = [], [], [], [], []
        for rel_id in dataset.test_types:
            pos_head.extend(dataset.train_pos[rel_id][0])
            pos_tail.extend(dataset.train_pos[rel_id][1])
            neg_head.extend(train_neg[rel_id][0])
            neg_tail.extend(train_neg[rel_id][1])
            rel_ids.extend([rel_id] * len(dataset.train_pos[rel_id][0]))
        if len(rel_ids) == 0:
            return 0.0

        pos_head = np.array(pos_head)
        pos_tail = np.array(pos_tail)
        neg_head = np.array(neg_head)
        neg_tail = np.array(neg_tail)
        rel_ids = np.array(rel_ids)
        order = torch.randperm(len(rel_ids)).cpu().numpy()
        batch_size = getattr(self.args, 'batch_size', len(rel_ids))
        total_loss, total_count = 0.0, 0
        for start in range(0, len(rel_ids), batch_size):
            idx = order[start:start + batch_size]
            left = np.concatenate([pos_head[idx], neg_head[idx]])
            right = np.concatenate([pos_tail[idx], neg_tail[idx]])
            mid = np.concatenate([rel_ids[idx], rel_ids[idx]])
            labels = torch.FloatTensor(
                np.concatenate([np.ones(len(idx)), np.zeros(len(idx))])
            ).to(self.device)
            scores = self.model.forward_lp(dataset.g, dataset.features_list, dataset.e_feat, left, right, mid)
            loss = F.binary_cross_entropy(scores, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * labels.shape[0]
            total_count += labels.shape[0]
        return total_loss / max(total_count, 1)

    def _mini_train_step(self):
        return self._full_train_step()

    def _full_test_step(self, split='valid'):
        dataset = self.slotgat_dataset
        self.model.eval()
        neigh, label = dataset.get_eval_data('valid' if split == 'valid' else 'test')
        all_scores, all_labels, all_left, all_right = [], [], [], []
        losses = []
        with torch.no_grad():
            for rel_id in dataset.test_types:
                left = np.array(neigh[rel_id][0])
                right = np.array(neigh[rel_id][1])
                if left.shape[0] == 0:
                    continue
                mid = np.full(left.shape[0], rel_id, dtype=np.int64)
                labels = torch.FloatTensor(label[rel_id]).to(self.device)
                scores = self.model.forward_lp(dataset.g, dataset.features_list, dataset.e_feat, left, right, mid)
                losses.append(F.binary_cross_entropy(scores, labels).item())
                all_scores.extend(scores.cpu().numpy().tolist())
                all_labels.extend(label[rel_id])
                all_left.extend(left.tolist())
                all_right.extend(right.tolist())
        metric = dataset.evaluate([all_left, all_right], all_scores, all_labels) if all_labels else {
            'roc_auc': 0.0,
            'MRR': 0.0,
        }
        metric['loss'] = float(np.mean(losses)) if losses else 0.0
        return metric

    def _mini_test_step(self, split='valid'):
        return self._full_test_step(split)
