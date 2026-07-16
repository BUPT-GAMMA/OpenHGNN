"""OpenHGNN trainer flow for HG-Adapter."""
import os
import torch
from sklearn.metrics import f1_score

from openhgnn.models import build_model
from openhgnn.trainerflow import register_flow
from openhgnn.trainerflow.base_flow import BaseFlow
from openhgnn.dataset.hg_adapter_dataset import HGAdapterArtifacts


@register_flow('hg_adapter_trainer')
class HGAdapterTrainer(BaseFlow):
    def __init__(self, args):
        super().__init__(args)
        self.category = self.task.dataset.category
        self.features = self.hg.nodes[self.category].data['h'].to(self.device)
        self.labels = self.task.get_labels().to(self.device).long()
        self.train_idx, self.val_idx, self.test_idx = [x.to(self.device) for x in self.task.dataset.get_split()]
        args.num_classes = self.task.dataset.num_classes
        self.artifacts = HGAdapterArtifacts.load(args.hg_adapter_root, self.task.dataset.source_name,
                                                 len(self.labels), args.embedding_dim, self.device)
        self.model = build_model(args.model).build_model_from_args(args, self.hg).to(self.device)
        self.optimizer = self.candidate_optimizer[args.optimizer](self.model.parameters(), lr=args.lr,
                                                                    weight_decay=args.weight_decay)

    def _evaluate(self, logits, index):
        truth = self.labels[index].detach().cpu()
        pred = logits[index].argmax(dim=1).detach().cpu()
        return {'Macro-F1': f1_score(truth, pred, average='macro', zero_division=0),
                'Micro-F1': f1_score(truth, pred, average='micro', zero_division=0)}

    def train(self):
        best_score, best_state, best_test, stale_epochs = -1.0, None, None, 0
        for epoch in range(self.max_epoch):
            self.model.train()
            logits, adjacency, reconstruction, hom_emb, het_emb = self.model(self.features, **self.artifacts)
            loss, terms = self.model.losses(logits, adjacency, reconstruction, hom_emb, het_emb, self.labels,
                                            self.train_idx, self.args.lambda_reconstruction,
                                            self.args.lambda_margin, self.args.margin)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.model.eval()
            with torch.no_grad():
                evaluated = self.model(self.features, **self.artifacts)[0]
                valid, test = self._evaluate(evaluated, self.val_idx), self._evaluate(evaluated, self.test_idx)
            self.logger.train_info('Epoch %d | loss %.4f | val Macro-F1 %.4f | test Macro-F1 %.4f' %
                                   (epoch, loss.item(), valid['Macro-F1'], test['Macro-F1']))
            if valid['Macro-F1'] > best_score:
                best_score, best_test = valid['Macro-F1'], test
                # ``detach().cpu()`` shares storage with CPU model parameters.
                # Clone each tensor so subsequent optimizer steps cannot mutate
                # the validation-best checkpoint.
                best_state = {key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()}
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= self.patience:
                    self.logger.info(f'Early stopping after {epoch + 1} epochs.')
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
            checkpoint = os.path.join(self.args.output_dir, f'HGAdapter_{self.args.dataset}.pt')
            torch.save(best_state, checkpoint)
            self.logger.info(f'Saved best HGAdapter checkpoint to {checkpoint}')
        return best_test
