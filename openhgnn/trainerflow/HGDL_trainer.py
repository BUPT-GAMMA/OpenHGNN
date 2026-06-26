"""HGDL trainerflow for OpenHGNN.

Full-batch trainer for the HGDL model (DBLP, ACM). Mirrors the upstream
training procedure (HGDL/main_HGDL.py + the validated external
openhgnn_HGDL/ fork): 2000 max epochs, early stopping on validation KL
with patience 50, Adam(lr, weight_decay=0), loss =
  KL(log(y+eps), y_true+eps) - gamma * Omega
where Omega = sum_{n, k} |nj - 1/k|.

This trainerflow drives the standard ``node_classification`` task, but
HGDL's labels are real-valued distributions rather than integer classes,
so the task's loss_fn (cross_entropy) and evaluate (argmax + F1) are
deliberately ignored. Loss and evaluation are computed here.

The trainerflow does NOT call ``self.preprocess()`` (which would attach
a HeteroFeature linear projection); the HGDL model has its own
``gcn1: Linear(in_dim, hidden_dim)`` that depends on a precise RNG
state, so we feed the dataset features straight through.
"""
from __future__ import annotations

import copy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import distance

from . import BaseFlow, register_flow
from ..models import build_model


# Same epsilon used by the upstream HGDL repo (model.py, main_HGDL.py).
EPS = 1e-9


def _ldl_metrics(pred, true):
    """Compute the six label-distribution learning metrics.

    Parameters
    ----------
    pred : torch.Tensor   (n, c) softmax outputs (already probabilities)
    true : torch.Tensor   (n, c) ground-truth distributions

    Returns
    -------
    dict with keys: KL, Cosine, Canberra, Chebyshev, Clark, Intersection
    """
    pred_t = pred.detach()
    true_t = true.detach()

    kl = F.kl_div(
        (pred_t + EPS).log(),
        true_t + EPS,
        reduction='batchmean',
    ).item()

    p = pred_t.cpu().numpy()
    t = true_t.cpu().numpy()
    n = p.shape[0]

    cos = can = che = 0.0
    for i in range(n):
        cos += distance.cosine(t[i], p[i])
        can += distance.canberra(t[i], p[i])
        che += distance.chebyshev(t[i], p[i])
    cos /= n
    can /= n
    che /= n

    # Clark and Intersection apply the same +EPS offset that upstream
    # score() applies before calling clark().
    p_eps = p + EPS
    t_eps = t + EPS
    clark = 0.0
    for i in range(n):
        num = (t_eps[i] - p_eps[i]) ** 2
        den = (t_eps[i] + p_eps[i]) ** 2
        clark += np.sqrt(np.sum(num / den))
    clark /= n

    inter = 0.0
    for i in range(n):
        inter += np.sum(np.minimum(p[i], t[i]))
    inter /= n

    return {
        'KL': kl,
        'Cosine': cos,
        'Canberra': can,
        'Chebyshev': che,
        'Clark': clark,
        'Intersection': inter,
    }


@register_flow('HGDL_trainer')
class HGDLTrainer(BaseFlow):
    """Full-batch trainer for HGDL.

    Reads scalar hyperparameters from ``args``:

    - lr, weight_decay, max_epoch, patience, evaluate_interval : standard
    - hidden_dim, attention_dim, num_heads, dropout : HGDL model
    - gamma : weight on the Omega regulariser (default 1e-4 to match the
              validated external run)
    """

    def __init__(self, args):
        super(HGDLTrainer, self).__init__(args)

        # Resolve task-derived attributes. BaseFlow has already built
        # self.task and self.hg for us.
        self.category = self.task.dataset.category
        self.num_classes = self.task.dataset.num_classes
        if not hasattr(args, 'out_dim') or args.out_dim != self.num_classes:
            self.logger.info('[HGDL] Setting args.out_dim to num_classes='
                             f'{self.num_classes}')
            args.out_dim = self.num_classes
        args.category = self.category
        args.in_dim = int(self.task.dataset.features.shape[1])
        args.num_heads = len(self.task.dataset.adj_list)  # auto: match meta-path count
        # ACM uses gamma=0 (no consistency loss); DBLP uses gamma=1e-4
        if hasattr(self.task.dataset, "dataset_name") and "acm" in self.task.dataset.dataset_name.lower():
            args.gamma = 0.0

        # Build the model. ``build_model_from_args`` derives num_nodes from
        # the graph; we then attach the precomputed adj_list manually,
        # because forward(hg, h_dict) leaves no room for it as an argument.
        self.model = build_model(self.model).build_model_from_args(
            self.args, self.hg).to(self.device)
        self.model.adj_list = [a.to(self.device)
                               for a in self.task.dataset.adj_list]

        # Cache the target-type features once on the device. HGDL does NOT
        # use OpenHGNN's shared HeteroFeature projection; the model has its
        # own gcn1 that consumes raw features. So we deliberately bypass
        # self.preprocess() / self.input_feature.
        self.features = self.task.dataset.features.to(self.device).float()
        # Labels: distributions, not integer classes. We pull them from the
        # dataset directly to be explicit about dtype/shape; the standard
        # task's get_labels also returns these since our HGDLDataset
        # overrides it.
        self.labels = self.task.dataset.labels.to(self.device).float()

        # Train/Val/Test splits (long tensors of node indices).
        self.train_idx, self.valid_idx, self.test_idx = self.task.get_split()
        self.train_idx = self.train_idx.to(self.device)
        self.valid_idx = self.valid_idx.to(self.device)
        self.test_idx = self.test_idx.to(self.device)

        # Optimiser: Adam, weight_decay defaults to 0 to match upstream.
        wd = float(getattr(args, 'weight_decay', 0.0))
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=args.lr, weight_decay=wd)

        # HGDL loss is KL divergence on label distributions. We do NOT
        # use self.task.get_loss_fn() because the standard
        # node_classification task returns cross_entropy / BCE.
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')

        # Omega regulariser weight; default 1e-4 to match the validated run
        # that produced KL = 0.0705 on DBLP.
        self.gamma = float(getattr(args, 'gamma', 1e-4))

        # Uniform reference distribution for Omega: shape (n, k).
        n = self.features.shape[0]
        k = self.model.num_heads
        self.uniform = torch.full((n, k), 1.0 / k, device=self.device)

    # ------------------------------------------------------------------
    def train(self):
        """Run training and return final test metrics."""
        max_epoch = int(getattr(self.args, 'max_epoch', 2000))
        patience_max = int(getattr(self.args, 'patience', 50))
        eval_every = int(getattr(self.args, 'evaluate_interval', 1))

        best_val = float('inf')
        best_state = None
        best_epoch = -1
        patience = 0
        start = time.time()

        for epoch in range(max_epoch):
            train_kl, omega = self._full_train_step()

            if epoch % eval_every == 0:
                val_kl = self._full_val_step()

                improved = val_kl < best_val
                if improved:
                    best_val = val_kl
                    best_epoch = epoch
                    best_state = copy.deepcopy(self.model.state_dict())
                    patience = 0
                else:
                    patience += 1

                self.logger.train_info(
                    f"Epoch {epoch:4d}  train_kl={train_kl:.4f}  "
                    f"val_kl={val_kl:.4f}  omega={omega:.2f}  "
                    f"patience={patience}/{patience_max}"
                    + ("  *" if improved else "")
                )

                if patience > patience_max:
                    self.logger.train_info(
                        f"[early stop] no val improvement for "
                        f"{patience_max} epochs (epoch {epoch})")
                    break

        elapsed = time.time() - start
        self.logger.train_info(
            f"[train done] best val_kl={best_val:.4f} at epoch {best_epoch}, "
            f"elapsed {elapsed:.1f}s")

        # Restore best weights for test.
        if best_state is not None:
            self.model.load_state_dict(best_state)

        # Final evaluation on the test set: six LDL metrics.
        test_metrics = self._full_test_step()
        self.logger.train_info('[Test Info] ' + '  '.join(
            f"{k}={v:.4f}" for k, v in test_metrics.items()))
        return dict(metric=test_metrics, epoch=best_epoch)

    # ------------------------------------------------------------------
    def _forward(self):
        """Run the model once and return ``(probs, nj)``.

        ``probs`` is (n, c) row-stochastic; ``nj`` is (n, k), the
        per-metapath attention used in the Omega regulariser.
        """
        h_dict = {self.category: self.features}
        out = self.model(self.hg, h_dict)
        probs = out[self.category]
        nj = self.model.last_attn_nj
        return probs, nj

    def _full_train_step(self):
        self.model.train()
        self.optimizer.zero_grad()
        probs, nj = self._forward()

        kl = self.kl_loss(
            (probs[self.train_idx] + EPS).log(),
            self.labels[self.train_idx] + EPS,
        )
        omega = torch.sum(torch.abs(nj - self.uniform))
        # Upstream loss form: minimise (kl - gamma * omega), i.e. push
        # Omega up to keep per-node metapath weights away from uniform.
        loss = kl - self.gamma * omega
        loss.backward()
        self.optimizer.step()
        return kl.item(), omega.item()

    def _full_val_step(self):
        self.model.eval()
        with torch.no_grad():
            probs, _ = self._forward()
            val_kl = self.kl_loss(
                (probs[self.valid_idx] + EPS).log(),
                self.labels[self.valid_idx] + EPS,
            ).item()
        return val_kl

    def _full_test_step(self):
        self.model.eval()
        with torch.no_grad():
            probs, _ = self._forward()
            return _ldl_metrics(probs[self.test_idx],
                                self.labels[self.test_idx])
