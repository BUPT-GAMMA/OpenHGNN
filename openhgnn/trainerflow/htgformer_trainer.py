"""
HTGformer Trainer
==================
Paper: HTGformer: Heterogeneous Temporal Graph Transformer (SIGIR 2025)

Unified training flow for four datasets/tasks:
  - OGBN-MAG:  Link prediction (multi-sample, LinkPredictor, AUC/AP)
  - Aminer:    Link prediction (time-step traversal, LinkPredictor, AUC/AP)
  - YELP:      Node classification (full-graph, CrossEntropy, Macro-F1/Recall)
  - COVID-19:  Node regression (multi-sample, L1Loss/MAE)

Hyperparameters (paper Section 4.1.3):
  Adam, lr=5e-3 (Aminer/YELP: 1e-3), weight_decay=5e-4
  hidden_dim=64 (COVID-19: 8), max_epoch=500, early_stopping=25
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import statistics
from sklearn.metrics import (
    f1_score, recall_score, roc_auc_score, average_precision_score
)

try:
    from openhgnn.trainerflow import BaseFlow, register_flow
    HAS_OPENHGNN = True
except ImportError:
    HAS_OPENHGNN = False
    BaseFlow = object
    def register_flow(name):
        def decorator(cls): return cls
        return decorator


class LinkPredictor(nn.Module):
    """Link prediction head (paper Section 3.4 MLP)."""
    def __init__(self, n_inp):
        super().__init__()
        self.fc1 = nn.Linear(n_inp * 2, n_inp)
        self.fc2 = nn.Linear(n_inp, 1)

    def forward(self, pos_g, neg_g, h):
        """DGL graph interface (OGBN-MAG)."""
        with pos_g.local_scope(), neg_g.local_scope():
            pos_g.ndata['h'] = h; neg_g.ndata['h'] = h
            pos_g.apply_edges(lambda e: {'s': self.fc2(F.relu(self.fc1(
                torch.cat([e.src['h'], e.dst['h']], 1))))})
            neg_g.apply_edges(lambda e: {'s': self.fc2(F.relu(self.fc1(
                torch.cat([e.src['h'], e.dst['h']], 1))))})
            return pos_g.edata['s'], neg_g.edata['s']

    def forward_ids(self, src, dst, h):
        """Index-based interface (Aminer)."""
        x = torch.cat([h[src], h[dst]], dim=1)
        return self.fc2(F.relu(self.fc1(x)))


def _compute_loss(pos_score, neg_score, device):
    """Binary cross-entropy loss for link prediction (paper Eq.8)."""
    pred = torch.cat([pos_score.squeeze(), neg_score.squeeze()])
    label = torch.cat([torch.ones(pos_score.shape[0]),
                        torch.zeros(neg_score.shape[0])]).to(device)
    return F.binary_cross_entropy_with_logits(pred, label)


def _compute_metric(pos_score, neg_score):
    """Compute AUC and AP metrics."""
    pred = torch.cat([pos_score.squeeze(), neg_score.squeeze()]).detach().cpu().numpy()
    label = np.concatenate([np.ones(pos_score.shape[0]), np.zeros(neg_score.shape[0])])
    try:
        return roc_auc_score(label, pred), average_precision_score(label, pred)
    except ValueError:
        return 0.5, 0.5


@register_flow('htgformer_trainer')
class HTGformerTrainer(BaseFlow):
    """HTGformer unified trainer. Automatically selects training/evaluation mode based on dataset.task."""

    def __init__(self, args):
        self.args = args
        # Compatible with both main.py (-d sets args.dataset) and script (args.dataset_name)
        if not getattr(args, 'dataset_name', None):
            args.dataset_name = getattr(args, 'dataset', '')
        self.device = torch.device(getattr(args, 'device', 'cpu'))
        self.dataset = self._load_dataset()
        self.task = self.dataset.task
        self.category = self.dataset.category

    def _load_dataset(self):
        """Load dataset via OpenHGNN build_dataset entry point."""
        from openhgnn.dataset import build_dataset
        name = getattr(self.args, 'dataset_name', '').lower()
        if 'ogbn' in name or 'mag' in name:
            ds_name, task = 'ogbn_mag4HGformer', 'link_prediction'
        elif 'aminer' in name:
            ds_name, task = 'aminer4HGformer', 'link_prediction'
        elif 'yelp' in name:
            ds_name, task = 'yelp4HGformer', 'node_classification'
        elif 'covid' in name:
            ds_name, task = 'covid4HGformer', 'node_regression'
        else:
            raise ValueError(f"Unknown dataset: {name}")
        logger = getattr(self.args, 'logger', None)
        return build_dataset(ds_name, task, logger=logger, args=self.args)

    def _build_model(self):
        """Build HTGformer model with dataset-specific config."""
        from openhgnn.models.HTGformer import HTGformer
        ds = self.dataset
        hidden_dim = getattr(self.args, 'hidden_dim', 64)
        if 'covid' in getattr(self.args, 'dataset_name', '').lower():
            hidden_dim = 8  # Paper Section 4.1.3: d=8 for COVID-19
        if self.task == 'node_classification':
            out_dim = ds.num_classes
            num_ts = len(ds.graphs)
        elif self.task == 'link_prediction':
            out_dim = hidden_dim
            num_ts = getattr(self.args, 'time_window', 1)
        else:
            out_dim = 1
            num_ts = getattr(self.args, 'time_window', 7)
        return HTGformer(
            in_dim_dict=ds.in_dim_dict,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_heads=getattr(self.args, 'num_heads', 4),
            num_layers=getattr(self.args, 'num_layers', 2),
            dropout=getattr(self.args, 'dropout', 0.1),
            num_timestamps=num_ts,
            node_types=list(ds.in_dim_dict.keys()),
            category=self.category,
            use_llm=getattr(self.args, 'use_llm', False),
            llm_embed_path=getattr(self.args, 'llm_embed_path', None),
        )

    def _max_epoch(self):
        """Use OpenHGNN's standard max_epoch while accepting older max_epochs args."""
        return getattr(self.args, 'max_epoch', getattr(self.args, 'max_epochs', 500))

    # Main entry
    def train(self):
        """Run training with multiple repeats (default 5, paper Section 4.1.3)."""
        num_repeats = getattr(self.args, 'num_repeats', 5)
        all_results = []
        for rep in range(1, num_repeats + 1):
            torch.manual_seed(rep); np.random.seed(rep)
            print(f"\n{'='*50}\nRepeat {rep}/{num_repeats}\n{'='*50}")
            result = self._train_one_run()
            all_results.append(result)
        self._print_summary(all_results)
        return all_results

    def _train_one_run(self):
        """Dispatch to dataset-specific training loop."""
        name = getattr(self.args, 'dataset_name', '').lower()
        if 'ogbn' in name or 'mag' in name:
            return self._train_ogbn_mag()
        elif 'aminer' in name:
            return self._train_aminer()
        elif 'yelp' in name:
            return self._train_yelp()
        elif 'covid' in name:
            return self._train_covid()

    # OGBN-MAG
    def _train_ogbn_mag(self):
        ds, device = self.dataset, self.device
        hidden_dim = getattr(self.args, 'hidden_dim', 64)
        model = self._build_model().to(device)
        predictor = LinkPredictor(hidden_dim).to(device)
        params = list(model.parameters()) + list(predictor.parameters())
        optimizer = torch.optim.Adam(params, lr=getattr(self.args, 'lr', 5e-3),
                                      weight_decay=getattr(self.args, 'weight_decay', 5e-4))
        print(f"# params: {sum(p.numel() for p in params if p.requires_grad)}")
        best_val_auc, best_test_auc, best_test_ap, best_ep = 0, 0, 0, 0
        patience_cnt, patience = 0, getattr(self.args, 'patience', 25)
        for epoch in range(1, self._max_epoch() + 1):
            model.train(); predictor.train(); epoch_loss = 0
            for wg, fds, pos_g, neg_g in ds.train_samples:
                gd = [g.to(device) for g in wg]
                fd = [{k: v.to(device) for k, v in d.items()} for d in fds]
                h = model.link_prediction_forward(gd, fd)
                ps, ns = predictor(pos_g.to(device), neg_g.to(device), h)
                loss = _compute_loss(ps, ns, device)
                optimizer.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); epoch_loss += loss.item()
            val_auc, val_ap = self._eval_link_mag(model, predictor, ds.val_samples, device)
            test_auc, test_ap = self._eval_link_mag(model, predictor, ds.test_samples, device)
            if val_auc > best_val_auc:
                best_val_auc = val_auc; best_test_auc = test_auc
                best_test_ap = test_ap; best_ep = epoch; patience_cnt = 0
            else:
                patience_cnt += 1
            if epoch % 10 == 0 or epoch <= 5:
                print(f"Epoch {epoch:4d} | Loss:{epoch_loss/len(ds.train_samples):.4f} "
                      f"| Val AUC:{val_auc*100:.2f}% | Val AP:{val_ap*100:.2f}%")
            if patience_cnt >= patience: break
        print(f"Test AUC:{best_test_auc*100:.2f}%, Test AP:{best_test_ap*100:.2f}%")
        return {'AUC': best_test_auc * 100, 'AP': best_test_ap * 100}

    @torch.no_grad()
    def _eval_link_mag(self, model, predictor, samples, device):
        model.eval(); predictor.eval(); aucs, aps = [], []
        for wg, fds, pos_g, neg_g in samples:
            gd = [g.to(device) for g in wg]
            fd = [{k: v.to(device) for k, v in d.items()} for d in fds]
            h = model.link_prediction_forward(gd, fd)
            ps, ns = predictor(pos_g.to(device), neg_g.to(device), h)
            auc, ap = _compute_metric(ps, ns); aucs.append(auc); aps.append(ap)
        return np.mean(aucs), np.mean(aps)

    # Aminer (time_window=1, aligned with DHGAS default)
    def _train_aminer(self):
        ds, device = self.dataset, self.device
        hidden_dim = getattr(self.args, 'hidden_dim', 64)
        tw = getattr(self.args, 'time_window', 1)
        model = self._build_model().to(device)
        predictor = LinkPredictor(hidden_dim).to(device)
        params = list(model.parameters()) + list(predictor.parameters())
        optimizer = torch.optim.Adam(params, lr=getattr(self.args, 'lr', 1e-3),
                                      weight_decay=getattr(self.args, 'weight_decay', 5e-4))
        print(f"# params: {sum(p.numel() for p in params if p.requires_grad)}")
        best_val_auc, best_test_auc, best_test_ap, best_ep = 0, 0, 0, 0
        patience_cnt, patience = 0, getattr(self.args, 'patience', 50)
        for epoch in range(1, self._max_epoch() + 1):
            model.train(); predictor.train(); epoch_loss, nb = 0, 0
            for t in ds.train_ts:
                pos_src, pos_dst, neg_src, neg_dst = ds.coauthor_samples[t]
                ws = max(0, t + 1 - tw)
                gw = [ds.snapshots[i].to(device) for i in range(ws, t + 1)]
                fw = [{k: v.to(device) for k, v in ds.all_feat_dicts[i].items()} for i in range(ws, t + 1)]
                h = model.link_prediction_forward(gw, fw)
                ps = predictor.forward_ids(pos_src.to(device), pos_dst.to(device), h)
                ns = predictor.forward_ids(neg_src.to(device), neg_dst.to(device), h)
                loss = _compute_loss(ps, ns, device)
                optimizer.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); epoch_loss += loss.item(); nb += 1
            val_auc, val_ap = self._eval_link_aminer(model, predictor, ds, ds.val_ts, tw, device)
            test_auc, test_ap = self._eval_link_aminer(model, predictor, ds, ds.test_ts, tw, device)
            if val_auc > best_val_auc:
                best_val_auc = val_auc; best_test_auc = test_auc
                best_test_ap = test_ap; best_ep = epoch; patience_cnt = 0
            else:
                patience_cnt += 1
            if epoch % 10 == 0 or epoch <= 5:
                print(f"Epoch {epoch:4d} | Loss:{epoch_loss/max(nb,1):.4f} "
                      f"| Val AUC:{val_auc*100:.2f}% | Test AUC:{test_auc*100:.2f}%")
            if patience_cnt >= patience: break
        print(f"Test AUC:{best_test_auc*100:.2f}%, Test AP:{best_test_ap*100:.2f}%")
        return {'AUC': best_test_auc * 100, 'AP': best_test_ap * 100}

    @torch.no_grad()
    def _eval_link_aminer(self, model, predictor, ds, t_list, tw, device):
        model.eval(); predictor.eval(); aucs, aps = [], []
        for t in t_list:
            if t not in ds.coauthor_samples: continue
            pos_src, pos_dst, neg_src, neg_dst = ds.coauthor_samples[t]
            ws = max(0, t + 1 - tw)
            gw = [ds.snapshots[i].to(device) for i in range(ws, t + 1)]
            fw = [{k: v.to(device) for k, v in ds.all_feat_dicts[i].items()} for i in range(ws, t + 1)]
            h = model.link_prediction_forward(gw, fw)
            ps = predictor.forward_ids(pos_src.to(device), pos_dst.to(device), h)
            ns = predictor.forward_ids(neg_src.to(device), neg_dst.to(device), h)
            auc, ap = _compute_metric(ps, ns); aucs.append(auc); aps.append(ap)
        return (np.mean(aucs), np.mean(aps)) if aucs else (0.5, 0.5)

    # YELP
    def _train_yelp(self):
        ds, device = self.dataset, self.device
        model = self._build_model().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=getattr(self.args, 'lr', 1e-3),
                                      weight_decay=getattr(self.args, 'weight_decay', 5e-4))
        criterion = nn.CrossEntropyLoss()
        print(f"# params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
        graphs_w = [g.to(device) for g in ds.graphs]
        fds_w = [{k: v.to(device) for k, v in fd.items()} for fd in ds.feat_dicts]
        best_val_f1, best_test_f1, best_test_recall, best_ep = 0, 0, 0, 0
        patience_cnt, patience = 0, getattr(self.args, 'patience', 50)
        for epoch in range(1, self._max_epoch() + 1):
            model.train()
            logits = model(graphs_w, fds_w)
            loss = criterion(logits[ds.train_idx.to(device)], ds.labels[ds.train_idx].to(device))
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            val_f1, val_r = self._eval_nclf(model, graphs_w, fds_w, ds, ds.val_idx, device)
            test_f1, test_r = self._eval_nclf(model, graphs_w, fds_w, ds, ds.test_idx, device)
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1; best_test_f1 = test_f1
                best_test_recall = test_r; best_ep = epoch; patience_cnt = 0
            else:
                patience_cnt += 1
            if epoch % 10 == 0 or epoch <= 5:
                print(f"Epoch {epoch:4d} | Loss:{loss.item():.4f} "
                      f"| Val F1:{val_f1:.2f}% | Test F1:{test_f1:.2f}%")
            if patience_cnt >= patience: break
        print(f"Test F1:{best_test_f1:.2f}%, Recall:{best_test_recall:.2f}%")
        return {'Macro-F1': best_test_f1, 'Recall': best_test_recall}

    @torch.no_grad()
    def _eval_nclf(self, model, graphs_w, fds_w, ds, idx, device):
        model.eval()
        logits = model(graphs_w, fds_w)
        preds = logits[idx].argmax(dim=-1).cpu().numpy()
        true = ds.labels[idx].cpu().numpy()
        return (f1_score(true, preds, average='macro', zero_division=0) * 100,
                recall_score(true, preds, average='macro', zero_division=0) * 100)

    # COVID-19
    def _train_covid(self):
        ds, device = self.dataset, self.device
        model = self._build_model().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=getattr(self.args, 'lr', 5e-3),
                                      weight_decay=getattr(self.args, 'weight_decay', 5e-4))
        print(f"# params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
        best_val, best_test, best_ep = float('inf'), float('inf'), 0
        patience_cnt, patience = 0, getattr(self.args, 'patience', 25)
        idx = np.random.permutation(len(ds.train_samples))
        for epoch in range(1, self._max_epoch() + 1):
            model.train(); losses = []
            for i in idx:
                wg, fd, lg = ds.train_samples[i]
                gd = [g.to(device) for g in wg]
                fdd = [{k: v.to(device) for k, v in d.items()} for d in fd]
                pred = model(gd, fdd)
                label = lg.nodes['state'].data['feat'].to(device)
                loss = F.l1_loss(pred, label)
                optimizer.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); losses.append(loss.item())
            val_mae = self._eval_covid(model, ds.val_samples, device)
            test_mae = self._eval_covid(model, ds.test_samples, device)
            if val_mae < best_val:
                best_val = val_mae; best_test = test_mae; best_ep = epoch; patience_cnt = 0
            else:
                patience_cnt += 1
            if epoch % 10 == 0 or epoch <= 5:
                print(f"Epoch {epoch:4d} | Loss:{np.mean(losses):.2f} "
                      f"| Val MAE:{val_mae:.2f} | Test MAE:{test_mae:.2f}")
            if patience_cnt >= patience: break
        print(f"Test MAE: {best_test:.2f}")
        return {'MAE': best_test}

    @torch.no_grad()
    def _eval_covid(self, model, samples, device):
        model.eval(); maes = []
        for wg, fd, lg in samples:
            gd = [g.to(device) for g in wg]
            fdd = [{k: v.to(device) for k, v in d.items()} for d in fd]
            pred = model(gd, fdd)
            label = lg.nodes['state'].data['feat'].to(device)
            maes.append(F.l1_loss(pred, label).item())
        return np.mean(maes)

    # Summary
    def _print_summary(self, results):
        """Print aggregated results across all repeats."""
        print(f"\n{'='*50}")
        if self.task == 'link_prediction':
            aucs = [r['AUC'] for r in results]; aps = [r['AP'] for r in results]
            if len(aucs) > 1:
                print(f"AUC: {statistics.mean(aucs):.2f} +/- {statistics.stdev(aucs):.2f}%")
                print(f"AP:  {statistics.mean(aps):.2f} +/- {statistics.stdev(aps):.2f}%")
            else:
                print(f"AUC: {aucs[0]:.2f}%, AP: {aps[0]:.2f}%")
        elif self.task == 'node_classification':
            f1s = [r['Macro-F1'] for r in results]; rs = [r['Recall'] for r in results]
            if len(f1s) > 1:
                print(f"Macro-F1: {statistics.mean(f1s):.2f} +/- {statistics.stdev(f1s):.2f}%")
                print(f"Recall:   {statistics.mean(rs):.2f} +/- {statistics.stdev(rs):.2f}%")
            else:
                print(f"Macro-F1: {f1s[0]:.2f}%, Recall: {rs[0]:.2f}%")
        elif self.task == 'node_regression':
            maes = [r['MAE'] for r in results]
            if len(maes) > 1:
                print(f"MAE: {statistics.mean(maes):.2f} +/- {statistics.stdev(maes):.2f}")
            else:
                print(f"MAE: {maes[0]:.2f}")
        print(f"{'='*50}")
