"""
LMSPS trainer flow: progressive sampling search followed by training
(NeurIPS 2024, `Long-range Meta-path Search through Progressive Sampling
<https://arxiv.org/abs/2409.12011>`__).

The flow runs the two stages of the paper:

1. **Search stage** (paper Section 5.1-5.2, Eq. 2-6).  A super-net is trained
   with a first-order approximation of the bilevel objective of Eq. 5: the
   network weights ``omega`` are updated on the training split and the
   architecture parameters ``alpha`` on the validation split, alternating.
   Every iteration samples ``M`` meta-paths from the dynamic search space of
   Eq. 3, and the temperature of the Gumbel-softmax of Eq. 2 is annealed.  The
   search ends with the sampling evaluation of Eq. 6, which draws candidate
   meta-path sets from the compact space ``S_V`` and keeps the one with the
   lowest validation loss.

2. **Training stage** (paper Eq. 7).  A target-net is built for the selected
   meta-path set ``S_M`` only and trained as usual.

Neighbour aggregation (Eq. 1) is a one-time preprocessing step shared by both
stages, exactly as in :class:`~openhgnn.trainerflow.SeHGNN_trainer`.

.. note::
    This flow currently covers the ``ogbn-mag`` pipeline without label
    propagation (``--label-feats``), which corresponds to the first command of
    the reference implementation and to the ``54.83`` entry of Table 1 of the
    paper.  The label-propagation variant (``+LP+MS``, Table 1 ``57.84``) and
    multi-stage self-training are left for a follow-up change.
"""

import gc
import os
import random
import time
import uuid

import dgl.function as fn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from openhgnn.models import build_model
from openhgnn.models.LMSPS import LMSPS_SearchNet
from . import BaseFlow, register_flow
from ..tasks import NodeClassification

# The task of OGBN-MAG predicts the label of the ``paper`` node type, whose
# feature key is ``P`` (see OGB_NodeClassification.mag4sehgnn).
TGT_TYPE = 'P'


@register_flow('LMSPS_trainer')
class LMSPStrainer(BaseFlow):
    r"""
    LMSPS trainer flow: search the meta-paths, then train on the searched set.

    Parameters
    ----------
    args : dict
        The arguments of the flow, see the ``[LMSPS]`` section of
        ``openhgnn/config.ini`` for the available options.
    """

    def __init__(self, args):
        super(LMSPStrainer, self).__init__(args)
        # str() because the same value can arrive as a string from config.ini
        # or as an int through Experiment(..., stages=2)
        args.stages = [int(item.strip()) for item in str(args.stages).split(',')]
        assert len(args.stages) == 1, \
            'LMSPS currently supports a single training stage; multi-stage ' \
            'self-training is a follow-up change'
        self.args = args
        self.flow = NodeClassification(args)

    # ------------------------------------------------------------------ #
    # entry point
    # ------------------------------------------------------------------ #
    def train(self):
        args = self.args
        if args.seed > 0:
            self.set_random_seed(args.seed)
        device = "cuda:{}".format(args.gpu) if not args.cpu else 'cpu'

        num_nodes = self.flow.dataset.SeHGNN_g.num_nodes(TGT_TYPE)
        n_classes = int(self.flow.labels.max()) + 1
        evaluator = self.flow.get_evaluator("acc")

        # ======= rearrange node idx so that train / valid / test are contiguous =======
        train_node_nums = len(self.flow.train_idx)
        valid_node_nums = len(self.flow.val_idx)
        test_node_nums = len(self.flow.test_idx)
        trainval_point = train_node_nums
        valtest_point = trainval_point + valid_node_nums
        total_num_nodes = train_node_nums + valid_node_nums + test_node_nums

        init2sort = torch.cat([self.flow.train_idx, self.flow.val_idx, self.flow.test_idx])
        sort2init = torch.argsort(init2sort)
        assert torch.all(self.flow.labels[init2sort][sort2init] == self.flow.labels)
        labels = self.flow.labels[init2sort]
        labels_cuda = labels.long().to(device)

        # ======= neighbour aggregation (paper Eq. 1), once for all meta-paths =======
        feats, feat_keys = self.propagate_features()
        feats = {k: v[init2sort] for k, v in feats.items()}
        print(f'K = {len(feat_keys)} candidate meta-paths: {feat_keys}')
        gc.collect()

        # ======= stage 1: progressive sampling search (Eq. 2 - Eq. 6) =======
        # `meta_paths` skips the search and trains on a given meta-path set,
        # which is how the searched meta-paths reported in the paper (Table 9)
        # are reused without paying for the search again -- the same split as
        # the `--arch` flag of the reference implementation.
        if args.meta_paths:
            selected = [p.strip() for p in args.meta_paths.split(',') if p.strip()]
            unknown = [p for p in selected if p not in feat_keys]
            assert not unknown, \
                f'unknown meta-paths {unknown}; available: {feat_keys}'
            print('Stage 1/2 skipped: training on the {} given meta-paths {}'.format(
                len(selected), selected))
        else:
            selected = self.search(feats, feat_keys, labels_cuda, train_node_nums,
                                   valid_node_nums, device, evaluator)
            print(f'searched meta-path set S_M ({len(selected)}): {selected}')
            assert len(selected) == min(args.ns, args.num_final, len(feat_keys)), \
                'the sampling evaluation must return min(M, V, K) meta-paths'

        # ======= stage 2: train the target-net on S_M (Eq. 7) =======
        checkpt_folder = f'./openhgnn/output/LMSPS/{args.dataset}/'
        if not os.path.exists(checkpt_folder):
            os.makedirs(checkpt_folder)
        checkpt_file = checkpt_folder + uuid.uuid4().hex
        print(checkpt_file)
        best_epoch, best_val_acc, best_test_acc = self.train_target(
            feats, selected, labels, labels_cuda, train_node_nums,
            valid_node_nums, test_node_nums, trainval_point, valtest_point,
            total_num_nodes, num_nodes, n_classes, device, evaluator,
            checkpt_file)

        # ======= report =======
        print('=' * 60)
        print('searched meta-paths: {}'.format(selected))
        print('Best Epoch {}, Val {:.4f}, Test {:.4f}'.format(
            best_epoch, best_val_acc * 100, best_test_acc * 100))
        print('=' * 60)

    # ------------------------------------------------------------------ #
    # stage 1: progressive sampling search (paper Eq. 2 - Eq. 6)
    # ------------------------------------------------------------------ #
    def search(self, feats, feat_keys, labels_cuda, train_node_nums,
               valid_node_nums, device, evaluator):
        r"""
        Run the progressive sampling search and return the meta-path set
        ``S_M`` chosen by the sampling evaluation (paper Algorithm 1).
        """
        args = self.args
        print('=' * 60)
        print('Stage 1/2: progressive sampling search (paper Eq. 2 - Eq. 6)')
        print('=' * 60)

        net = LMSPS_SearchNet(
            feat_keys=feat_keys,
            data_size={k: feats[k].size(-1) for k in list(feat_keys) + [TGT_TYPE] if k in feats},
            hidden=args.hidden,
            nclass=int(labels_cuda.max()) + 1,
            tgt_key=TGT_TYPE,
            dropout=args.search_dropout,
            input_drop=args.search_input_drop,
            num_final=args.num_final,
            num_sampled=args.ns,
            residual=args.residual,
        ).to(device)
        print(net)
        print("# Params:", self.get_n_params(net))

        # Eq. 5: omega and alpha are optimized separately, so that the gradient
        # of the training loss never reaches the architecture parameters.
        alpha_id = id(net.alpha)
        optimizer_w = torch.optim.Adam(
            [p for p in net.parameters() if id(p) != alpha_id],
            lr=args.lr, weight_decay=args.weight_decay)
        optimizer_a = torch.optim.Adam(
            [net.alpha], lr=args.alr, weight_decay=args.weight_decay)

        loss_fcn = nn.CrossEntropyLoss()
        scalar = torch.cuda.amp.GradScaler() if args.amp else None

        train_loader = torch.utils.data.DataLoader(
            torch.arange(train_node_nums), batch_size=args.batch_size,
            shuffle=True, drop_last=False)
        val_loader = torch.utils.data.DataLoader(
            torch.arange(train_node_nums, train_node_nums + valid_node_nums),
            batch_size=args.batch_size, shuffle=True, drop_last=False)

        epochs = args.stages[0]
        val_iter = self.batches(val_loader)
        for epoch in range(epochs):
            gc.collect()
            torch.cuda.empty_cache()
            start = time.time()

            # lambda warmup: the full search space is kept for the first
            # `warmup_epochs` epochs, then lambda decays linearly to 0 (Eq. 3)
            if epoch < args.warmup_epochs:
                lam = 1.0
            else:
                lam = 1.0 - (epoch - args.warmup_epochs) / \
                    max(epochs - 1 - args.warmup_epochs, 1)
                lam = max(lam, 0.0)

            # Eq. 2: anneal the Gumbel temperature from tau_max to tau_min.
            # It is set before sampling because the path strength of Eq. 3 --
            # and therefore the dynamic search space -- depends on tau.
            net.set_tau(args.tau_max - (args.tau_max - args.tau_min) * epoch / max(epochs - 1, 1))
            # Eq. 4: activate M meta-paths out of the dynamic search space
            epoch_sampled = net.epoch_sample(lam)

            loss_w, loss_a, acc_w, acc_a = self.search_step(
                net, train_loader, val_iter, loss_fcn, optimizer_w, optimizer_a,
                epoch_sampled, evaluator, device, feats, labels_cuda, scalar)
            end = time.time()

            print("Epoch {}, Time(s): {:.4f}, C={}, sampled {} paths\n"
                  "\testimated train loss {:.4f}, acc {:.4f}\n"
                  "\testimated val   loss {:.4f}, acc {:.4f}".format(
                      epoch, end - start,
                      int(np.ceil(lam * (len(feat_keys) - args.num_final))) + args.num_final,
                      len(epoch_sampled), loss_w, acc_w * 100, loss_a, acc_a * 100),
                  flush=True)

        # ======= sampling evaluation (Eq. 6) =======
        print('Sampling evaluation: drawing {} meta-path sets from the compact '
              'space S_V (V={}) and keeping the one with the lowest validation '
              'loss'.format(args.repeat, args.num_final))
        net.eval()
        best_loss, best_index = None, None
        for i in range(args.repeat):
            index_sampled = net.sample_eval()
            val_loss = self.eval_search_net(net, index_sampled, feats, labels_cuda,
                                            train_node_nums, train_node_nums + valid_node_nums,
                                            device, loss_fcn)
            if best_loss is None or val_loss < best_loss:
                best_loss, best_index = val_loss, index_sampled
                print('\t[{}] new best val loss {:.6f} with {}'.format(i, val_loss, index_sampled),
                      flush=True)
        print('best val loss {:.6f}'.format(best_loss))

        return sorted(feat_keys[i] for i in best_index)

    def search_step(self, net, train_loader, val_iter, loss_fcn, optimizer_w,
                    optimizer_a, epoch_sampled, evaluator, device, feats,
                    labels_cuda, scalar=None):
        r"""
        One search iteration: first-order approximation of the bilevel problem
        of Eq. 5, i.e. ``omega`` is updated on a training batch and ``alpha`` on
        a validation batch, alternating.
        """
        net.train()
        total_loss_w, total_loss_a, iter_num = 0., 0., 0
        y_true, y_pred, val_y_true, val_y_pred = [], [], [], []

        for batch in train_loader:
            # a fresh validation batch every iteration, as in the reference
            # implementation (a shuffled loader yields a random subset)
            val_batch = next(val_iter)
            batch_feats = {k: x[batch].to(device) for k, x in feats.items()}
            batch_y = labels_cuda[batch]
            val_batch_feats = {k: x[val_batch].to(device) for k, x in feats.items()}
            val_batch_y = labels_cuda[val_batch]

            # ---- omega: training loss on the training split ----
            optimizer_w.zero_grad()
            if scalar is not None:
                with torch.cuda.amp.autocast():
                    output = net(epoch_sampled, batch_feats)
                    loss_w = loss_fcn(output, batch_y)
                scalar.scale(loss_w).backward()
                scalar.step(optimizer_w)
                scalar.update()
            else:
                output = net(epoch_sampled, batch_feats)
                loss_w = loss_fcn(output, batch_y)
                loss_w.backward()
                optimizer_w.step()

            # ---- alpha: validation loss on the validation split ----
            # zero_grad also drops the alpha gradient contributed by the
            # training loss above, which is what decouples the two levels
            optimizer_a.zero_grad()
            if scalar is not None:
                with torch.cuda.amp.autocast():
                    val_output = net(epoch_sampled, val_batch_feats)
                    loss_a = loss_fcn(val_output, val_batch_y)
                scalar.scale(loss_a).backward()
                scalar.step(optimizer_a)
                scalar.update()
            else:
                val_output = net(epoch_sampled, val_batch_feats)
                loss_a = loss_fcn(val_output, val_batch_y)
                loss_a.backward()
                optimizer_a.step()

            y_true.append(batch_y.cpu().to(torch.long))
            y_pred.append(output.argmax(dim=-1, keepdim=True).cpu())
            val_y_true.append(val_batch_y.cpu().to(torch.long))
            val_y_pred.append(val_output.argmax(dim=-1, keepdim=True).cpu())
            total_loss_w += loss_w.item()
            total_loss_a += loss_a.item()
            iter_num += 1

        acc_w = evaluator(torch.cat(y_true, dim=0), torch.cat(y_pred, dim=0))
        acc_a = evaluator(torch.cat(val_y_true, dim=0), torch.cat(val_y_pred, dim=0))
        return total_loss_w / iter_num, total_loss_a / iter_num, acc_w, acc_a

    @torch.no_grad()
    def eval_search_net(self, net, index_sampled, feats, labels_cuda, start, end,
                        device, loss_fcn):
        r"""Validation loss of one candidate meta-path set (used by Eq. 6)."""
        preds = []
        for batch in torch.split(torch.arange(start, end), self.args.batch_size):
            batch_feats = {k: x[batch].to(device) for k, x in feats.items()}
            preds.append(net(index_sampled, batch_feats).cpu())
        return loss_fcn(torch.cat(preds, dim=0), labels_cuda[start:end].cpu()).item()

    # ------------------------------------------------------------------ #
    # stage 2: train the target-net (paper Eq. 7)
    # ------------------------------------------------------------------ #
    def train_target(self, feats, selected, labels, labels_cuda, train_node_nums,
                     valid_node_nums, test_node_nums, trainval_point,
                     valtest_point, total_num_nodes, num_nodes, n_classes,
                     device, evaluator, checkpt_file):
        args = self.args
        print('=' * 60)
        print('Stage 2/2: training the target-net on the searched meta-paths (paper Eq. 7)')
        print('=' * 60)

        # Eq. 7 concatenates over S_M only, so the concatenated channels are
        # exactly the searched meta-paths.  The target key is additionally kept
        # in the feature dict (it feeds the residual connection) but is *not* a
        # channel unless the search selected it.
        model_keys = sorted(selected)
        keep = set(model_keys) | {TGT_TYPE}
        # The discarded features are deleted from the dictionary the caller
        # passed in, not just left behind by a rebinding: rebinding a local
        # name would leave the caller's dictionary holding all K propagated
        # features for the whole training stage, which on ogbn-mag is tens of
        # GB of host memory that nothing reads any more.
        for k in [k for k in feats if k not in keep]:
            del feats[k]
        feats = {k: feats[k] for k in sorted(keep)}

        # the model is built through the registry so that LMSPS behaves like
        # any other model of the library
        args.data_size = {k: v.size(-1) for k, v in feats.items()}
        args.feat_keys = model_keys
        args.num_feats = len(model_keys)
        args.num_label_feats = 0
        args.nfeat = args.embed_size
        args.nclass = n_classes
        args.tgt_key = TGT_TYPE

        model = build_model(args.model).build_model_from_args(args).to(device)
        print(model)
        print("# Params:", self.get_n_params(model))

        loss_fcn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                     weight_decay=args.weight_decay)
        scalar = torch.cuda.amp.GradScaler() if args.amp else None

        train_loader = torch.utils.data.DataLoader(
            torch.arange(train_node_nums), batch_size=args.batch_size,
            shuffle=True, drop_last=False)
        # validation + test nodes, in the rearranged order
        eval_loader = torch.utils.data.DataLoader(
            torch.arange(trainval_point, num_nodes), batch_size=args.batch_size,
            shuffle=False, drop_last=False)

        best_epoch, best_val_acc, best_test_acc, count = 0, 0., 0., 0
        for epoch in range(args.stages[0]):
            gc.collect()
            torch.cuda.empty_cache()
            start = time.time()
            loss, acc = self.run(model, train_loader, loss_fcn, optimizer,
                                 evaluator, device, feats, labels_cuda, scalar)
            end = time.time()
            log = "Epoch {}, Time(s): {:.4f}, estimated train loss {:.4f}, acc {:.4f}\n".format(
                epoch, end - start, loss, acc * 100)

            if epoch % args.eval_every == 0:
                raw_preds = self.gen_output(model, feats, eval_loader, device)
                preds = raw_preds.argmax(dim=-1)
                val_acc = evaluator(preds[:valid_node_nums],
                                    labels[trainval_point:valtest_point])
                test_acc = evaluator(preds[valid_node_nums:valid_node_nums + test_node_nums],
                                     labels[valtest_point:total_num_nodes])
                loss_val = loss_fcn(raw_preds[:valid_node_nums],
                                    labels[trainval_point:valtest_point]).item()
                log += 'Val loss: {:.4f}, Val acc: {:.4f}, Test acc: {:.4f}\n'.format(
                    loss_val, val_acc * 100, test_acc * 100)

                if val_acc > best_val_acc:
                    best_epoch, best_val_acc, best_test_acc = epoch, val_acc, test_acc
                    torch.save(model.state_dict(), checkpt_file + '.pkl')
                    count = 0
                else:
                    count += args.eval_every
                    if count >= args.patience:
                        break
                log += "Best Epoch {}, Val {:.4f}, Test {:.4f}".format(
                    best_epoch, best_val_acc * 100, best_test_acc * 100)
            print(log, flush=True)

        model.load_state_dict(torch.load(checkpt_file + '.pkl'))
        all_loader = torch.utils.data.DataLoader(
            torch.arange(num_nodes), batch_size=args.batch_size,
            shuffle=False, drop_last=False)
        raw_preds = self.gen_output(model, feats, all_loader, device, verbose=True)
        torch.save(raw_preds, checkpt_file + '.pt')
        return best_epoch, best_val_acc, best_test_acc

    def run(self, model, train_loader, loss_fcn, optimizer, evaluator, device,
            feats, labels_cuda, scalar=None):
        """Train the target-net for one epoch."""
        model.train()
        total_loss, iter_num = 0, 0
        y_true, y_pred = [], []
        for batch in train_loader:
            batch_feats = {k: x[batch].to(device) for k, x in feats.items()}
            batch_y = labels_cuda[batch]

            optimizer.zero_grad()
            if scalar is not None:
                with torch.cuda.amp.autocast():
                    output = model({'0': batch_feats})
                    loss_train = loss_fcn(output, batch_y)
                scalar.scale(loss_train).backward()
                scalar.step(optimizer)
                scalar.update()
            else:
                output = model({'0': batch_feats})
                loss_train = loss_fcn(output, batch_y)
                loss_train.backward()
                optimizer.step()

            y_true.append(batch_y.cpu().to(torch.long))
            y_pred.append(output.argmax(dim=-1, keepdim=True).cpu())
            total_loss += loss_train.item()
            iter_num += 1
        loss = total_loss / iter_num
        acc = evaluator(torch.cat(y_true, dim=0), torch.cat(y_pred, dim=0))
        return loss, acc

    @torch.no_grad()
    def gen_output(self, model, feats, loader, device, verbose=False):
        r"""
        Raw predictions of the target-net over all nodes yielded by ``loader``.

        Parameters
        ----------
        feats : dict[str, torch.Tensor]
            The pre-propagated meta-path features; every batch is gathered from
            it, so the same tensors are shared by the training and the inference
            pass.
        loader : torch.utils.data.DataLoader
            Yields the indexes of the nodes to predict.
        """
        model.eval()
        preds = []
        for batch in tqdm(loader) if verbose else loader:
            batch_feats = {k: x[batch].to(device) for k, x in feats.items()}
            preds.append(model({'0': batch_feats}).cpu())
        return torch.cat(preds, dim=0)

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def propagate_features(self):
        r"""
        The neighbour aggregation block (paper Eq. 1, following SeHGNN): one
        pass of message passing pre-computes ``X_k = A_{c_0,c_1} ... A_{c_{l-1},c_l} X_{c_l}``
        for every meta-path, so that both stages operate on regular tensors.

        Returns
        -------
        (dict, list)
            The feature of every meta-path and the meta-path names in a fixed
            (sorted) order.  The order matters: index ``i`` of the architecture
            parameters ``alpha`` refers to ``feat_keys[i]``.
        """
        args = self.args
        g = self.flow.dataset.SeHGNN_g
        g = self.hg_propagate(g, TGT_TYPE, args.num_hops, args.num_hops + 1, echo=False)
        keys = list(g.nodes[TGT_TYPE].data.keys())
        feats = {k: g.nodes[TGT_TYPE].data.pop(k) for k in keys}
        self.flow.dataset.SeHGNN_g = self.clear_hg(g, echo=False)
        feat_keys = sorted(feats.keys())
        return {k: feats[k] for k in feat_keys}, feat_keys

    def hg_propagate(self, new_g, tgt_type, num_hops, max_hops, echo=False):
        """One-time message passing over every meta-path (same as SeHGNN_trainer)."""
        for hop in range(1, max_hops):
            for etype in new_g.etypes:
                stype, _, dtype = new_g.to_canonical_etype(etype)
                for k in list(new_g.nodes[stype].data.keys()):
                    if len(k) == hop:
                        current_dst_name = f'{dtype}{k}'
                        if (hop == num_hops and dtype != tgt_type) or (hop > num_hops):
                            continue
                        if echo:
                            print(k, etype, current_dst_name)
                        new_g[etype].update_all(fn.copy_u(k, 'm'),
                                                fn.mean('m', current_dst_name), etype=etype)
            # remove no-use items
            for ntype in new_g.ntypes:
                if ntype == tgt_type:
                    continue
                removes = [k for k in new_g.nodes[ntype].data.keys() if len(k) <= hop]
                for k in removes:
                    new_g.nodes[ntype].data.pop(k)
            gc.collect()
        return new_g

    def clear_hg(self, new_g, echo=False):
        if echo:
            print('Remove keys left after propagation')
        for ntype in new_g.ntypes:
            keys = list(new_g.nodes[ntype].data.keys())
            if len(keys):
                if echo:
                    print(ntype, keys)
                for k in keys:
                    new_g.nodes[ntype].data.pop(k)
        return new_g

    @staticmethod
    def batches(loader):
        """Yield batches of `loader` forever, reshuffling at every pass."""
        while True:
            for batch in loader:
                yield batch

    def set_random_seed(self, seed=0):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    def get_n_params(self, model):
        pp = 0
        for p in list(model.parameters()):
            nn_ = 1
            for s in list(p.size()):
                nn_ = nn_ * s
            pp += nn_
        return pp
