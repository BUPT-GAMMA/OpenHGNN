"""
LMSPS: Long-range Meta-path Search through Progressive Sampling (NeurIPS 2024).

Paper: `Long-range Meta-path Search through Progressive Sampling
       <https://arxiv.org/abs/2409.12011>`__
Code:  https://github.com/JHL-HUST/LMSPS

LMSPS searches over *all* meta-paths related to the target node type and shrinks
the candidate set progressively, which makes long-range meta-paths affordable
instead of exponentially costly.  Two networks are involved:

* :class:`LMSPS_SearchNet` -- the super-net used in the search stage
  (paper Section 5.1, Eq. 2-4).  Neighbour aggregation is performed once as a
  preprocessing step (the flow calls ``hg_propagate``), so the super-net only
  contains the feature projection block (one MLP per meta-path, Eq. 1), the
  progressive sampling search (Eq. 3), the sampling evaluation (Eq. 6) and an
  MLP head (Eq. 4).

* :class:`LMSPS` -- the target-net trained after the search (paper Eq. 7).  It
  differs from :class:`~openhgnn.models.SeHGNN` only in that the transformer
  based semantic fusion is replaced by the concatenation of the per-meta-path
  projections, i.e. a pure MLP architecture with fewer inductive biases
  (paper Section 5.3).  ``Conv1d1x1`` is therefore imported from ``SeHGNN``:
  with ``groups == num_feats`` each meta-path owns an independent kernel, which
  is exactly the per-meta-path ``MLP_k`` of the paper.
"""

import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import BaseModel, register_model
from .SeHGNN import Conv1d1x1


class LMSPS_SearchNet(nn.Module):
    r"""
    Search-stage super-net of LMSPS (paper Section 5.1).

    The search space is ``P = {P_1, ..., P_K}``, the set of all meta-paths that
    start from the target node type.  One architecture parameter ``alpha_k`` is
    assigned to every meta-path; its path strength is a Gumbel-softmax sample
    (Eq. 2)::

        q_k = exp((alpha_k + u_k) / tau) / sum_j exp((alpha_j + u_j) / tau)

    with ``u_k = -log(-log(U))``, ``U ~ Uniform(0, 1)``.  During the search a
    dynamic search space ``S_C`` (Eq. 3) keeps the ``C = ceil(lam * (K - V)) + V``
    strongest meta-paths, ``M`` of which are activated per iteration (Eq. 4)::

        Z = MLP( sum_{k in S} q'_k * MLP_k(X_k) ),  S = UniformSample(S_C, M)

    ``q'`` is renormalized over the activated set ``S`` only.  After the search,
    ``M`` meta-paths are drawn from the compact space ``S_V`` with probability
    proportional to the path strength (sampling evaluation, Eq. 6).

    Parameters
    ----------
    feat_keys : list[str]
        Names of all ``K`` meta-path features, in a fixed order.  Index ``i`` of
        ``alpha`` refers to ``feat_keys[i]``, so this order must be stable
        between the search and the training stage.
    data_size : dict[str, int]
        Raw feature dimension of every meta-path feature.  Dimensions may differ
        across meta-paths (they are decided by the source node type of the path),
        hence the per-path projections are kept in ``ParameterDict`` instead of
        a single grouped convolution.
    hidden : int
        Hidden dimension.
    nclass : int
        Number of classes of the target node type.
    tgt_key : str
        Name of the meta-path feature holding the target nodes themselves, used
        by the residual connection.
    num_final : int
        Size ``V`` of the compact search space kept at the end of the search.
    num_sampled : int
        Number ``M`` of meta-paths activated in each iteration and finally
        selected.
    """

    def __init__(self, feat_keys, data_size, hidden, nclass, tgt_key,
                 dropout=0.5, input_drop=0.1, num_final=60, num_sampled=30,
                 residual=False):
        super(LMSPS_SearchNet, self).__init__()
        self.feat_keys = list(feat_keys)
        self.num_paths = len(self.feat_keys)
        self.num_final = num_final
        self.num_res = self.num_paths - self.num_final
        self.num_sampled = num_sampled
        self.tgt_key = tgt_key
        self.residual = residual
        # Gumbel temperature tau of Eq. 2, annealed by the flow through set_tau()
        self.tau = 1.0

        # ============ feature projection block (Eq. 1) ============
        # one two-layer MLP per meta-path, kept as ParameterDict because the
        # input dimension differs across meta-paths
        self.feat_proj_l1 = nn.ParameterDict()
        self.feat_proj_l2 = nn.ParameterDict()
        self.proj_keys = list(self.feat_keys)
        if self.residual and self.tgt_key not in self.proj_keys:
            self.proj_keys.append(self.tgt_key)
        for k in self.proj_keys:
            self.feat_proj_l1[k] = nn.Parameter(torch.empty(data_size[k], hidden))
            self.feat_proj_l2[k] = nn.Parameter(torch.empty(hidden, hidden))

        # ============ MLP head (outer MLP of Eq. 4) ============
        self.prelu = nn.PReLU()
        self.prelu_proj = nn.PReLU()
        self.dropout = nn.Dropout(dropout)
        self.input_drop = nn.Dropout(input_drop)
        self.lr_output = nn.Sequential(
            nn.Linear(hidden, nclass, bias=False),
            nn.BatchNorm1d(nclass),
        )

        # ============ architecture parameters (Eq. 2) ============
        self.alpha = nn.Parameter(torch.ones(self.num_paths))

        if self.residual:
            self.res_fc = nn.Linear(hidden, hidden)

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        for k in self.proj_keys:
            nn.init.xavier_uniform_(self.feat_proj_l1[k], gain=gain)
            nn.init.xavier_uniform_(self.feat_proj_l2[k], gain=gain)
        for layer in self.lr_output:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=gain)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        if self.residual:
            nn.init.xavier_uniform_(self.res_fc.weight, gain=gain)

    # ------------------------------------------------------------------ #
    # progressive sampling (paper Section 5.1, Eq. 2 - Eq. 4)
    # ------------------------------------------------------------------ #
    def _get_gumbel_prob(self, xins):
        r"""
        Gumbel-softmax over architecture parameters (paper Eq. 2).

        ``softmax((log_softmax(alpha) + gumbel) / tau)`` equals
        ``softmax((alpha + gumbel) / tau)`` because the additive constant of
        ``log_softmax`` cancels out, so this is Eq. 2 verbatim.  Samples
        containing non-finite values are redrawn, as in the reference
        implementation.
        """
        while True:
            gumbels = -torch.empty_like(xins).exponential_().log()
            logits = (xins.log_softmax(dim=-1) + gumbels) / self.tau
            probs = F.softmax(logits, dim=-1)
            if not (torch.isinf(gumbels).any()
                    or torch.isinf(probs).any()
                    or torch.isnan(probs).any()):
                return probs

    def dynamic_search_space(self, lam):
        r"""
        Dynamic search space ``S_C`` of Eq. 3.

        ``C = ceil(lam * (K - V)) + V``, and ``S_C`` retains every meta-path
        whose path strength ``q_k`` is no smaller than the ``C``-th largest
        strength ``e_q_C``.  ``lam`` decays from 1 to 0 during the search, so
        ``C`` shrinks from ``K`` to ``V``.

        Returns
        -------
        (torch.Tensor, int)
            Indexes of the retained meta-paths and the value of ``C``.
        """
        c = int(math.ceil(lam * self.num_res)) + self.num_final
        c = min(max(c, self.num_final), self.num_paths)
        q = self._get_gumbel_prob(self.alpha.detach())
        e_q_c = torch.topk(q, c).values.min()
        return torch.nonzero(q >= e_q_c).flatten(), c

    def compact_search_space(self):
        """``S_V``: the ``V`` strongest meta-paths, i.e. Eq. 3 with ``lam -> 0``."""
        v = min(self.num_final, self.num_paths)
        return torch.sort(torch.topk(self.alpha.detach(), v).indices).values

    def epoch_sample(self, lam):
        r"""
        Meta-paths activated in one search iteration (Eq. 4).

        ``S = UniformSample(S_C, M)``: ``M`` meta-paths are drawn uniformly
        *without* replacement from the dynamic search space.  Only the
        parameters on the activated meta-paths receive a gradient, so the search
        cost depends on ``M`` instead of ``K``.
        """
        space, _ = self.dynamic_search_space(lam)
        m = min(self.num_sampled, space.numel())
        return sorted(random.sample(space.tolist(), m))

    def sample_eval(self):
        r"""
        Sampling evaluation (paper Eq. 6): ``S_bar = DiscreteSample(S_V, M, Q_bar)``.

        ``M`` meta-paths are drawn from the compact space ``S_V`` *without*
        replacement with probability proportional to the path strength ``Q_bar``,
        which is Eq. 2 evaluated on the ``V`` retained meta-paths.  Unlike
        :meth:`epoch_sample`, the sampling is weighted, because after the search
        the relative strength of the candidates carries the information the
        search has accumulated.  The flow evaluates each draw on the validation
        set and keeps the set with the lowest loss.
        """
        space = self.compact_search_space()
        q_bar = self._get_gumbel_prob(self.alpha.detach()[space])
        m = min(self.num_sampled, space.numel())
        picked = torch.multinomial(q_bar, m, replacement=False)
        return sorted(space[picked].tolist())

    def set_tau(self, tau):
        """Set the Gumbel temperature (annealed from ``tau_max`` to ``tau_min``)."""
        self.tau = tau

    # ------------------------------------------------------------------ #
    # forward propagation of the super-net (paper Eq. 4)
    # ------------------------------------------------------------------ #
    def _project(self, k, v):
        """``MLP_k(X_k)``: the two-layer per-meta-path feature projection."""
        h = self.input_drop(v) @ self.feat_proj_l1[k]
        h = self.prelu_proj(h)
        return self.input_drop(h) @ self.feat_proj_l2[k]

    def forward(self, epoch_sampled, feats_dict, **kwargs):
        r"""
        Forward propagation of the super-net (paper Eq. 4)::

            Z = MLP( sum_{k in S} q'_k * MLP_k(X_k) )

        Parameters
        ----------
        epoch_sampled : list[int]
            Indexes of the activated meta-paths ``S``, produced by
            :meth:`epoch_sample` or :meth:`sample_eval`.
        feats_dict : dict[str, torch.Tensor]
            Pre-propagated meta-path features, keyed by meta-path name.

        Notes
        -----
        ``q'`` is normalized over the activated set only, which is what Eq. 4
        specifies; this is why the set is passed on to
        :meth:`_get_gumbel_prob` instead of being read from ``self.alpha``.
        Extra keyword arguments are accepted (and ignored) so that the signature
        matches the target-net, whose forward takes label features as well.
        """
        x = torch.stack([self._project(k, feats_dict[k])
                         for k in [self.feat_keys[i] for i in epoch_sampled]], dim=1)
        q_prime = self._get_gumbel_prob(self.alpha[epoch_sampled])
        x = torch.einsum('bcd,c->bd', x, q_prime)

        if self.residual:
            x = x + self.res_fc(self._project(self.tgt_key, feats_dict[self.tgt_key]))

        x = self.dropout(self.prelu(x))
        return self.lr_output(x)


@register_model('LMSPS')
class LMSPS(BaseModel):
    r"""
    Target-net of LMSPS (paper Eq. 7)::

        Z_hat = MLP( ||_{k in S_M} MLP_k(X_k) )

    ``S_M`` is the meta-path set selected by the sampling evaluation of the
    search stage; the projections of the selected meta-paths are concatenated
    and mapped by an MLP head.  There is neither neighbour attention nor
    semantic attention -- the parametric modules are pure MLPs (paper
    Section 5.2).

    Parameters
    ----------
    args : dict
        The model arguments, providing ``data_size``, ``nfeat``, ``hidden``,
        ``nclass``, ``feat_keys``, ``tgt_key``, ``dropout``, ``input_drop``,
        ``residual``, ``bns`` and ``n_layers_2``.
    """

    @classmethod
    def build_model_from_args(cls, args):
        return cls(args)

    def __init__(self, args):
        super(LMSPS, self).__init__()
        self.feat_keys = list(args.feat_keys)
        self.data_size = args.data_size
        self.nfeat = args.nfeat
        self.hidden = args.hidden
        self.nclass = args.nclass
        self.num_feats = len(self.feat_keys)
        self.tgt_key = args.tgt_key
        self.dropout = args.dropout
        self.input_drop = args.input_drop
        self.residual = args.residual
        self.bns = args.bns

        # Meta-path features coming from non-target node types may have a raw
        # dimension different from nfeat (paper Table 7: OGBN-MAG uses 128-d
        # paper features and 256-d random embeddings for author/institution/
        # field-of-study).  They are lifted to nfeat once, then share the
        # per-meta-path projection below.
        if any(v != self.nfeat for v in self.data_size.values()):
            self.embedings = nn.ParameterDict({})
            for k, v in self.data_size.items():
                if v != self.nfeat:
                    self.embedings[k] = nn.Parameter(
                        torch.Tensor(v, self.nfeat).uniform_(-0.5, 0.5))
        else:
            self.embedings = None

        # ============ feature projection block (Eq. 7) ============
        # one independent kernel per meta-path (groups == num_feats)
        self.feat_project_layers = nn.Sequential(
            Conv1d1x1(self.nfeat, self.hidden, self.num_feats, bias=True, cformat='channel-first'),
            nn.LayerNorm([self.num_feats, self.hidden]),
            nn.PReLU(),
            nn.Dropout(self.dropout),
            Conv1d1x1(self.hidden, self.hidden, self.num_feats, bias=True, cformat='channel-first'),
            nn.LayerNorm([self.num_feats, self.hidden]),
            nn.PReLU(),
            nn.Dropout(self.dropout),
        )
        # concatenation of the projections, followed by the outer MLP
        self.concat_project_layer = nn.Linear(self.num_feats * self.hidden, self.hidden)

        if self.residual:
            self.res_fc = nn.Linear(self.nfeat, self.hidden, bias=False)

        def add_nonlinear_layers(dropout, bns=False):
            if bns:
                return [nn.BatchNorm1d(self.hidden), nn.PReLU(), nn.Dropout(dropout)]
            else:
                return [nn.PReLU(), nn.Dropout(dropout)]

        lr_output_layers = [
            [nn.Linear(self.hidden, self.hidden, bias=not self.bns)]
            + add_nonlinear_layers(self.dropout, self.bns)
            for _ in range(args.n_layers_2 - 1)]
        self.lr_output = nn.Sequential(*(
            [ele for li in lr_output_layers for ele in li] + [
                nn.Linear(self.hidden, self.nclass, bias=False),
                nn.BatchNorm1d(self.nclass)]))

        self.prelu = nn.PReLU()
        self.dropout = nn.Dropout(self.dropout)
        self.input_drop = nn.Dropout(self.input_drop)

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        for layer in self.feat_project_layers:
            if isinstance(layer, Conv1d1x1):
                layer.reset_parameters()
        nn.init.xavier_uniform_(self.concat_project_layer.weight, gain=gain)
        nn.init.zeros_(self.concat_project_layer.bias)
        if self.residual:
            nn.init.xavier_uniform_(self.res_fc.weight, gain=gain)
        for layer in self.lr_output:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=gain)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, fk):
        r"""
        Parameters
        ----------
        fk : dict
            ``fk['0']`` holds the pre-propagated meta-path features as a dict
            keyed by meta-path name; the remaining entries are accepted for
            interface compatibility with the SeHGNN family of flows.

        Notes
        -----
        ``fk['0']`` must contain the target key even when the target meta-path
        was not selected, because it feeds the residual connection; it is then
        excluded from the concatenation, matching Eq. 7 where only the selected
        meta-paths ``S_M`` contribute a term.
        """
        feats_dict = fk['0']
        if self.embedings is not None:
            feats_dict = {k: (v @ self.embedings[k]) if k in self.embedings else v
                          for k, v in feats_dict.items()}

        tgt_feat = self.input_drop(feats_dict[self.tgt_key])
        x = self.input_drop(torch.stack([feats_dict[k] for k in self.feat_keys], dim=1))
        x = self.feat_project_layers(x)
        x = self.concat_project_layer(x.reshape(x.size(0), -1))

        if self.residual:
            x = x + self.res_fc(tgt_feat)

        x = self.dropout(self.prelu(x))
        return self.lr_output(x)
