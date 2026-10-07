LMSPS
=====

LMSPS (Long-range Meta-path Search through Progressive Sampling) is a NeurIPS
2024 model for heterogeneous node classification.  It searches over *all*
meta-paths related to the target node type instead of a hand-picked set, and
shrinks the candidate set progressively so that long-range meta-paths stay
affordable.  The search and the final training are two separate stages: a
super-net is searched first, then a target-net is trained on the meta-paths the
search selected.

Paper and Code
--------------

* Paper: https://arxiv.org/abs/2409.12011
* Upstream code: https://github.com/JHL-HUST/LMSPS

How to Run
----------

Run LMSPS through the OpenHGNN node classification entry point:

.. code-block:: bash

   python main.py -m LMSPS -d ogbn-mag -t node_classification -g 0

Use CPU by setting ``-g -1``.

Supported dataset:

* ``ogbn-mag``

The pipeline propagates the features of all ``K = 226`` meta-paths of up to six
hops in a single pass, before the search starts, and keeps them in main memory,
which is what the reference implementation does as well.  That propagation is
the memory peak of the pipeline and it is paid in full before any meta-path has
been selected, so setting ``meta_paths`` in the ``[LMSPS]`` section of
``openhgnn/config.ini`` to a comma-separated meta-path list (for example the
meta-paths reported in Table 9 of the paper) skips the *search* and trains the
target-net on that list directly, but it does not lower the peak.  Lower
``num_hops`` to reduce it.

Reproduction Result
-------------------

Node classification on OGBN-MAG:

.. list-table::
   :header-rows: 1

   * - Method
     - Validation accuracy
     - Test accuracy
   * - Paper (Table 1)
     - --
     - 54.83 ± 0.20
   * - Reference implementation (authors' code, 10 seeds)
     - --
     - 53.28 ± 0.14
   * - OpenHGNN (search + train, seed 1)
     - 47.8784
     - 47.1590
   * - OpenHGNN (paper meta-paths, seed 1)
     - 54.1685
     - 53.4348

The reference implementation row is the mean and standard deviation over 10
seeds of the authors' code run with ``--arch ogbn_withoutLabel``, i.e. the
meta-paths of Table 9 of the paper; the authors' OGBN-MAG command does not
re-run the search, and mixed precision is off as in the reference runs.  The
first OpenHGNN row is a single run (``seed = 1``) of the default two-stage
pipeline (search, then train): the search is stochastic and a single search
can land on a weaker meta-path set.  The second OpenHGNN row fixes the
meta-paths to Table 9 via ``meta_paths`` and trains the target-net only; it
lands inside the reference distribution, which shows the training stage of
this port matches the reference implementation.

Environment:

* OpenHGNN v0.9.0, Python 3.9, PyTorch 2.5.1+cu121, DGL 2.2.1+cu121 (CUDA
  12.6 toolkit), a single NVIDIA RTX 3090.
* Log file: ``openhgnn/output/LMSPS/LMSPS-<TIME>.log``.

Implementation Notes
--------------------

The two stages of the paper map to the following two classes:

* ``LMSPS_SearchNet`` is the super-net of Section 5.1.  Neighbour aggregation
  (Eq. 1) is done once as a preprocessing step by the flow, so the super-net
  holds the per-meta-path feature projections, the progressive sampling search
  (Eq. 3), the sampling evaluation (Eq. 6) and the MLP head of Eq. 4.  One
  architecture parameter is learned per meta-path and its path strength is a
  Gumbel-softmax sample (Eq. 2).  The parameters of the network and the
  architecture parameters are updated alternately on the training and the
  validation split (first-order approximation of Eq. 5).
* ``LMSPS`` is the target-net of Eq. 7.  It concatenates the projections of the
  selected meta-paths and maps them with an MLP head.  Compared with
  ``SeHGNN`` it has neither neighbour attention nor semantic attention, i.e. it
  is a pure MLP architecture with fewer inductive biases, as Section 5.3 of the
  paper requires.  ``Conv1d1x1`` is imported from ``SeHGNN`` because with
  ``groups == num_feats`` each meta-path owns an independent kernel, which is
  exactly the per-meta-path projection of the paper.

Scope of this version:

* The ``ogbn-mag`` pipeline without label propagation, i.e. the ``54.83`` entry
  of Table 1 of the paper.
* Label propagation and multi-stage self-training (the ``+LP+MS`` entry of
  Table 1) are not included.
* The meta-paths are sorted before the search, so that index ``i`` of the
  architecture parameters refers to the same meta-path across runs.

Smoke Test
----------

The unit tests cover the search space, both samplers and both networks on
synthetic features, and run on CPU in a few seconds:

.. code-block:: bash

   pytest tests/test_lmsps.py -v

The end-to-end command downloads and preprocesses ``ogbn-mag``; see the notes
in ``openhgnn/output/LMSPS/Readme.md`` for its memory requirement and runtime.
