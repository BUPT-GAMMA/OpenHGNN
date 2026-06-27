HGOT
====

* Paper: Self-supervised Heterogeneous Graph Neural Network with Optimal
  Transport, ICML 2025
* Registered model name: ``HGOT``
* Task: ``node_classification``
* Upstream code: https://github.com/yanbeiliu/HGOT-ICML2025

Reproduction
------------

.. list-table::
   :header-rows: 1

   * - Field
     - Value
   * - Data source
     - ``imdb4MAGNN`` from the OpenHGNN dataset mirror.
   * - Preprocessing
     - Builds metapath reachable graphs for IMDB and projects heterogeneous
       node features into a shared latent space before optimal transport
       alignment.
   * - Command
     - ``python main.py -m HGOT -t node_classification -d imdb4MAGNN -g 0``
   * - Metric
     - Macro-F1 and Micro-F1 on IMDB node classification.
   * - Expected result
     - OpenHGNN: Macro-F1 60.54, Micro-F1 60.70. Paper reference:
       Macro-F1 60.75, Micro-F1 60.98. MAGNN baseline: Macro-F1 58.65,
       Micro-F1 59.20.
   * - Hardware/runtime
     - GPU is recommended. CPU smoke uses ``-g -1``; full runtime was not
       recorded in merged materials.
   * - Seed
     - ``seed = 0`` in ``config.ini``.

Implementation Notes
--------------------

HGOT exposes optimal transport hyperparameters in ``config.ini`` including
``feats_drop_rate``, ``attn_vec_dim``, ``feats_opt``, ``loss_lambda``,
``src_node_type``, ``dropout``, ``num_heads``, and ``HIN``. The paper setting
uses SVM classification, so the semi-supervised OpenHGNN trainer may differ
slightly from the reported evaluation protocol.
