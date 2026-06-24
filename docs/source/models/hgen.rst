HGEN
====

HGEN (Heterogeneous Graph Ensemble Networks) is an IJCAI 2025 model for
heterogeneous node classification.

Paper and Code
--------------

* Paper: Heterogeneous Graph Ensemble Networks, IJCAI 2025
* Upstream code: https://github.com/Chrisshen12/HGEN

How to Run
----------

Run HGEN through the OpenHGNN node classification entry point:

.. code-block:: bash

   python main.py -m HGEN -t node_classification -d dblp4GTN -g 0

Use CPU by setting ``-g -1``.

Supported datasets:

* ``dblp4GTN``
* ``acm4GTN``
* ``imdb4GTN``

Reproduction Result
-------------------

Default-parameter node classification results:

.. list-table::
   :header-rows: 1

   * - Dataset
     - Macro-F1
     - Micro-F1
   * - DBLP
     - 88.15
     - 89.09
   * - IMDB
     - 56.31
     - 57.51
   * - ACM
     - 89.62
     - 89.46

The paper reports DBLP Macro-F1 94.18 and IMDB Micro-F1 61.58 with tuned
hyperparameters. The default OpenHGNN reproduction follows the released code
setting.

Implementation Notes
--------------------

HGEN uses DGL metapath reachable graphs and DGL GraphConv modules. For each
meta-path, the model extracts a homogeneous view with
``dgl.metapath_reachable_graph``, runs independent GCN encoders, fuses
meta-path logits with attention, and adds a path-correlation regularization
term in ``HGEN_trainer``.

Key hyperparameters are configured in ``openhgnn/config.ini``:

.. code-block:: ini

   hidden_dim = 64
   out_dim = 16
   num_gcn = 3
   num_layers = 2
   attention_dim = 8
   dropout = 0.1
   lambda_cov = 0
   max_epoch = 500
   patience = 50

Smoke Test
----------

The release smoke should cover:

.. code-block:: bash

   python main.py -m HGEN -t node_classification -d dblp4GTN -g -1
