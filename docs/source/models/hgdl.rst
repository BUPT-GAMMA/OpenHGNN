HGDL
====

HGDL (Heterogeneous Graph Label Distribution Learning) is a NeurIPS 2024 model
for label distribution learning on heterogeneous graphs.

Paper and Code
--------------

* Paper: https://openreview.net/forum?id=OwguhIAh8R
* Upstream code: https://github.com/Listener-Watcher/HGDL

How to Run
----------

HGDL is driven by the OpenHGNN node classification task, but its labels are
real-valued distributions rather than integer class labels.

.. code-block:: bash

   python main.py -m HGDL -t node_classification -d dblp4HGDL -g 0 --use_best_config
   python main.py -m HGDL -t node_classification -d acm4HGDL -g 0 --use_best_config

Use CPU by setting ``-g -1``.

Supported datasets:

* ``dblp4HGDL``
* ``acm4HGDL``

Reproduction Result
-------------------

Label distribution learning metrics. Lower is better for KL, Canberra,
Chebyshev, and Clark. Higher is better for Cosine and Intersection.

.. list-table::
   :header-rows: 1

   * - Dataset
     - KL
     - Cosine
     - Canberra
     - Chebyshev
     - Clark
     - Intersection
   * - dblp4HGDL
     - 0.0704
     - 0.0198
     - 2.8117
     - 0.0637
     - 1.6377
     - 0.9356
   * - acm4HGDL
     - 0.7514
     - 0.1962
     - 13.1013
     - 0.3516
     - 3.4767
     - 0.6414

Implementation Notes
--------------------

HGDL uses a dedicated ``HGDL_trainer`` because the target labels are
distributions. The trainer uses KL divergence and six label-distribution
metrics instead of the default argmax/F1 evaluation used by ordinary node
classification.

The model combines:

* metapath-based DGL GraphConv over preprocessed HGDL views;
* transformer-style attention implemented with DGL message passing,
  ``apply_edges``, ``edge_softmax``, and ``update_all``;
* a final GraphConv and softmax head for per-node label distributions.

Key hyperparameters are configured in ``openhgnn/config.ini``:

.. code-block:: ini

   seed = 0
   lr = 0.001
   weight_decay = 0
   max_epoch = 2000
   patience = 50
   hidden_dim = 64
   dropout = 0.3
   attention_dim = 5
   num_heads = 2
   gamma = 0.0001
   mini_batch_flag = False

Smoke Test
----------

The release smoke should cover:

.. code-block:: bash

   python main.py -m HGDL -t node_classification -d dblp4HGDL -g -1
