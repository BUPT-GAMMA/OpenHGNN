HGOT
====

HGOT (Self-supervised Heterogeneous Graph Neural Network with Optimal
Transport) is an ICML 2025 model for heterogeneous node representation
learning.

Paper and Code
--------------

* Paper: https://arxiv.org/abs/2506.02619
* Upstream code: https://github.com/yanbeiliu/HGOT-ICML2025

How to Run
----------

Run HGOT through the OpenHGNN node classification entry point:

.. code-block:: bash

   python main.py -m HGOT -t node_classification -d imdb4MAGNN -g 0

Use CPU by setting ``-g -1``.

Supported dataset:

* ``imdb4MAGNN``

Reproduction Result
-------------------

Node classification on IMDB:

.. list-table::
   :header-rows: 1

   * - Method
     - Macro-F1
     - Micro-F1
   * - MAGNN
     - 58.65
     - 59.20
   * - Paper
     - 60.75
     - 60.98
   * - OpenHGNN
     - 60.54
     - 60.70

The paper setting uses SVM classification, so the semi-supervised OpenHGNN
trainer may differ slightly from the reported evaluation protocol.

Implementation Notes
--------------------

HGOT first projects heterogeneous node features into a unified latent space,
then aggregates multiple metapath-based views into a central view. The model
uses DGL metapath reachable graphs and DGL GATConv components.

Key hyperparameters are configured in ``openhgnn/config.ini``:

.. code-block:: ini

   feats_drop_rate = 0.3
   attn_vec_dim = 64
   feats_opt = 110
   loss_lambda = 0.2
   src_node_type = 2
   dropout = 0.1
   num_heads = 8
   HIN = MAGNN

Smoke Test
----------

The release smoke should cover:

.. code-block:: bash

   python main.py -m HGOT -t node_classification -d imdb4MAGNN -g -1
