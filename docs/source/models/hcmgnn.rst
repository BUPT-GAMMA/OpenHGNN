HCMGNN
======

HCMGNN (Heterogeneous Causal Metapath Graph Neural Network) is an IJCAI 2024
model for gene-microbe-disease association prediction.

Paper and Code
--------------

* Paper: https://arxiv.org/pdf/2406.19156
* Upstream code: https://github.com/zkxinxin/HCMGNN

How to Run
----------

Run HCMGNN through the OpenHGNN recommendation entry point:

.. code-block:: bash

   python main.py -m HCMGNN -d GMD4HCMGNN -t recommendation -g 0 --use_best_config

Use CPU by setting ``-g -1``.

Supported dataset:

* ``GMD4HCMGNN``

Reproduction Result
-------------------

Recommendation metrics on the processed GMD dataset:

.. list-table::
   :header-rows: 1

   * - Setting
     - Hits@5
     - Hits@3
     - Hits@1
     - NDCG@5
     - NDCG@3
     - NDCG@1
     - MRR
   * - HCMGNN-CV
     - 0.9647
     - 0.9381
     - 0.7791
     - 0.8861
     - 0.8752
     - 0.7791
     - 0.8629
   * - HCMGNN-Indep
     - 0.9650
     - 0.9534
     - 0.8222
     - 0.9064
     - 0.9015
     - 0.8222
     - 0.8899

Dataset
-------

The dataset is processed from the GMD data released by the upstream HCMGNN
project and is expected to be available as ``GMD4HCMGNN`` through OpenHGNN's
dataset path.

Implementation Notes
--------------------

HCMGNN combines causal metapaths with heterogeneous graph neural networks for
gene-microbe-disease association prediction. The implementation runs both
5-fold cross-validation and independent testing.

Release Follow-up
-----------------

The model should eventually expose the standard OpenHGNN
``build_model_from_args(cls, args, hg)`` signature. The current implementation
uses a model-specific constructor call from ``HCMGNN_trainer``.

Smoke Test
----------

The release smoke should cover:

.. code-block:: bash

   python main.py -m HCMGNN -d GMD4HCMGNN -t recommendation -g -1 --use_best_config
