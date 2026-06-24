HTGformer
=========

HTGformer (Heterogeneous Temporal Graph Transformer) is a SIGIR 2025 short
paper for heterogeneous temporal graph learning.

Paper and Code
--------------

* Paper: https://doi.org/10.1145/3726302.3730209
* Upstream code: not publicly available

How to Run
----------

.. code-block:: bash

   python main.py -m HTGformer -d ogbn_mag4HGformer -t link_prediction -g 0
   python main.py -m HTGformer -d aminer4HGformer -t link_prediction -g 0
   python main.py -m HTGformer -d yelp4HGformer -t node_classification -g 0
   python main.py -m HTGformer -d covid4HGformer -t node_regression -g 0

The trainer also accepts the short aliases ``ogbn_mag``, ``aminer``, ``yelp``,
and ``covid``.

Datasets
--------

.. list-table::
   :header-rows: 1

   * - Dataset
     - Task
     - Files
   * - ``ogbn_mag4HGformer``
     - link prediction
     - ``ogbn4HGformer.bin``, ``mp2vec.zip``
   * - ``aminer4HGformer``
     - link prediction
     - ``aminer4HGformer.pt``
   * - ``yelp4HGformer``
     - node classification
     - ``yelp4HGformer.pt``
   * - ``covid4HGformer``
     - node regression
     - ``covid4HGformer.bin``

Reproduction Result
-------------------

The OpenHGNN reproduction runs in ``w/o_LLM`` mode with learnable embeddings
instead of LLaMA3 type encodings.

.. list-table::
   :header-rows: 1

   * - Dataset
     - Task
     - Metric
     - Paper
     - OpenHGNN
   * - OGBN-MAG
     - link prediction
     - AUC
     - 92.56
     - 94.61
   * - OGBN-MAG
     - link prediction
     - AP
     - 91.64
     - 93.98
   * - Aminer
     - link prediction
     - AUC
     - 89.78
     - 88.41
   * - YELP
     - node classification
     - Macro-F1
     - 43.24
     - 35.91
   * - COVID-19
     - node regression
     - MAE
     - 532
     - 511.59

Release Follow-up
-----------------

The node regression command requires OpenHGNN's formal ``node_regression`` task
to be present in the main task registry. Until then, the release smoke should
cover link prediction first.

Smoke Test
----------

The initial release smoke should cover:

.. code-block:: bash

   python main.py -m HTGformer -d ogbn_mag4HGformer -t link_prediction -g -1
