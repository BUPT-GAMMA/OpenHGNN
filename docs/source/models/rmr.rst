RMR
===

RMR (Reserving-Masking-Reconstruction) is a KDD 2024 self-supervised
heterogeneous graph representation model. It learns representations with
information reserving, relation-subgraph masking, and reverse-relation
reconstruction.

Paper and Code
--------------

* Paper: https://dl.acm.org/doi/10.1145/3637528.3671719
* Upstream code: https://github.com/DuanhaoranCC/RMR

How to Run
----------

Run RMR through the OpenHGNN node classification entry point:

.. code-block:: bash

   python main.py -m RMR -t node_classification -d acm4RMR -g 0

Use CPU by setting ``-g -1``.

Supported datasets:

* ``acm4RMR``
* ``imdb4RMR``
* ``aminer4RMR``

Reproduction Result
-------------------

RMR reports node classification metrics over multiple label ratios. The
OpenHGNN reproduction covers ACM, IMDB, and Aminer.

.. list-table::
   :header-rows: 1

   * - Dataset
     - Main metric
     - Notes
   * - ``acm4RMR``
     - Macro-F1 / Micro-F1 / AUC
     - Close to the paper under 5/10/20 label ratios
   * - ``imdb4RMR``
     - Macro-F1 / Micro-F1 / AUC
     - Some low-ratio runs differ from the paper
   * - ``aminer4RMR``
     - Macro-F1 / Micro-F1
     - Uses the reproduced OpenHGNN split

Implementation Notes
--------------------

RMR uses a dedicated ``rmr_trainer``. Device placement must follow OpenHGNN's
``args.gpu``/``self.device`` contract; the trainer should not bind to
``cuda:0`` at import time.

Dataset Download
----------------

Datasets are expected to be available through the OpenHGNN dataset download
path:

.. code-block:: bash

   wget https://s3.cn-north-1.amazonaws.com.cn/dgl-data/dataset/openhgnn/acm4RMR.zip
   wget https://s3.cn-north-1.amazonaws.com.cn/dgl-data/dataset/openhgnn/imdb4RMR.zip
   wget https://s3.cn-north-1.amazonaws.com.cn/dgl-data/dataset/openhgnn/aminer4RMR.zip

Smoke Test
----------

The release smoke should cover:

.. code-block:: bash

   python main.py -m RMR -t node_classification -d acm4RMR -g -1
