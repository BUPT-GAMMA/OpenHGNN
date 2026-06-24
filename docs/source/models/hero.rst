HERO
====

HERO is an ICLR 2024 self-supervised heterogeneous graph learning model with a
homophily and heterogeneity view.

Paper and Code
--------------

* Paper: https://openreview.net/forum?id=3FJOKjooIj
* Upstream code: https://github.com/YujieMo/HERO

How to Run
----------

.. code-block:: bash

   python main.py -m HERO -d Yelp4HERO -t node_classification -g 0
   python main.py -m HERO -d ACM4HERO -t node_classification -g 0
   python main.py -m HERO -d DBLP4HERO -t node_classification -g 0
   python main.py -m HERO -d Aminer4HERO -t node_classification -g 0

Supported datasets:

* ``ACM4HERO``
* ``Aminer4HERO``
* ``DBLP4HERO``
* ``Yelp4HERO``

Reproduction Result
-------------------

.. list-table::
   :header-rows: 1

   * - Dataset
     - Target node
     - F1-micro
     - F1-macro
   * - ACM
     - paper
     - 0.91501
     - 0.91595
   * - Aminer
     - paper
     - 0.72020
     - 0.66705
   * - DBLP
     - author
     - 0.93588
     - 0.92678
   * - Yelp
     - business
     - 0.92354
     - 0.92677

Dataset Download
----------------

The PR downloads datasets from the OpenHGNN HERO dataset mirror and processes
them into DGL heterographs.

Smoke Test
----------

The release smoke should cover:

.. code-block:: bash

   python main.py -m HERO -d Yelp4HERO -t node_classification -g -1

If CPU execution is not supported, the PR should document the reason and the
CI smoke should use an explicit GPU-enabled environment.
