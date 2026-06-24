HERO Homo
=========

``HERO_homo`` is the homogeneous-graph entry point for HERO.

Paper and Code
--------------

* Paper: https://openreview.net/forum?id=3FJOKjooIj
* Upstream code: https://github.com/YujieMo/HERO

How to Run
----------

.. code-block:: bash

   python main.py -m HERO_homo -d photo4HERO -t node_classification -g 0
   python main.py -m HERO_homo -d computers4HERO -t node_classification -g 0
   python main.py -m HERO_homo -d cs4HERO -t node_classification -g 0
   python main.py -m HERO_homo -d physics4HERO -t node_classification -g 0

Supported datasets:

* ``photo4HERO``
* ``computers4HERO``
* ``cs4HERO``
* ``physics4HERO``
* ``cora4HERO``
* ``citeseer4HERO``
* ``pubmed4HERO``
* ``corafull4HERO``
* ``wikics4HERO``
* ``ogbn-arxiv4HERO``

Reproduction Result
-------------------

.. list-table::
   :header-rows: 1

   * - Dataset
     - Target node
     - F1-micro
     - F1-macro
   * - Photo
     - photo
     - 0.93343
     - 0.92021
   * - Computers
     - computer
     - 0.88533
     - 0.86264
   * - CS
     - author
     - 0.92754
     - 0.87665
   * - Physics
     - author
     - 0.95925
     - 0.94587

Smoke Test
----------

The release smoke should cover:

.. code-block:: bash

   python main.py -m HERO_homo -d photo4HERO -t node_classification -g -1

If CPU execution is not supported, the PR should document the reason and the
CI smoke should use an explicit GPU-enabled environment.
