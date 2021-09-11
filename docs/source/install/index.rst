Install and Setup
=================

System requirements
-------------------
OpenHGNN works with the following operating systems:

* Ubuntu 16.04
* macOS X
* Windows 10

OpenHGNN requires Python version 3.6, 3.7, 3.8 or 3.9.

Install
-------------------------

Python environment requirments

* `Python <https://www.python.org/>`_ >= 3.6
- `PyTorch <https://pytorch.org/get-started/locally/>`_  >= 1.7.1
- `DGL <https://github.com/dmlc/dgl>`_ >= 0.7.0


**1. Python environment (Optional):** We recommend using Conda package manager

.. code:: bash

    conda create -n openhgnn python=3.7
    source activate openhgnn


**2. Pytorch:** Install `PyTorch <https://pytorch.org/>`_. We have verified GraphGym under PyTorch 1.8.0. For example:

.. code:: bash

    # CUDA versions: cpu, cu92, cu101, cu102, cu101, cu111
    pip install torch==1.8.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html


**3. DGL:** Install `DGL <https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html>`_,
follow their instructions. For example:

.. code:: bash

    # CUDA versions: cpu, cu101, cu102, cu110, cu111
    pip install --pre dgl-cu101 -f https://data.dgl.ai/wheels-test/repo.html

**4. OpenHGNN and other dependencies:**

.. code:: bash

    git clone https://github.com/BUPT-GAMMA/OpenHGNN
    cd OpenHGNN
    pip install -r requirements.txt
