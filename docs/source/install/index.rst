Install and Setup
=================

System requirements
-------------------
OpenHGNN works with the following operating systems:

* Ubuntu 16.04
* macOS X
* Windows 10

OpenHGNN requires Python version 3.6, 3.7, 3.8, or 3.9.

Install
-------------------------

Python environment requirments

- `Python <https://www.python.org/>`_ >= 3.6
- `PyTorch <https://pytorch.org/>`_  >= 1.9.0
- `DGL <https://github.com/dmlc/dgl>`_ >= 0.8.0


**1. Python environment (Optional):** We recommend using Conda package manager

.. code:: bash

    conda create -n openhgnn python=3.7
    source activate openhgnn


**2. Pytorch:** Install `PyTorch <https://pytorch.org/>`_. For example:

.. code:: bash


    pip install torch==1.8.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html


**3. DGL:** Install `DGL <https://github.com/dmlc/dgl>`_,
follow their instructions. For example:

.. code:: bash


    pip install dgl dglgo -f https://data.dgl.ai/wheels/repo.html

**4. OpenHGNN and other dependencies:**

.. code:: bash

    git clone https://github.com/BUPT-GAMMA/OpenHGNN
    cd OpenHGNN
    pip install -r requirements.txt
