Install
=================

OpenHGNN works with the following operating systems:

* Ubuntu 16.04
* macOS X
* Windows 10

Python environment requirments

- `Python <https://www.python.org/>`_ >= 3.6
- `PyTorch <https://pytorch.org/>`_  <= 2.3.0
- `DGL <https://github.com/dmlc/dgl>`_ <= 2.2.1


**1. Python environment (Optional):** We recommend using Conda package manager

.. code:: bash

    conda create -n openhgnn python=3.6
    source activate openhgnn

**2. Pytorch:** Follow their `tutorial <https://pytorch.org/get-started/>`_ to run the proper command according to
your OS and CUDA version. For example:

.. code:: bash

    pip install torch torchvision torchaudio

**3. DGL:** Follow their `tutorial <https://www.dgl.ai/pages/start.html>`_ to run the proper command according to
your OS and CUDA version. For example:

.. code:: bash

    pip install dgl -f https://data.dgl.ai/wheels/repo.html

**4. Install openhgnn:**

* install from pypi

.. code:: bash

    pip install openhgnn

* install from source

.. code:: bash

    git clone https://github.com/BUPT-GAMMA/OpenHGNN
    # If you encounter a network error, try git clone from openi as following.
    # git clone https://git.openi.org.cn/GAMMALab/OpenHGNN.git
    cd OpenHGNN
    pip install .

