Install
=================

OpenHGNN v0.9 targets reproducible installation first. The recommended
runtime is Linux with CPU or NVIDIA GPU support. macOS and Windows can be
used for development, but GPU and DGL wheel availability should be checked
against the upstream PyTorch and DGL installation pages.

After installation, start with:

.. toctree::
   :maxdepth: 1

   quick_start
   model_overview
   task_overview
   reproduce_model
   ai_assistant_skills

Supported environment matrix
----------------------------

.. list-table::
   :header-rows: 1

   * - Profile
     - Python
     - PyTorch
     - DGL
     - CUDA
   * - Primary
     - 3.11
     - 2.4.0
     - 2.4.0+cu121
     - 12.1
   * - Compatibility
     - 3.10
     - 2.3.1
     - 2.2.1
     - 12.1

The repository keeps the primary profile in ``environment.yml``. Use it when
you need a reproducible local or CI environment.

**1. Python environment (Optional):** We recommend using Conda package manager

.. code:: bash

    conda create -n openhgnn python=3.11
    conda activate openhgnn

**2. PyTorch:** Follow the `PyTorch installation guide <https://pytorch.org/get-started/>`_ to run the proper command according to
your OS and CUDA version. For example:

.. code:: bash

    pip install torch==2.4.0 torchvision torchaudio

**3. DGL:** Follow the `DGL installation guide <https://www.dgl.ai/pages/start.html>`_ to run the proper command according to
your OS and CUDA version. For example:

.. code:: bash

    pip install dgl==2.4.0+cu121 -f https://data.dgl.ai/wheels/torch-2.4/cu121/repo.html

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
    # To create the full pinned source environment directly, you can also use:
    # conda env create -f environment.yml && conda activate openhgnn
    pip install -r requirements.txt
    pip install -e .

* install from pinned source environment

.. code:: bash

    pip install -r requirements.txt
    pip install -e .

**5. Verify the environment and registry**

OpenHGNN v0.9 provides a small command line interface for checking the current
environment and discovering registered capabilities:

.. code:: bash

    openhgnn env
    openhgnn list models
    openhgnn list tasks
    openhgnn list datasets
    openhgnn validate-registry

**6. Docker smoke check**

The repository also contains a Dockerfile for the primary setup:

.. code:: bash

    docker build -t openhgnn:v0.9 .
    docker run --rm openhgnn:v0.9

If installation fails, include the output of ``openhgnn env --format json`` in
your issue report.
