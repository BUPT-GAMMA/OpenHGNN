Reproduce a Model
=================

This page describes the standard OpenHGNN path for reproducing a model result.
Use it when you want to rerun a released model, check a model contribution, or
write a reproducible model README.

1. Prepare the Environment
--------------------------

Use the pinned v0.9 environment when you need comparable results across
machines:

.. code:: bash

   git clone https://github.com/BUPT-GAMMA/OpenHGNN
   cd OpenHGNN
   conda create -n openhgnn python=3.11
   conda activate openhgnn
   pip install -r requirements.txt
   pip install -e .

Then check the local package and dependency versions:

.. code:: bash

   openhgnn env --format json
   openhgnn validate-registry --format json

The primary v0.9 profile is Python 3.11, PyTorch 2.4.0, DGL 2.4.0+cu121, and
CUDA 12.1. CPU runs are useful for smoke tests, but paper-level numbers should
usually be reproduced on the GPU setting used by the model README or paper.

2. Find the Model, Task, and Dataset Names
------------------------------------------

OpenHGNN experiments are identified by three public names:

.. code:: bash

   openhgnn list models
   openhgnn list tasks
   openhgnn list datasets

Use the exact registered names in the command line:

.. code:: bash

   python main.py -m MODEL_NAME -d DATASET_NAME -t TASK_NAME -g GPU_ID

For example:

.. code:: bash

   python main.py -m RGCN -d aifb -t node_classification -g -1 --use_best_config
   python main.py -m SEHTGNN -d sehtgnn_covid -t node_regression -g 0 --use_best_config

Use ``-g -1`` for CPU. Use ``-g 0`` or another GPU id for CUDA. Some large
models and datasets are not expected to finish quickly on CPU.

3. Choose the Configuration Source
----------------------------------

OpenHGNN loads configuration in this order:

* ``openhgnn/config.ini`` provides default model parameters.
* ``--use_best_config`` applies the built-in best configuration from
  ``openhgnn/utils/best_config.py`` when the model and dataset have an entry.
* Keyword arguments passed through ``Experiment(...)`` override mutable
  parameters for Python API runs.

For release reproduction, start with the command in the model README. If the
README says to use the best config, run:

.. code:: bash

   python main.py -m MODEL_NAME -d DATASET_NAME -t TASK_NAME -g 0 --use_best_config

For a quick smoke check, reduce the epoch count through the Python API:

.. code-block:: python

   from openhgnn import Experiment

   Experiment(
       model="RGCN",
       dataset="aifb",
       task="node_classification",
       gpu=-1,
       max_epoch=1,
   ).run()

4. Control Randomness
---------------------

``Experiment`` sets ``seed=0`` if no seed is configured. Many models also define
a seed in ``openhgnn/config.ini`` or in their model-specific configuration.
For an exact reproduction attempt, record all of the following:

* OpenHGNN commit hash and package version.
* Python, PyTorch, DGL, CUDA, and GPU model.
* Model, dataset, task, and command line.
* Whether ``--use_best_config`` or ``--load_from_pretrained`` was used.
* Seed value and number of repeated runs.
* Final metric and the log file path.

For paper-level comparison, prefer reporting mean and standard deviation over
multiple seeds when the model README or paper reports multi-run results.

5. Run and Save Logs
--------------------

OpenHGNN writes logs to:

.. code:: text

   openhgnn/output/<MODEL_NAME>/<MODEL_NAME>-<TIME>.log

The log records the configuration object and training/evaluation messages. When
checking a run, keep the command, environment report, and final metric together:

.. code:: bash

   openhgnn env --format json > env.json
   python main.py -m RGCN -d aifb -t node_classification -g 0 --use_best_config

Runtime outputs, checkpoints, TensorBoard events, and downloaded datasets
should not be committed to the repository. They are local reproduction
artifacts.

6. Reproduce with the Python API
--------------------------------

The API entry point is useful when you need controlled overrides:

.. code-block:: python

   from openhgnn import Experiment

   result = Experiment(
       model="RGCN",
       dataset="aifb",
       task="node_classification",
       gpu=0,
       use_best_config=True,
       seed=0,
   ).run()

   print(result)

Only override parameters that are part of the model or trainer configuration.
The immutable experiment identity fields are ``model``, ``dataset``, and
``task``.

7. Minimal Reproduction Checklist
---------------------------------

Before considering a model reproduction complete, verify:

* ``openhgnn env --format json`` matches a supported environment.
* ``openhgnn validate-registry --format json`` passes.
* The command uses registered ``model``, ``dataset``, and ``task`` names.
* The data download or preprocessing step is documented.
* The command can run from a clean checkout.
* The log contains the final metric used in the paper or README.
* The result table states seed, number of runs, metric, and hardware.

8. Model README Template
------------------------

Each model README or documentation page should include:

.. code:: text

   ## Model
   Paper title, venue, and registered model name.

   ## Dataset
   Dataset name, source, preprocessing steps, and expected location.

   ## Task and Metric
   OpenHGNN task name and reported metric.

   ## Environment
   OpenHGNN version or commit, Python, PyTorch, DGL, CUDA, and GPU.

   ## Command
   Exact command line, including -m, -d, -t, -g, --use_best_config,
   --load_from_pretrained, or model-specific flags.

   ## Expected Result
   Metric value, seed or mean/std over seeds, and log file location.

   ## Notes
   Known memory requirements, expected runtime, and deviations from the paper.
