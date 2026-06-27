Model PR Checklist
==================

OpenHGNN v0.9 model pull requests must be reviewable through the same
framework contract as existing models. A new model should not be merged as a
standalone paper-code dump.

Required Structure
------------------

Every model PR should include:

* A model class under ``openhgnn/models/`` that inherits ``BaseModel`` and uses
  ``@register_model``.
* A ``build_model_from_args(cls, args, hg)`` implementation unless the PR
  explicitly documents why the model cannot consume a DGL graph.
* A trainerflow registered with ``@register_flow`` or a clear statement that an
  existing trainerflow is reused without modification.
* Dataset support through ``BaseDataset``/``DGLDataset`` and the OpenHGNN
  dataset registry. Model-specific dataset logic should live in a dedicated
  dataset module, not in a long block inside a shared dataset file.
* Config defaults in ``openhgnn/config.ini`` and config parsing in
  ``openhgnn/config.py`` only for parameters that cannot be handled by existing
  defaults.
* Documentation with data source, preprocessing, command line example, expected
  task, reported metric, seed policy, environment, and expected result. Use
  :doc:`../get_started/reproduce_model` as the minimum reproduction standard.
  During the v0.9 transition, legacy
  ``openhgnn/output/<MODEL>/README.md`` files are still present for some merged
  models, but new release documentation should prefer ``docs/source`` or an
  example README that can be linked from the documentation site.
* A smoke test or an end-to-end test entry that exercises the model through
  ``Experiment(...)`` or the command-line entry point.

DGL Implementation Rules
------------------------

Prefer DGL-native implementation whenever possible:

* Construct graphs with ``dgl.heterograph`` or existing DGL dataset utilities.
* Use DGL graph operators, samplers, message passing, and ``dgl.nn`` layers when
  they match the paper method.
* Avoid converting to NetworkX, PyG, scipy adjacency matrices, or custom dense
  graph loops unless the algorithm requires it. If a conversion is necessary,
  document the reason and keep the boundary small.
* Do not hard-code devices such as ``cuda:0``. Use ``args.gpu`` and the device
  assigned by the trainerflow.

Repository Hygiene
------------------

The following files should not be committed in model PRs:

* Runtime files under ``openhgnn/output/**``. If a PR still uses a legacy
  model README under ``openhgnn/output/<MODEL>/README.md``, it must contain
  reproducible model documentation only and must not include logs, checkpoints,
  generated metrics, or TensorBoard events.
* TensorBoard event files, checkpoints, model weights, logs, and run outputs.
* Dataset archives or extracted raw data under ``openhgnn/dataset/**``.
* Root-level ad-hoc scripts such as ``main.py``, ``debug.py``, ``check.py``, or
  ``test_model.py``.

Use these locations instead:

* Model documentation: ``docs/source/`` or ``examples/<model>/README.md``.
* Stable command-line helpers: ``scripts/``.
* Tests: ``tests/``.
* Runtime outputs: user-selected output directories, not committed source
  paths.

Review Gate
-----------

Before requesting review, run:

.. code:: bash

   python scripts/validate_pr_hygiene.py --base origin/main --head HEAD
   openhgnn validate-registry --format json

For a model intended to appear in a release note, also provide one smoke command
that uses the public entry point:

.. code:: bash

   python main.py -m MODEL_NAME -d DATASET_NAME -t TASK_NAME -g -1 --use_best_config

For result reproduction, the model documentation must also record environment,
data preprocessing, seed, metric, expected value, and log location. The accepted
format is described in :doc:`../get_started/reproduce_model`.

The hygiene check blocks high-risk artifacts and local-only code. It may print
warnings for ``print()``, ``TODO``, or ``FIXME`` in changed Python files; these
warnings should be resolved before release candidates are merged.
