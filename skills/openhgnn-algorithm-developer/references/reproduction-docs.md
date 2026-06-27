# Reproduction Documentation

Every release-quality model page should answer the questions below.

## Required Fields

- Paper and venue.
- Registered model name.
- Supported task(s).
- Dataset source.
- Preprocessing.
- Command.
- Metric.
- Expected result.
- Hardware/runtime.
- Seed.
- Known limitations.

## Model Page Template

```rst
ModelName
=========

* Paper: <paper title>, <venue year>
* Registered model name: ``ModelName``
* Task: ``node_classification``

Reproduction
------------

.. list-table::
   :header-rows: 1

   * - Field
     - Value
   * - Data source
     - <where the dataset comes from>
   * - Preprocessing
     - <how raw data becomes DGL/OpenHGNN inputs>
   * - Command
     - ``python main.py -m ModelName -d dataset_name -t task_name -g 0 --use_best_config``
   * - Metric
     - <metric name>
   * - Expected result
     - <mean/std or representative validated result>
   * - Hardware/runtime
     - <GPU/CPU, memory, runtime if known>
   * - Seed
     - ``seed = 0``

Implementation Notes
--------------------

<DGL usage, optional dependencies, limitations>
```

## Policy

- Do not use "see README" as the only reproduction instruction.
- Do not claim paper-level reproduction when expected result, seed, or dataset
  preprocessing is missing.
- If a full benchmark is not practical in CI, state the limitation and provide
  a component or smoke validation path.
