# Integration Checklist

Use this checklist before writing or modifying OpenHGNN algorithm code.

## Model

- Add the model under `openhgnn/models/<Model>.py`.
- Inherit `BaseModel`.
- Add `@register_model("<Model>")`.
- Implement `build_model_from_args(cls, args, hg)`.
- Keep pure neural layers in the model file or local helpers.
- Do not parse command-line args inside the model.
- Do not load datasets inside the model.
- Avoid hard-coded devices.

## Trainerflow

Add a trainerflow only when an existing flow cannot express the training loop.

- Add `openhgnn/trainerflow/<model>_trainer.py`.
- Inherit `BaseFlow`.
- Add `@register_flow("<flow_name>")`.
- Route training, validation, test, logging, and device movement through the
  flow.
- Keep model-specific loss logic in the flow if it depends on sampling,
  pretraining, or multiple optimizers.
- Register the flow in `openhgnn/trainerflow/__init__.py`.

## Dataset

- Reuse an existing dataset wrapper when possible.
- Add a dataset module only when the data contract is new.
- Prefer DGL graph objects and DGL dataset utilities.
- Store only small metadata/configuration in the repo.
- Do not commit raw datasets, downloaded archives, generated embeddings, logs,
  checkpoints, or TensorBoard events.
- Document download location and preprocessing.

## Task

Add a task only when the existing task interface does not fit.

- Register through `openhgnn/tasks/__init__.py`.
- Define the graph, labels, masks/splits, loss, and metric contract.
- Add task docs in `docs/source/get_started/task_overview.rst`.
- Add at least one smoke path.

## Config

- Add model defaults in the repository's established config location.
- Keep defaults runnable on a small dataset.
- Avoid hidden machine-local paths.
- Include seed, `max_epoch`, hidden size, dropout, lr, and task-specific
  metrics when relevant.

## Docs

- Add or update `docs/source/models/<model>.rst`.
- Link it from `docs/source/models/index.rst`.
- Include paper, venue, registered name, task, dataset, preprocessing,
  command, metric, expected result, hardware/runtime, seed, and limitations.

## Tests

- Add an end-to-end smoke via `Experiment(..., max_epoch=1)` when data and
  dependencies are lightweight.
- Add component tests when the full pipeline requires large data or optional
  dependencies.
- Run registry validation and Sphinx strict build.
