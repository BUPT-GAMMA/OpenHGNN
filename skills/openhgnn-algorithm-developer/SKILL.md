---
name: openhgnn-algorithm-developer
description: Guide OpenHGNN users and contributors through running existing models, reproducing results, debugging Experiment failures, and integrating new heterogeneous graph algorithms with proper model/trainerflow/dataset/task/config/docs/tests. Use when asked how to use OpenHGNN, run a model, choose a task, add a new algorithm, port paper code, write reproduction docs, diagnose registry or DGL errors, or prepare an algorithm contribution before PR review.
---

# OpenHGNN Algorithm Developer

Use this skill as an interactive developer and user guide for OpenHGNN. It is
for helping people use the library or add algorithms correctly before formal PR
review.

## First Actions

1. Inspect live repository state before answering:
   - `git status --short`
   - `python skills/openhgnn-algorithm-developer/scripts/inspect_registry.py`
   - `rg -n "<model>|<task>|<dataset>" openhgnn docs README.md README_EN.md`
2. Identify the user's path:
   - Running or reproducing an existing model.
   - Integrating a new algorithm.
   - Debugging an OpenHGNN failure.
   - Preparing contribution docs/tests.
3. Prefer existing OpenHGNN patterns over new abstractions.

## Route: User Wants to Run a Model

Answer with a minimal reproducible path:

1. Confirm the registered model, task, dataset, and trainerflow.
2. Point to the model page under `docs/source/models/` when available.
3. Give install/environment checks:
   - `openhgnn env`
   - `openhgnn list models`
   - `openhgnn list tasks`
   - `openhgnn validate-registry`
4. Give the smallest runnable command, then the paper reproduction command.
5. Explain expected metric, output location, seed, and common failure modes.

Read `references/user-workflows.md` for detailed user-facing answer patterns.

## Route: Developer Wants to Add an Algorithm

Start with an integration plan before editing:

1. Classify the algorithm task: node classification, link prediction,
   recommendation, node regression, pretrain/fine-tune, or unsupported task.
2. Decide whether it needs:
   - New model only.
   - New trainerflow.
   - New dataset wrapper.
   - New task registry entry.
   - Optional dependencies.
3. Keep model-specific logic in model-specific modules. Public registry files
   should contain thin imports and mapping entries only.
4. Use DGL-native graph construction, message passing, sampling, and layers
   whenever the algorithm operates on graphs.
5. Add docs and a minimal smoke or component test in the same change.

Read `references/integration-checklist.md` before implementing. Use
`references/task-selection.md` when task ownership is unclear.

## Required OpenHGNN Contracts

For model contributions, verify these contracts:

- Model uses `@register_model` and inherits `BaseModel`.
- Model exposes `build_model_from_args(cls, args, hg)` unless the exception is
  documented.
- Trainerflow uses `@register_flow` when a custom flow is needed.
- Dataset uses OpenHGNN dataset registry or a DGL dataset abstraction.
- Device placement is controlled by `Experiment`/trainerflow; avoid hard-coded
  `cuda:0`, `.cuda()`, or `.to("cuda")`.
- Reproduction docs include data source, preprocessing, command, metric,
  expected result, hardware/runtime, and seed.
- Runtime outputs, checkpoints, logs, raw datasets, and TensorBoard files are
  not committed.

## DGL-First Rules

Prefer:

- `dgl.heterograph`, DGL dataset utilities, DGL samplers, `dgl.nn`, and DGL
  graph operators.
- DGL graph features (`g.ndata`, `g.edata`) and canonical edge types.
- Sparse or message-passing implementations over dense adjacency conversion.

Flag or avoid:

- Unnecessary NetworkX/PyG conversions.
- Copied DGL layers without a clear algorithmic difference.
- Trainer-local ad-hoc data loading that bypasses dataset/task contracts.
- Hard-coded machine paths or GPU IDs.

## Documentation and Reproduction

Every new model should have a stable page under `docs/source/models/`.

Use:

```bash
python skills/openhgnn-algorithm-developer/scripts/scaffold_model_doc.py \
  --model MyModel --venue "KDD 2026" --task node_classification \
  --dataset my_dataset --output docs/source/models/mymodel.rst
```

Then fill all placeholders before release. See
`references/reproduction-docs.md` for the required format.

## Debugging Flow

For failures, diagnose in this order:

1. Environment: Python, PyTorch, DGL, CUDA, optional dependencies.
2. Registry: model/task/flow/dataset listed and importable.
3. Config: required args exist and types match.
4. Dataset: files, download, splits, labels, category, features.
5. Graph/device: DGL graph type, node/edge types, CPU/GPU placement.
6. Shape/metric: output dimension, labels, mask, evaluator.

Read `references/troubleshooting.md` for symptoms and commands.

## Validation Commands

Run the smallest relevant set:

```bash
openhgnn validate-registry --format json
python -m sphinx -W -b html docs/source /tmp/openhgnn-docs-build
python -m pytest -c /dev/null -q tests/end2end/test_run.py::<nodeid>
git diff --check
```

If the current checkout lacks the newly merged v0.9 model/task, state that
local smoke tests may skip and should be rerun on the final integration branch.
