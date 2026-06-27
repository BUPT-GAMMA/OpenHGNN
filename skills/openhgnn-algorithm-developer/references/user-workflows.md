# User Workflows

Use these patterns when helping OpenHGNN users.

## Run an Existing Model

Answer with:

1. Environment setup.
2. Registry checks.
3. Model docs link.
4. Minimal command.
5. Expected metric and output.
6. Common fixes.

Example:

```bash
openhgnn env
openhgnn list models
openhgnn list tasks
python main.py -m HAN -d acm_han_raw -t node_classification -g 0 --use_best_config
```

## Find Models for a Task

1. Use `openhgnn list tasks`.
2. Search `tests/end2end/test_run.py` for smoke-covered examples.
3. Search `docs/source/models/` and `README.md` for task tables.
4. Prefer models with docs, expected metrics, and smoke tests.

## Reproduce a Model

1. Start from `docs/source/models/<model>.rst`.
2. Install optional dependencies listed there.
3. Use the documented dataset and preprocessing.
4. Run the documented command.
5. Compare metric, seed, hardware, and config.

## Explain OpenHGNN Architecture

Use this short mental model:

- `Experiment` is the user entry point.
- `task` defines the supervised or self-supervised problem contract.
- `dataset` loads graph data and labels/splits.
- `model` defines neural computation.
- `trainerflow` owns the training/evaluation loop.
- `config` provides defaults.

## User-Facing Troubleshooting

Keep answers concrete. Ask for the command and traceback only when needed.
Always mention the exact registry, config, or dataset path to inspect next.
