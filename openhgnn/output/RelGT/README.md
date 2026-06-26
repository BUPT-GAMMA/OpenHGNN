# RelGT [arXiv 2025]

- **Paper**: [Relational Graph Transformer](https://arxiv.org/abs/2505.10960)
- **Task**: RelBench supervised prediction tasks
- **Trainerflow**: `RelGT_trainer`

## Dependencies

RelGT follows the RelBench data pipeline and requires optional dependencies that
are not needed by the core OpenHGNN models:

```bash
pip install relbench pytorch-frame torch-geometric sentence-transformers h5py einops
```

## How to Run

```bash
python main.py -m RelGT -t link_prediction -d rel-f1 -g 0 --use_best_config
```

The default configuration uses:

- `relbench_dataset = rel-f1`
- `relbench_task = driver-top3`
- `gt_conv_type = full`
- `num_neighbors = 100`

RelGT downloads RelBench data and materializes token caches under the configured
cache directory during the first run. Subsequent runs reuse the precomputed HDF5
neighbor-token files.

## OpenHGNN Integration Notes

RelGT is integrated as a model-specific trainerflow because RelBench constructs
PyG/TorchFrame relational tables instead of a standard DGL heterogeneous graph.
The OpenHGNN entry points are still registered through `Experiment`, model
registry, and trainerflow registry.

The included unit tests focus on local model components, token collation, and
registry wiring with synthetic tensors so they can run without downloading
RelBench datasets.
