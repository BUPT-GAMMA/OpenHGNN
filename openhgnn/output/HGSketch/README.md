# HGSketch

HGSketch is a graph-level heterogeneous graph representation method. It maps
heterogeneous graphs into a low-dimensional Hamming space by extracting
simplicial complexes, building Hodge Laplacian matrices, and applying
locality-sensitive hashing.

## Model Scope

HGSketch is different from ordinary message-passing node classification models:

- It is non-parametric and has no gradient-based training loop.
- It produces graph-level embeddings.
- Downstream classification is handled by a linear classifier over the generated sketches.
- It converts a DGL heterogeneous graph to a homogeneous NetworkX view to extract higher-order simplices.

The NetworkX/scipy usage is part of the HGSketch algorithmic pipeline and should
remain documented as an intentional dependency boundary.

## How to Run

HGSketch is registered as an OpenHGNN model with the dedicated
`HGSketch_trainer` trainerflow.

```bash
python main.py -m HGSketch -t graph_classification -d <graph_dataset> -g -1
```

OpenHGNN does not currently provide a general `graph_classification` task in the
main task registry. Before this command can become an end-to-end release smoke,
the task and dataset contract for graph classification should be added to
OpenHGNN or the trainer should be wired to an existing supported task.

## Implementation Notes

The HGSketch implementation follows these steps:

1. Convert the DGL heterogeneous graph to an undirected NetworkX graph.
2. Extract k-simplices and construct Hodge Laplacian matrices.
3. Initialize heterogeneous features with node-type one-hot encodings.
4. Apply local information amplification and global structure enhancement.
5. Use iterated locality-sensitive hashing to create binary graph sketches.
6. Linearize sketches for downstream linear classification.

Key hyperparameters are configured in `openhgnn/config.ini`:

```ini
K = 2
R = 3
D = 128
seed = 0
max_epoch = 1
```

## Testing

The PR includes component tests in `tests/test_hgsketch.py`. These tests cover
registration, configuration, heterogeneous graph conversion, simplex extraction,
sketch computation, deterministic behavior, and linearization.

## Release Follow-up

To make HGSketch release-ready, OpenHGNN should either:

- Add a formal `graph_classification` task and a tiny graph dataset fixture,
  then register an end-to-end smoke in `tests/end2end/test_run.py`.
- Explicitly scope HGSketch as a component-level graph representation method
  and keep its release validation in unit/component tests.
