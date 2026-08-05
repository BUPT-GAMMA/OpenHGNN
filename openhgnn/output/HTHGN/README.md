# HTHGN

**HTHGN: Heterogeneous Temporal Hypergraph Neural Network**

- **Official implementation:** https://github.com/huanliucs/HTHGN
- **Reference commit:** `b8195861308e042afcf9471cde875a0a577a1638`
- **Task:** Temporal link prediction
- **Supported dataset:** Yelp

## Overview

HTHGN models both heterogeneous temporal relations and higher-order
interactions. It first constructs a heterogeneous temporal hypergraph and
then learns node representations through hierarchical spatial and temporal
attention.

The OpenHGNN integration contains:

1. relation-specific heterogeneous message passing based on DGL
   `GATv2Conv`;
2. semantic attention over heterogeneous relations;
3. temporal self-attention over graph snapshots;
4. gated residual fusion;
5. an MLP link predictor;
6. public-code-compatible temporal link and new-link evaluation.

The current integration supports the Yelp dataset released by the authors.
DBLP and AMiner are not registered because their complete preprocessed files
were not available for verification during this reproduction.

## OpenHGNN integration

HTHGN is integrated through the standard OpenHGNN interfaces:

- `openhgnn/models/HTHGN.py`
  - registered with `@register_model("HTHGN")`;
  - implements the HTHGN encoder and link predictor;
- `openhgnn/dataset/hthgn_dataset.py`
  - registered with `@register_dataset("hthgn_dataset")`;
  - loads the temporal Yelp graphs and metapath2vec features;
  - constructs bidirectional label graphs and temporal hypergraphs;
- `openhgnn/trainerflow/hthgn_trainer.py`
  - registered with `@register_flow("hthgn_trainer")`;
  - implements training, checkpoint selection and evaluation;
- `openhgnn/config.ini`
  - contains the default HTHGN hyperparameters;
- `tests/test_hthgn.py`
  - provides self-contained registration, sampling and
    forward/backward smoke tests.

## Data preparation

The authors' released Yelp data contains two required files:

```text
yelp.bin
mp2vec/yelp_mpfeat.bin
````

Clone the official repository:

```bash
git clone https://github.com/huanliucs/HTHGN.git
cd HTHGN
git checkout b8195861308e042afcf9471cde875a0a577a1638
```

Copy the files into OpenHGNN:

```bash
cd /path/to/OpenHGNN

mkdir -p \
  openhgnn/dataset/HTHGN/mp2vec \
  openhgnn/dataset/HTHGN/cache

cp \
  /path/to/HTHGN/data/yelp.bin \
  openhgnn/dataset/HTHGN/yelp.bin

cp \
  /path/to/HTHGN/data/mp2vec/yelp_mpfeat.bin \
  openhgnn/dataset/HTHGN/mp2vec/yelp_mpfeat.bin
```

The expected directory structure is:

```text
openhgnn/dataset/HTHGN/
├── yelp.bin
├── mp2vec/
│   └── yelp_mpfeat.bin
└── cache/
```

The binary data, generated hypergraph cache and model checkpoints should not
be committed to Git.

## How to run

Run the model through the standard OpenHGNN entry point:

```bash
python main.py \
  -m HTHGN \
  -d yelp4HTHGN \
  -t link_prediction \
  -g 0
```

The main experiment parameters are defined in the `[HTHGN]` section of
`openhgnn/config.ini`.

The validated configuration is:

```ini
[HTHGN]
seed = 0
lr = 0.001
weight_decay = 0.0005
max_epoch = 300
patience = 1000

hidden_dim = 32
num_layers = 2
num_heads = 4
dropout = 0.2
norm = False

time_window = 3
prediction_length = 3
negative_count = 5
edge_ratio = 1.0
eval_negative_ratio = 0.3

k = 3
p = 100
hyperedge_type = ring
hypernode_type = u
keep_original = True

num_repeats = 1
mini_batch_flag = False

data_dir = ./openhgnn/dataset/HTHGN
cache_dir = ./openhgnn/dataset/HTHGN/cache
```

For multi-seed evaluation, run independent processes after changing `seed`
to `0`, `1`, `2`, `3` and `4`. Independent processes ensure that model
initialization, training sampling and evaluation sampling are regenerated
for each seed.

## Public-code reproduction protocol

This implementation reproduces the observable behavior of the authors'
released code.

### Temporal windows

Yelp contains 10 temporal snapshots. With `time_window = 3`, the public code
uses the following sliding windows:

```text
snapshots 0,1,2 -> target 3
snapshots 1,2,3 -> target 4
snapshots 2,3,4 -> target 5
snapshots 3,4,5 -> target 6
snapshots 4,5,6 -> target 7
snapshots 5,6,7 -> target 8
snapshots 6,7,8 -> target 9
```

All available windows participate in training. Therefore, the results in this
README represent the released public-code protocol rather than a separate
leakage-free temporal split.

### Hypergraph construction

Before constructing the temporal hypergraph, the original canonical
relations are augmented with reverse relations.

The validated Yelp hypergraph configuration is:

```text
k = 3
hyperedge type = ring
hypernode type = u
p = 100
keep original relations = True
```

`ring` uses nodes reachable at exactly the selected hop distance rather than
combining all distances from one to `k`.

### Link-prediction supervision

Hyperedge relations participate in encoder message passing but are excluded
from both positive and negative link-prediction supervision.

The training objective uses:

```text
positive edge ratio = 1.0
negative edges per positive edge = 5
```

The ordinary evaluation protocol uses all positive edges and approximately
`0.3` as many negative edges. This creates a positive prevalence of roughly
`1 / 1.3`, so a random-ranking AP can be substantially higher than `0.5`.

DGL may return fewer negative edges than requested for dense or small
relations. The OpenHGNN implementation samples only from the negative
candidates actually returned by DGL, preventing out-of-bounds indexing while
preserving the requested ratio whenever enough candidates are available.

### New-link definition

A new link is an edge that exists in the current snapshot but does not exist
in the immediately preceding snapshot.

It is not defined as an edge that has never appeared in the complete earlier
history.

### Checkpoint evaluation

The public training behavior monitors the loss of the final sliding window
for early stopping.

After loading the best checkpoint, this OpenHGNN implementation recomputes
the node embeddings before evaluation. This avoids evaluating stale
embeddings generated before the best checkpoint was restored.

## Reproduction results

### Five-seed OpenHGNN results

The following results use 300 epochs and seeds `0` through `4`.

|           Seed | Best epoch |         Link AUC |          Link AP |     New Link AUC |      New Link AP |
| -------------: | ---------: | ---------------: | ---------------: | ---------------: | ---------------: |
|              0 |        299 |            80.27 |            92.49 |            67.63 |            70.58 |
|              1 |        289 |            78.88 |            91.98 |            63.39 |            68.05 |
|              2 |        291 |            77.46 |            91.23 |            62.67 |            66.13 |
|              3 |        285 |            77.76 |            91.40 |            62.42 |            67.32 |
|              4 |        255 |            77.23 |            91.39 |            68.69 |            70.47 |
| **Mean ± std** |          — | **78.32 ± 1.13** | **91.70 ± 0.47** | **64.96 ± 2.65** | **68.51 ± 1.76** |

The standard deviations above use NumPy population standard deviation
(`numpy.std`, `ddof=0`), matching the aggregation behavior used in the
released implementation.

### Comparison with the paper

| Protocol             |         Link AUC |          Link AP |     New Link AUC |      New Link AP |
| -------------------- | ---------------: | ---------------: | ---------------: | ---------------: |
| Paper, Yelp HTHGN    |     74.04 ± 4.82 |     89.56 ± 3.01 |     64.93 ± 1.91 |     68.47 ± 1.42 |
| OpenHGNN, five seeds | **78.32 ± 1.13** | **91.70 ± 0.47** | **64.96 ± 2.65** | **68.51 ± 1.76** |

The OpenHGNN means are within or very close to the ranges reported by the
paper. The New Link AUC and AP means are nearly identical to the paper
values.

### Intermediate numerical alignment

The first-epoch accumulated loss is closely aligned with the authors'
released implementation:

```text
Authors' implementation: 4.1524
OpenHGNN:               4.1533
Absolute difference:   0.0009
```

A 20-epoch intermediate comparison produced:

| Implementation          | Link AUC | Link AP | New Link AUC | New Link AP |
| ----------------------- | -------: | ------: | -----------: | ----------: |
| Authors' implementation |    61.69 |   82.16 |        46.51 |       47.25 |
| OpenHGNN                |    59.43 |   80.91 |        46.33 |       46.69 |

These checks were used to verify the data path, model computation, training
loss and evaluation pipeline before the final five-seed experiment.

## Tests

Run the self-contained HTHGN smoke tests with:

```bash
python -m pytest \
  -o addopts='' \
  -q tests/test_hthgn.py
```

Expected result:

```text
3 passed
```

The smoke tests verify:

1. model, dataset and trainerflow registration;
2. safe handling when DGL returns fewer negative samples than requested;
3. HTHGN forward propagation across three temporal snapshots;
4. zero initialization for hypernodes without original input features;
5. link-predictor output;
6. backward propagation and finite gradients.

The tests use small synthetic DGL graphs only as test fixtures. The default
training path always loads the released Yelp data.

## Verified environment

The reproduction was validated with:

```text
Operating system: Linux under WSL2
Python:           3.11.15
PyTorch:          2.4.0+cu121
DGL:              2.4.0+cu121
PyG:              2.6.1
GPU:              NVIDIA GeForce RTX 4060 Laptop GPU
```

The authors' reference environment used older PyTorch and DGL versions.
Small seed-level numerical differences are therefore expected.

## Limitations

* Only the Yelp dataset is currently registered and validated.
* DBLP and AMiner support is not claimed in this integration.
* The data files are obtained separately from the authors' repository.
* The reported results follow the authors' released public-code protocol.
* Strict leakage-free temporal evaluation is outside the scope of this PR.

## Contributor

* `zhoujun1zzz`
