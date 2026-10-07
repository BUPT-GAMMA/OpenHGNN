# LMSPS[NeurIPS 2024]

- paper: [Long-range Meta-path Search through Progressive Sampling](https://arxiv.org/abs/2409.12011)
- Code from author: [LMSPS](https://github.com/JHL-HUST/LMSPS)

## Model

Registered model name: `LMSPS`, trained by the flow `LMSPS_trainer`.

LMSPS searches over all meta-paths that start from the target node type
instead of a hand-picked set, and shrinks the candidate set progressively so
that long-range meta-paths stay affordable.  The search and the final training
are two stages: `LMSPS_SearchNet` (super-net, paper Section 5.1) is searched
first, then `LMSPS` (target-net, Eq. 7) is trained on the meta-paths that the
sampling evaluation selected.  The target-net is a pure MLP architecture
without neighbour or semantic attention, as required by Section 5.3 of the
paper.

## Dataset

Supported dataset: `ogbn-mag` (Open Graph Benchmark, MAG version).  The first
run downloads and preprocesses it into `openhgnn/dataset/data/`; the propagated
meta-path features are then built by the flow itself, so no extra file has to
be prepared by hand.

## Task and Metric

OpenHGNN task name: `node_classification`.  Reported metric: test accuracy on
the paper nodes of `ogbn-mag`, with validation accuracy for model selection.

## Environment

- OpenHGNN v0.9.0 (commit `efc3923`), Python 3.9, PyTorch 2.5.1+cu121,
  DGL 2.2.1+cu121 (CUDA 12.6 toolkit), a single NVIDIA RTX 3090.
- Log file: `openhgnn/output/LMSPS/LMSPS-<TIME>.log`.

## Command

```
python main.py -m LMSPS -d ogbn-mag -t node_classification -g 0
```

If you do not have a gpu, set `-g -1`.

The pipeline propagates the features of all `K = 226` meta-paths of up to six
hops in a single pass, before the search starts, and keeps them in main memory,
which is what the reference implementation does as well.  That propagation is
the memory peak of the pipeline and it is paid in full before any meta-path has
been selected, so setting `meta_paths` in the `[LMSPS]` section of
`openhgnn/config.ini` to a comma-separated meta-path list (for example the
meta-paths reported in Table 9 of the paper) skips the *search* and trains the
target-net on that list directly, but it does not lower the peak.  Lower
`num_hops` to reduce it.

## Expected Result

Node classification on `ogbn-mag`:

| Method | Validation accuracy | Test accuracy |
| ------ | ------------------- | ------------- |
| Paper (Table 1) | - | 54.83±0.20 |
| Reference implementation (authors' code, 10 seeds) | - | 53.28±0.14 |
| OpenHGNN (search + train, seed 1) | 47.8784 | 47.1590 |
| OpenHGNN (paper meta-paths, seed 1) | 54.1685 | 53.4348 |

The reference implementation row is the mean and standard deviation over 10
seeds of the authors' code run with `--arch ogbn_withoutLabel`, i.e. the
meta-paths of Table 9 of the paper; the authors' ogbn-mag command does not
re-run the search, and mixed precision is off as in the reference runs.  The
first OpenHGNN row is a single run (`seed = 1`) of the default two-stage
pipeline (search, then train): the search is stochastic and a single search
can land on a weaker meta-path set.  The second OpenHGNN row fixes the
meta-paths to Table 9 via `meta_paths` and trains the target-net only; it
lands inside the reference distribution, which shows the training stage of
this port matches the reference implementation.

Seed: `seed = 1` in the `[LMSPS]` section of `openhgnn/config.ini`, single run.

## Notes

- **Scope.**  This version covers the `ogbn-mag` pipeline without label
  propagation, i.e. the `54.83` entry of Table 1 of the paper.  Label
  propagation and multi-stage self-training (the `+LP+MS` entry of Table 1) are
  not included.
- **Memory.**  The features of all `K` meta-paths are propagated in one pass
  and held in main memory while the search runs (`K = 226` for `ogbn-mag` at
  6 hops; the reference implementation peaks at about 255 GB of host memory on
  a 256 GB machine, and this port follows the same propagation scheme).  That
  peak is not reduced by `meta_paths`, which only skips the search; lower
  `num_hops` to reduce it.
- **Meta-path order.**  The meta-paths are sorted before the search, so that
  index `i` of the architecture parameters `alpha` refers to a fixed meta-path
  across runs.
- **Hyper-parameters.**  The defaults in the `[LMSPS]` section of
  `openhgnn/config.ini` follow the reference implementation: 6 hops,
  `hidden = 512`, `embed_size = 256`, `lr = 3e-3`, no weight decay, 200 search
  epochs with `M = 30` meta-paths activated per iteration out of a compact
  space of `V = 60`; the Gumbel temperature is annealed from 8 to 4 and
  `lambda` is kept at 1 for the first 20 epochs before decaying to 0.

## More

#### Smoke Test

The unit tests cover the search space, both samplers and both networks on
synthetic features, and run on CPU in a few seconds:

```
pytest tests/test_lmsps.py -v
```
#### Contributor

15803723000 [GAMMA LAB]

#### If you have any questions,

Submit an issue or email to 766564132@qq.com
