# AutoML
## Hyperparameter Optimization(HPO)

We use a lightweight HPO framework [Optuna](https://github.com/optuna/optuna) where we can enjoy high modularity.

Just with a few lines of codes, we can define the whole experiment.

For now, we just  need define the search func in [*func_search(trial)*](./hpo.py). How to define, refer to [Pythonic Search Space](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/002_configurations.html#pythonic-search-space).

### [Efficient Optimization Algorithms](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/003_efficient_optimization_algorithms.html#efficient-optimization-algorithms)

Optuna enables efficient hyperparameter optimization by adopting state-of-the-art algorithms for sampling hyperparameters and pruning efficiently unpromising trials.

#### How to use

We can use these optimization algorithms through passing the parameter when we define [optuna.create_study](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.create_study.html#optuna-create-study).

#### Sampling Algorithms

Optuna provides the following sampling algorithms:

- Tree-structured Parzen Estimator algorithm implemented in [`optuna.samplers.TPESampler`](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.samplers.TPESampler.html#optuna.samplers.TPESampler)
- CMA-ES based algorithm implemented in [`optuna.samplers.CmaEsSampler`](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.samplers.CmaEsSampler.html#optuna.samplers.CmaEsSampler)
- Grid Search implemented in [`optuna.samplers.GridSampler`](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.samplers.GridSampler.html#optuna.samplers.GridSampler)
- Random Search implemented in [`optuna.samplers.RandomSampler`](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.samplers.RandomSampler.html#optuna.samplers.RandomSampler)

The default sampler is [`optuna.samplers.TPESampler`](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.samplers.TPESampler.html#optuna.samplers.TPESampler).

#### Pruning Algorithms

`Pruners` automatically stop unpromising trials at the early stages of the training (a.k.a., automated early-stopping).

Optuna provides the following pruning algorithms:

- Asynchronous Successive Halving algorithm implemented in [`optuna.pruners.SuccessiveHalvingPruner`](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.SuccessiveHalvingPruner.html#optuna.pruners.SuccessiveHalvingPruner)
- Hyperband algorithm implemented in [`optuna.pruners.HyperbandPruner`](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.HyperbandPruner.html#optuna.pruners.HyperbandPruner)
- Median pruning algorithm implemented in [`optuna.pruners.MedianPruner`](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.MedianPruner.html#optuna.pruners.MedianPruner)
- Threshold pruning algorithm implemented in [`optuna.pruners.ThresholdPruner`](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.ThresholdPruner.html#optuna.pruners.ThresholdPruner)

We use [`optuna.pruners.MedianPruner`](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.MedianPruner.html#optuna.pruners.MedianPruner) in most examples, though basically it is outperformed by [`optuna.pruners.SuccessiveHalvingPruner`](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.SuccessiveHalvingPruner.html#optuna.pruners.SuccessiveHalvingPruner) and [`optuna.pruners.HyperbandPruner`](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.HyperbandPruner.html#optuna.pruners.HyperbandPruner) as in [this benchmark result](https://github.com/optuna/optuna/wiki/[Under-Construction]-Benchmarks-with-Kurobako).