# Task Selection

Use this when deciding where a model belongs.

## Existing Tasks

- `node_classification`: Predict discrete labels for nodes. Requires target
  node type/category, labels, train/valid/test masks, and classification
  metric.
- `link_prediction`: Predict edges or knowledge graph triples. Requires edge
  splits, negative sampling or ranking protocol, and AUC/MRR/Hits-style metric.
- `recommendation`: Predict user-item or entity association rankings. Requires
  ranking splits and recommendation metrics such as Recall/NDCG/MRR/Hits.
- `node_regression`: Predict continuous node labels. Requires target node type,
  float labels, masks/splits, regression loss, and MAE/RMSE-style metric.
- `pretrain`: Use when the upstream stage is separable from downstream
  fine-tuning and needs reusable checkpoints or embeddings.

## Unsupported or Special Cases

- `graph_classification`: Do not present as fully supported unless a registry
  entry and dataset contract exist in the current checkout. Use component tests
  or add a formal task first.
- Relational table benchmarks such as RelBench may need a model-specific
  trainerflow and optional dependencies. Keep this documented as an exception
  to the default DGL graph path.
- Non-parametric algorithms may be valid but should document why no standard
  gradient training loop exists.

## Decision Questions

1. What object is predicted: node, edge/triple, user-item relation, graph, or
   pretraining target?
2. Are labels discrete, continuous, or rankings?
3. Does an existing dataset wrapper already expose the required masks/splits?
4. Can an existing trainerflow handle the loop?
5. Is a new task truly necessary, or is this a model-specific flow?
