# HG-Adapter

- Paper: [HG-Adapter: Improving Pre-Trained Heterogeneous Graph Neural Networks with Dual Adapters](https://proceedings.iclr.cc/paper_files/paper/2025/hash/870c1e0589822bf37590b84984c345c4-Abstract-Conference.html) (ICLR 2025).
- Author implementation: [YujieMo/HG-Adapter](https://github.com/YujieMo/HG-Adapter).

HG-Adapter fine-tunes two lightweight adapters over frozen homogeneous and
heterogeneous pre-trained embeddings for node classification.  This OpenHGNN
port keeps the original data split and treats the four frozen tensors as
versioned external inputs rather than silently replacing them with predictions.

## Installation and assets

HG-Adapter uses the original four benchmark packages and four frozen tensors per
dataset.  They are deliberately not versioned in this repository.  Set
`--hg_adapter_root <path>` (or use the default `./openhgnn/dataset/HGAdapter`)
with this layout:

```
<root>/raw/{ACM,DBLP,Yelp}/...original extracted author files...
<root>/raw/Aminer/raw/...original Aminer files...
<root>/artifacts/<dataset>/manifest.json
<root>/artifacts/<dataset>/<four tensor files>
```

`manifest.json` must declare the original dataset name, the target-node count,
`node_order: "source_flat_target_order_v1"`, and a SHA256 for each of
`hom_before`, `hom_after`, `het_before`, and `het_after`.  Loading fails before
training if a file, checksum, tensor shape, or node-order contract is wrong.

### Supported datasets and source layouts

| OpenHGNN dataset ID | Original name | Target node type | Required extracted data path |
| --- | --- | --- | --- |
| `acm4HGAdapter` | ACM | `p` | `<root>/raw/ACM/{edges,labels,node_features}.pkl` |
| `dblp4HGAdapter` | DBLP | `a` | `<root>/raw/DBLP/{edges,labels,node_features}.pkl` |
| `yelp4HGAdapter` | Yelp | `b` | `<root>/raw/Yelp/{edges,labels,node_features,meta_data}.pkl` |
| `aminer4HGAdapter` | Aminer | `p` | `<root>/raw/Aminer/raw/{features_0,features_1,features_2,labels}.npy` plus `pa.txt`, `pr.txt`, and `*_20.npy` splits |

The loader converts the original flattened feature order into a DGL
heterograph, stores features as `g.nodes[ntype].data['h']`, labels as
`g.nodes[target].data['label']`, and preserves the original train/validation/
test splits as boolean masks.  It supports only these source formats; it does
not substitute similarly named OpenHGNN datasets.

## How to run

```bash
python main.py -m HGAdapter -t node_classification -d acm4HGAdapter -g 0 --use_best_config
python main.py -m HGAdapter -t node_classification -d dblp4HGAdapter -g 0 --use_best_config
python main.py -m HGAdapter -t node_classification -d yelp4HGAdapter -g 0 --use_best_config
python main.py -m HGAdapter -t node_classification -d aminer4HGAdapter -g 0 --use_best_config
```

Use `-g -1` for the CPU smoke run.  Results are Macro-F1 and Micro-F1 selected
by validation Macro-F1 with seed 0; the best checkpoint is written under
`openhgnn/output/HGAdapter/`.

## Performance and reproduction status

| Dataset | Historical author-package run Macro-F1 | Historical author-package run Micro-F1 | Current OpenHGNN run |
| --- | --- | --- | --- |
| ACM | 0.927132 | 0.926588 | CPU, seed 0, 71 epochs early stop: **0.901113 / 0.899294** (Macro/Micro); current port, not paper-equivalent yet |
| DBLP | 0.941121 | 0.947847 | CPU, seed 0, 299 epochs early stop: **0.871362 / 0.888694**; current port, not paper-equivalent yet |
| Yelp | 0.932821 | 0.927507 | CPU, seed 0, 67 epochs early stop: **0.918016 / 0.910626**; current port, not paper-equivalent yet |
| Aminer | 0.793907 | 0.887000 | **Blocked**: authors did not publish the four pre-trained tensors; historical value is saved-prediction evaluation, not a full training run |

Saved predictions are not accepted as a substitute for the four frozen
pre-training inputs.  Record the hardware, exact artifact checksums and the
mean/std over the chosen fixed-seed runs before comparing with the paper.

### Aminer artifact audit

The [author repository](https://github.com/YujieMo/HG-Adapter) publishes only
[one embedding archive](https://raw.githubusercontent.com/YujieMo/HG-Adapter/main/pre_trained_embedding/pre_trained_embedding.7z).
Its SHA256 is `da36b70f23b164a886c302e33588d780617bc01bcae442afe282bff48f48b812`
and it contains exactly the 12 ACM/DBLP/Yelp tensors; it has no
`{emb_hom, embs_het, x_emb, vec_list}_Aminer.pt`.  The repository has no
release assets and contains only `saved_model/prediction_Aminer.pt`, which is
not accepted by this port as a retraining input.  Aminer can be run only after
the authors provide all four source embeddings (or a provenance-equivalent
retraining pipeline).

## TrainerFlow

`HGAdapterTrainer` is registered as `hg_adapter_trainer` and selected by
`Experiment(model='HGAdapter', task='node_classification', ...)`.  It:

1. loads the native HG-Adapter dataset adapter and freezes the checked
   pre-training tensors;
2. builds the registered `HGAdapter` model and optimizes only its adapter,
   reconstruction, attention, and classifier parameters;
3. computes classification, reconstruction, margin-ranking, and graph
   smoothness losses each epoch;
4. chooses the checkpoint using validation Macro-F1, then reports test
   Macro-F1 and Micro-F1.

## Model

The homogeneous branch transforms `hom_before`, propagates it over a learned
soft similarity graph, and adds it to `hom_after`.  The heterogeneous branch
maps every target-aligned tensor in `het_before`, then uses learned attention
to combine them and adds `het_after`.  Their concatenation is classified by a
linear head.  The loss is:

`cross_entropy + lambda_reconstruction * reconstruction + lambda_margin * margin + 0.01 * smoothness`.

The source implementation is in `openhgnn/models/HGAdapter.py`; pre-training
artifact loading is intentionally kept in the dataset adapter.

## Hyperparameters

| Dataset | lr | epochs | embedding dim | homogeneous bottleneck | heterogeneous bottleneck | dropout | lambda reconstruction | lambda margin | margin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ACM | 0.0075 | 200 | 256 | 8 | 16 | 0.1 | 0.01 | 0.1 | 1.0 |
| DBLP | 0.00025 | 1000 | 512 | 2 | 4 | 0.1 | 0.01 | 0.1 | 1.0 |
| Yelp | 0.01 | 500 | 128 | 4 | 8 | 0.1 | 0.01 | 0.1 | 1.0 |
| Aminer | 0.0005 | 500 | 256 | 4 | 64 | 0.1 | 0.01 | 0.1 | 1.0 |

The common optimizer is Adam with weight decay `0.0`, seed `0`, and patience
`30`.  Values are configured in `openhgnn/config.ini` and
`openhgnn/utils/best_config.py`.

## Related DGL APIs

- [`dgl.heterograph`](https://www.dgl.ai/dgl_docs/generated/dgl.heterograph.html): constructs the typed source graph.
- [`DGLHeteroGraph.nodes`](https://www.dgl.ai/dgl_docs/generated/dgl.DGLGraph.nodes.html): stores per-node-type `h`, labels, and masks.
- [`DGLHeteroGraph.to`](https://www.dgl.ai/dgl_docs/generated/dgl.DGLGraph.to.html): moves the graph to the chosen OpenHGNN device.
- [`dgl.load_graphs`](https://www.dgl.ai/dgl_docs/generated/dgl.load_graphs.html): relevant when materializing a processed graph cache; the current adapter builds directly from the author files to preserve ordering.

## More

### Contributor
- Jiayi Ji [BUPT]

### Contact
- If you have any issues, contact me with 2024213756@bupt.cn.
