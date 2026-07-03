# SlotGAT

Paper: SlotGAT: Slot-based Graph Attention for Heterogeneous Graph Representation Learning

This OpenHGNN migration supports two SlotGAT tasks:

- node classification (NC)
- link prediction (LP)

SlotGAT keeps the original paper implementation style: it uses a flattened homogeneous graph, per-node-type feature matrices, and edge-type id tensors instead of the standard OpenHGNN `h_dict` model input.

## How to run

### Node classification

```bash
python main.py -m SlotGAT -t node_classification -d acmSlotGAT -g 0
```

If you do not have GPU, set `-g -1`.

The `SlotGAT` model with task `node_classification` is routed to the dedicated trainerflow:

```text
openhgnn/trainerflow/slotgat_node_classification.py
```

### Link prediction

```bash
python main.py -m SlotGAT -t link_prediction -d slotgat_lp -g 0
```

If you do not have GPU, set `-g -1`.

The `SlotGAT` model with task `link_prediction` is routed to the dedicated trainerflow:

```text
openhgnn/trainerflow/slotgat_link_prediction.py
```

The link prediction dataset should contain the original SlotGAT LP raw files:

- `node.dat`
- `link.dat`
- `link.dat.test`

By default, `slotgat_lp` looks under:

```text
./openhgnn/dataset/SlotGAT/LP
```

You can also set `slotgat_data_path` to a directory containing `node.dat`, `link.dat`, and `link.dat.test`.

## Current support

### Node classification datasets

The current OpenHGNN implementation keeps compatibility with the original `acmSlotGAT` raw files:

- `node.dat`
- `link.dat`
- `label.dat`
- `label.dat.test`

These files are parsed by:

```text
openhgnn/dataset/SlotGAT_nc_dataset.py
```

The node classification trainerflow is:

```text
openhgnn/trainerflow/slotgat_node_classification.py
```

### Link prediction datasets

SlotGAT link prediction is migrated from the original `SlotGAT_ICML23-new/LP` implementation.

The LP raw files are parsed by:

```text
openhgnn/dataset/SlotGAT_lp_dataset.py
```

The link prediction trainerflow is:

```text
openhgnn/trainerflow/slotgat_link_prediction.py
```

The LP trainerflow preserves the original SlotGAT LP training style:

- positive edges from `link.dat`
- test edges from `link.dat.test`
- random negative sampling
- `left`, `right`, `mid` edge triplet inputs
- internal `DistMult` or `Dot` decoder in `SlotGAT`
- AUC and MRR evaluation

## Dataset adaptation

### Node classification adapter

`SlotGAT_nc_dataset.py` can:

- load the original `acmSlotGAT` raw text files, or
- adapt a standard `DGLHeteroGraph` / OpenHGNN heterogeneous dataset into the SlotGAT input format.

The adapter automatically prepares:

- `features_list`
- `e_feat`
- flattened homogeneous graph structure
- node type indexer
- edge type indexer
- `in_dim`
- `num_ntype`
- `num_etypes`

### Link prediction adapter

`SlotGAT_lp_dataset.py` prepares the LP-specific SlotGAT inputs:

- flattened homogeneous graph
- per-node-type `features_list`
- edge-type tensor `e_feat`
- train / validation positive edges
- test candidate edges
- random negative samples
- AUC / MRR evaluation data

## TrainerFlow layout

SlotGAT uses dedicated trainerflows because its model interface differs from standard OpenHGNN models.

Relevant implementation files:

- Model: `openhgnn/models/SlotGAT.py`
- NC dataset adapter: `openhgnn/dataset/SlotGAT_nc_dataset.py`
- LP dataset adapter: `openhgnn/dataset/SlotGAT_lp_dataset.py`
- NC trainerflow: `openhgnn/trainerflow/slotgat_node_classification.py`
- LP trainerflow: `openhgnn/trainerflow/slotgat_link_prediction.py`

The OpenHGNN experiment routing maps:

```text
SlotGAT + node_classification -> slotgat_node_classification
SlotGAT + link_prediction     -> slotgat_link_prediction
```

## Model input format

SlotGAT expects the following inputs during full-batch training:

- a homogeneous graph converted from a heterogeneous graph
- one feature matrix for each node type in `features_list`
- one edge-type id tensor `e_feat`

For link prediction, SlotGAT additionally uses:

- `left`: source node ids
- `right`: destination node ids
- `mid`: relation / edge-type ids

Compared with standard OpenHGNN models that directly consume `h_dict`, SlotGAT still uses its own specialized input structure.

## Hyper-parameters

Important hyper-parameters include:

```python
edge_dim
hid_dim
num_layers
num_heads
feat_drop
attn_drop
negative_slope
alpha
residual
SAattDim
```

For link prediction, additional important options include:

```python
decoder
slot_aggregator
inProcessEmb
l2BySlot
l2use
batch_size
```

Several parameters can be inferred automatically from the adapted dataset:

- `in_dim`
- `num_ntype`
- `num_etypes`

## Notes

- Node classification is best verified on `acmSlotGAT`.
- Link prediction expects the original SlotGAT LP raw-file format.
- Mini-batch training is not the main target path for SlotGAT in the current implementation.
- The legacy `SlotGAT_trainer.py` name is no longer the canonical trainerflow name; NC and LP are now split into dedicated task-specific files.

## More

Recommended next steps:

1. add more documented runnable LP datasets,
2. further align SlotGAT's model interface with standard OpenHGNN models,
3. improve automatic adaptation from standard heterogeneous datasets for both NC and LP.
