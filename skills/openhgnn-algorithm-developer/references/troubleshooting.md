# Troubleshooting

Use this order for OpenHGNN failures.

## Environment

Commands:

```bash
python -V
python - <<'PY'
import torch, dgl
print(torch.__version__)
print(dgl.__version__)
print(torch.cuda.is_available())
PY
openhgnn env --format json
```

Common causes:

- PyTorch/DGL wheel mismatch.
- Python version outside supported range.
- Missing optional dependencies.
- CUDA visible but incompatible with installed wheels.

## Registry

Commands:

```bash
openhgnn list models
openhgnn list tasks
openhgnn list datasets
openhgnn validate-registry --format json
```

If console script is unavailable in a source checkout:

```bash
python - <<'PY'
from openhgnn.cli import main
raise SystemExit(main(['validate-registry', '--format', 'json']))
PY
```

## Dataset

Check:

- Dataset name spelling.
- Download path.
- Whether raw data is expected outside the repository.
- Target node category.
- Label dtype and shape.
- Train/valid/test masks or split indices.

## Config

Check:

- Model section exists.
- Required args are present.
- `out_dim` matches label dimension or number of classes.
- `max_epoch=1` smoke path works before long training.

## DGL and Device

Check:

- Graph is a DGL graph or documented exception.
- Node/edge types match model assumptions.
- Features are on the same device as graph/model.
- Code does not hard-code `cuda:0`.

## Shape and Metric

Check:

- Logits shape vs labels.
- Mask shape and dtype.
- Regression labels are floating point.
- Recommendation/link prediction ranking protocol matches evaluator.
