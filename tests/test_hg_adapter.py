import hashlib
import json

import pytest
import torch

from openhgnn.dataset.hg_adapter_dataset import HGAdapterArtifactError, HGAdapterArtifacts
from openhgnn.models.HGAdapter import HGAdapter


def test_hg_adapter_forward_and_losses_cpu():
    torch.manual_seed(0)
    model = HGAdapter(embedding_dim=4, num_classes=3, bottleneck_dim=2, hetero_bottleneck_dim=2)
    features = torch.rand(6, 5)
    inputs = {
        'hom_before': torch.rand(6, 4), 'hom_after': torch.rand(6, 4),
        # The author artifacts store relation channels as [channels, nodes, dim].
        'het_before': [torch.rand(2, 6, 4), torch.rand(1, 3, 4)], 'het_after': torch.rand(6, 4),
    }
    logits, adjacency, reconstruction, hom_emb, het_emb = model(features, **inputs)
    loss, terms = model.losses(logits, adjacency, reconstruction, hom_emb, het_emb,
                               torch.tensor([0, 1, 2, 0, 1, 2]), torch.tensor([0, 1, 2]),
                               0.01, 0.1, 1.0)
    assert logits.shape == (6, 3)
    assert adjacency.shape == (6, 6)
    assert set(terms) == {'classification', 'reconstruction', 'margin', 'smoothness'}
    loss.backward()


def test_artifact_manifest_rejects_missing_or_bad_checksum(tmp_path):
    directory = tmp_path / 'artifacts' / 'ACM'
    directory.mkdir(parents=True)
    tensor = torch.zeros(2, 4)
    files = {}
    for key in HGAdapterArtifacts.REQUIRED:
        path = directory / f'{key}.pt'
        torch.save(tensor, path)
        files[key] = {'name': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    manifest = {'dataset': 'ACM', 'node_order': 'source_flat_target_order_v1',
                'target_node_count': 2, 'files': files}
    (directory / 'manifest.json').write_text(json.dumps(manifest))
    loaded = HGAdapterArtifacts.load(tmp_path, 'ACM', 2, 4, 'cpu')
    assert loaded['hom_before'].shape == (2, 4)
    files['hom_after']['sha256'] = 'not-a-checksum'
    (directory / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(HGAdapterArtifactError, match='Checksum'):
        HGAdapterArtifacts.load(tmp_path, 'ACM', 2, 4, 'cpu')
