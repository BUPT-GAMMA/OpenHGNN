#!/usr/bin/env python3
"""Prepare non-versioned author HG-Adapter assets for the OpenHGNN port.

The script reads the author's ``dataset.zip`` and ``pre_trained_embedding.7z``
without copying either archive into the repository.  It extracts the original
data layout and creates checksummed manifests consumed by HGAdapterArtifacts.
"""
import argparse
import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

import torch


TENSOR_NAMES = {
    'hom_before': 'x_emb_{dataset}.pt',
    'hom_after': 'emb_hom_{dataset}.pt',
    'het_before': 'vec_list_{dataset}.pt',
    'het_after': 'embs_het_{dataset}.pt',
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True,
                        help='Author HG-Adapter checkout containing dataset/ and pre_trained_embedding/.')
    parser.add_argument('--output', type=Path, default=Path('openhgnn/dataset/HGAdapter'))
    args = parser.parse_args()
    dataset_archive = args.source / 'dataset' / 'dataset.zip'
    embedding_archive = args.source / 'pre_trained_embedding' / 'pre_trained_embedding.7z'
    if not dataset_archive.exists() or not embedding_archive.exists():
        raise FileNotFoundError('Expected dataset/dataset.zip and pre_trained_embedding/pre_trained_embedding.7z.')

    raw_root = args.output / 'raw'
    raw_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dataset_archive) as archive:
        archive.extractall(raw_root)

    try:
        import py7zr
    except ImportError as error:
        raise RuntimeError('Install py7zr to extract the author embedding archive.') from error

    with tempfile.TemporaryDirectory() as temporary:
        temporary = Path(temporary)
        with py7zr.SevenZipFile(embedding_archive, mode='r') as archive:
            archive.extract(path=temporary)
        for dataset in ('ACM', 'DBLP', 'Yelp'):
            target = args.output / 'artifacts' / dataset
            target.mkdir(parents=True, exist_ok=True)
            files = {}
            for key, template in TENSOR_NAMES.items():
                source = temporary / template.format(dataset=dataset)
                destination = target / f'{key}.pt'
                shutil.copy2(source, destination)
                files[key] = {'name': destination.name, 'sha256': sha256(destination)}
            hom_before = torch.load(target / files['hom_before']['name'], map_location='cpu')
            if not isinstance(hom_before, torch.Tensor) or hom_before.ndim != 2:
                raise ValueError(f'Invalid hom_before tensor for {dataset}.')
            manifest = {
                'dataset': dataset,
                'node_order': 'source_flat_target_order_v1',
                'target_node_count': hom_before.shape[0],
                'embedding_dim': hom_before.shape[1],
                'files': files,
            }
            (target / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
            print(f'Prepared {dataset}: {hom_before.shape[0]} target nodes, dim {hom_before.shape[1]}')


if __name__ == '__main__':
    main()
