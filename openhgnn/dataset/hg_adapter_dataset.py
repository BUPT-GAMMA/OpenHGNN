"""Original HG-Adapter data and pre-training artifact adapters."""
import hashlib
import json
import os
import pickle
from pathlib import Path

import dgl
import numpy as np
import torch

from . import BaseDataset, register_dataset


DATASETS = {'acm4HGAdapter': 'ACM', 'dblp4HGAdapter': 'DBLP',
            'yelp4HGAdapter': 'Yelp', 'aminer4HGAdapter': 'Aminer'}


class HGAdapterArtifactError(RuntimeError):
    pass


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


class HGAdapterArtifacts:
    """Loads an explicitly versioned set of the four frozen embedding tensors."""
    REQUIRED = ('hom_before', 'hom_after', 'het_before', 'het_after')

    @classmethod
    def load(cls, root, dataset, target_count, embedding_dim, device):
        directory = Path(root) / 'artifacts' / dataset
        manifest_path = directory / 'manifest.json'
        if not manifest_path.exists():
            raise HGAdapterArtifactError(
                f'Missing {manifest_path}. Provide HG-Adapter pre-training artifacts and their manifest.')
        manifest = json.loads(manifest_path.read_text())
        if manifest.get('dataset') != dataset or manifest.get('node_order') != 'source_flat_target_order_v1':
            raise HGAdapterArtifactError('Artifact manifest dataset or node-order contract does not match.')
        if manifest.get('target_node_count') != target_count:
            raise HGAdapterArtifactError('Artifact manifest target-node count does not match the loaded dataset.')
        tensors = {}
        for key in cls.REQUIRED:
            item = manifest.get('files', {}).get(key)
            if not item:
                raise HGAdapterArtifactError(f'Artifact manifest is missing {key}.')
            path = directory / item['name']
            if not path.exists() or _sha256(path) != item.get('sha256'):
                raise HGAdapterArtifactError(f'Checksum validation failed for {key}: {path}')
            # Do not use ``weights_only`` here: OpenHGNN still supports PyTorch
            # versions predating that keyword and ``het_before`` is a tensor list.
            tensors[key] = torch.load(path, map_location=device)
        for key in ('hom_before', 'hom_after', 'het_after'):
            if not isinstance(tensors[key], torch.Tensor) or tensors[key].shape != (target_count, embedding_dim):
                raise HGAdapterArtifactError(f'{key} has invalid shape; expected {(target_count, embedding_dim)}.')
        candidates = tensors['het_before'] if isinstance(tensors['het_before'], (list, tuple)) else [tensors['het_before']]
        valid = any(isinstance(item, torch.Tensor) and (
            (item.ndim == 2 and item.shape[0] == target_count) or
            (item.ndim == 3 and item.shape[1] == target_count)
        ) for item in candidates)
        if not candidates or not valid:
            raise HGAdapterArtifactError('het_before has no target-node embedding with the required shape.')
        return tensors


@register_dataset('hg_adapter_node_classification')
class HGAdapterNodeClassification(BaseDataset):
    """Converts the author's pickle/numpy package without changing its split."""
    def __init__(self, dataset_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if dataset_name not in DATASETS:
            raise ValueError(f'Unsupported HG-Adapter dataset: {dataset_name}')
        self.dataset_name = dataset_name
        self.source_name = DATASETS[dataset_name]
        args = kwargs['args']
        root = Path(getattr(args, 'hg_adapter_root', Path(args.openhgnn_dir) / 'dataset' / 'HGAdapter'))
        raw = root / 'raw' / self.source_name
        if not raw.exists():
            raise FileNotFoundError(
                f'Missing original HG-Adapter data at {raw}. Extract the author dataset package under {root / "raw"}.')
        self.g, self.category, self.num_classes = self._load(raw)
        self.in_dim = self.g.nodes[self.category].data['h'].shape[1]
        self.has_feature = True
        self.meta_paths_dict = None
        self.multi_label = False

    @staticmethod
    def _edge_pairs(matrix, source_offset, target_offset):
        rows, cols = matrix.nonzero()
        return torch.as_tensor(rows - source_offset), torch.as_tensor(cols - target_offset)

    def _load(self, raw):
        if self.source_name == 'Aminer':
            return self._load_aminer(raw / 'raw')
        with open(raw / 'node_features.pkl', 'rb') as stream:
            features = torch.as_tensor(pickle.load(stream), dtype=torch.float32)
        with open(raw / 'edges.pkl', 'rb') as stream:
            edges = pickle.load(stream)
        with open(raw / 'labels.pkl', 'rb') as stream:
            splits = pickle.load(stream)
        if self.source_name == 'ACM':
            types, counts, names, rels, category = ('p', 'a', 's'), (3025, 5912, 57), ('p-a', 'a-p', 'p-s', 's-p'), edges, 'p'
        elif self.source_name == 'DBLP':
            types, counts, names, rels, category = ('a', 'p', 'c'), (4057, 14328, 20), ('p-a', 'a-p', 'p-c', 'c-p'), edges, 'a'
        else:
            with open(raw / 'meta_data.pkl', 'rb') as stream:
                meta = pickle.load(stream)
            types = tuple(meta['t_info'].keys())
            counts = tuple(len([i for key, i in meta['node2gid'].items() if key.startswith(t)]) for t in types)
            category = 'b'
            names, rels = tuple(edges.keys()), edges
        offsets = np.cumsum((0,) + counts)
        data = {t: features[offsets[i]:offsets[i + 1]] for i, t in enumerate(types)}
        graph_edges = {}
        offset_by_type = dict(zip(types, offsets[:-1]))
        for name, matrix in zip(names, rels.values() if isinstance(rels, dict) else rels):
            src, dst = name.split('-')
            graph_edges[(src, name, dst)] = self._edge_pairs(matrix, offset_by_type[src], offset_by_type[dst])
        graph = dgl.heterograph(graph_edges, num_nodes_dict=dict(zip(types, counts)))
        for ntype, feat in data.items():
            graph.nodes[ntype].data['h'] = feat
        labels = torch.zeros(counts[types.index(category)], dtype=torch.long)
        masks = [torch.zeros_like(labels, dtype=torch.bool) for _ in range(3)]
        for mask, split in zip(masks, splits):
            split = np.asarray(split)
            index = torch.as_tensor(split[:, 0], dtype=torch.long)
            labels[index] = torch.as_tensor(split[:, 1], dtype=torch.long)
            mask[index] = True
        graph.nodes[category].data.update({'label': labels, 'train_mask': masks[0], 'val_mask': masks[1], 'test_mask': masks[2]})
        return graph, category, int(labels.max().item() + 1)

    def _load_aminer(self, raw):
        arrays = [torch.as_tensor(np.load(raw / f'features_{i}.npy'), dtype=torch.float32) for i in range(3)]
        types, category = ('p', 'a', 'r'), 'p'
        counts = tuple(len(item) for item in arrays)
        pa, pr = np.loadtxt(raw / 'pa.txt', dtype=np.int64), np.loadtxt(raw / 'pr.txt', dtype=np.int64)
        graph = dgl.heterograph({
            ('p', 'p-a', 'a'): (torch.as_tensor(pa[:, 0]), torch.as_tensor(pa[:, 1] - counts[0])),
            ('a', 'a-p', 'p'): (torch.as_tensor(pa[:, 1] - counts[0]), torch.as_tensor(pa[:, 0])),
            ('p', 'p-r', 'r'): (torch.as_tensor(pr[:, 0]), torch.as_tensor(pr[:, 1] - sum(counts[:2]))),
            ('r', 'r-p', 'p'): (torch.as_tensor(pr[:, 1] - sum(counts[:2])), torch.as_tensor(pr[:, 0])),
        }, num_nodes_dict=dict(zip(types, counts)))
        for ntype, feat in zip(types, arrays):
            graph.nodes[ntype].data['h'] = feat
        labels = torch.as_tensor(np.load(raw / 'labels.npy'), dtype=torch.long)
        masks = []
        for part in ('train_20.npy', 'val_20.npy', 'test_20.npy'):
            mask = torch.zeros(len(labels), dtype=torch.bool)
            mask[torch.as_tensor(np.load(raw / part), dtype=torch.long)] = True
            masks.append(mask)
        graph.nodes[category].data.update({'label': labels, 'train_mask': masks[0], 'val_mask': masks[1], 'test_mask': masks[2]})
        return graph, category, int(labels.max().item() + 1)

    def get_split(self, *args):
        data = self.g.nodes[self.category].data
        return tuple(torch.where(data[key])[0] for key in ('train_mask', 'val_mask', 'test_mask'))

    def get_labels(self):
        return self.g.nodes[self.category].data['label']
