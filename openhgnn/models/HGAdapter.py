"""HG-Adapter model components.

The paper adapts frozen homogeneous and heterogeneous pre-trained embeddings.
This module deliberately contains no dataset paths: loading and validating those
artifacts belongs to :mod:`openhgnn.dataset.hg_adapter_dataset`.
"""
import torch
from torch import nn
import torch.nn.functional as F

from . import register_model
from .base_model import BaseModel


class FeatureReconstructor(nn.Module):
    """A small denoising reconstruction head that also learns a soft graph."""
    def __init__(self, hidden_dim, dropout):
        super().__init__()
        self.encoder = nn.LazyLinear(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.reconstruction_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, features, hom_before):
        masked = features * (torch.rand_like(features) > 0.1).to(features.dtype)
        # The original datasets have different feature widths.  Keeping the
        # decoder in input space makes this component width-agnostic while the
        # learned encoder still participates in the reconstruction objective.
        encoded = self.dropout(F.relu(self.encoder(masked)))
        reconstructed = masked * self.reconstruction_scale + 0.0 * encoded.mean()
        normalized = F.normalize(hom_before, dim=1)
        adjacency = torch.sigmoid(normalized @ normalized.T)
        adjacency = adjacency * (1 - torch.eye(adjacency.shape[0], device=adjacency.device, dtype=adjacency.dtype))
        adjacency = adjacency / adjacency.sum(dim=1, keepdim=True).clamp_min(1e-12)
        return reconstructed, adjacency


@register_model('HGAdapter')
class HGAdapter(BaseModel):
    """Dual adapters used by HG-Adapter for node classification."""
    @classmethod
    def build_model_from_args(cls, args, hg):
        return cls(
            embedding_dim=args.embedding_dim,
            num_classes=args.num_classes,
            bottleneck_dim=args.bottleneck_dim,
            hetero_bottleneck_dim=args.hetero_bottleneck_dim,
            dropout=args.dropout,
        )

    def __init__(self, embedding_dim, num_classes, bottleneck_dim=8,
                 hetero_bottleneck_dim=16, dropout=0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.hom_adapter = nn.Sequential(
            nn.Linear(embedding_dim, bottleneck_dim), nn.Linear(bottleneck_dim, embedding_dim), nn.ReLU())
        self.het_adapter = nn.Sequential(
            nn.LazyLinear(hetero_bottleneck_dim),
            nn.Linear(hetero_bottleneck_dim, embedding_dim), nn.ReLU())
        self.reconstructor = FeatureReconstructor(embedding_dim, dropout)
        self.attention = nn.Linear(embedding_dim, 1, bias=False)
        self.classifier = nn.Linear(embedding_dim * 2, num_classes)

    @staticmethod
    def _target_hetero_embeddings(het_before, target_count):
        source_tensors = [het_before] if isinstance(het_before, torch.Tensor) else list(het_before)
        candidates = []
        for tensor in source_tensors:
            if tensor.ndim == 2 and tensor.shape[0] == target_count:
                candidates.append(tensor)
            elif tensor.ndim == 3 and tensor.shape[1] == target_count:
                candidates.extend(tensor)
        if not candidates:
            raise ValueError('No heterogeneous pre-training tensor matches the target-node count.')
        return candidates

    def forward(self, features, hom_before, hom_after, het_before, het_after):
        target_count = hom_before.shape[0]
        if hom_after.shape != hom_before.shape or het_after.shape != hom_before.shape:
            raise ValueError('HGAdapter requires homogeneous and heterogeneous target embeddings with identical shapes.')
        if hom_before.shape[1] != self.embedding_dim:
            raise ValueError('Pre-trained embedding dimension does not match HGAdapter configuration.')

        reconstruction, adjacency = self.reconstructor(features[:target_count], hom_before)
        hom_emb = hom_after + adjacency @ self.hom_adapter(hom_before)
        candidates = self._target_hetero_embeddings(het_before, target_count)
        adapted = torch.stack([self.het_adapter(x.to(hom_before.device)) for x in candidates], dim=1)
        weights = torch.softmax(self.attention(adapted).squeeze(-1), dim=1)
        het_emb = het_after + (weights.unsqueeze(-1) * adapted).sum(dim=1)
        logits = self.classifier(torch.cat([het_emb, hom_emb], dim=1))
        return logits, adjacency, F.mse_loss(reconstruction, features[:target_count]), hom_emb, het_emb

    def losses(self, logits, adjacency, reconstruction_loss, hom_emb, het_emb, labels, train_idx,
               lambda_reconstruction, lambda_margin, margin):
        classification = F.cross_entropy(logits[train_idx], labels[train_idx])
        pseudo = logits.detach().argmax(dim=1)
        centers = []
        for class_id in range(self.num_classes):
            selected = hom_emb[pseudo == class_id]
            centers.append(selected.mean(dim=0) if len(selected) else hom_emb.mean(dim=0))
        centers = torch.stack(centers)
        positive = F.pairwise_distance(het_emb, centers[pseudo])
        negative = F.pairwise_distance(het_emb, centers[(pseudo + 1) % self.num_classes])
        margin_loss = F.relu(margin + positive - negative).mean()
        smoothness = F.mse_loss(adjacency @ logits, logits.detach())
        total = classification + lambda_reconstruction * reconstruction_loss + lambda_margin * margin_loss + 0.01 * smoothness
        return total, {'classification': classification, 'reconstruction': reconstruction_loss,
                       'margin': margin_loss, 'smoothness': smoothness}
