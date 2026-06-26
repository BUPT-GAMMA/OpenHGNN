"""
HGSketch Trainer
================
Training flow for HGSketch model.

Since HGSketch is a non-parametric method (no gradient-based training),
the trainer computes graph-level embeddings for all graphs in the dataset,
applies linearization, and uses a linear classifier (Logistic Regression)
for graph classification.
"""

import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

from . import register_flow
from .base_flow import BaseFlow
from ..models import build_model


@register_flow('HGSketch_trainer')
class HGSketchTrainer(BaseFlow):
    """Trainer flow for HGSketch graph classification."""

    def __init__(self, args):
        # HGSketch is non-parametric, skip standard BaseFlow graph loading
        self.args = args
        self.logger = args.logger
        self.model_name = args.model_name
        self.device = args.device

        # Build task to get dataset
        from ..tasks import build_task
        self.task = build_task(args)

    def train(self):
        """
        Main training pipeline:
        1. Build model from the first graph in dataset
        2. Compute HGSketch embeddings for all graphs
        3. Linearize embeddings
        4. Train LogisticRegression and evaluate
        """
        dataset = self.task.dataset
        graphs, labels = self._load_graph_dataset(dataset)

        if len(graphs) == 0:
            self.logger.train_info("No graphs found in dataset.")
            return {'accuracy': 0.0}

        # Build model using the first graph as reference
        model_cls = build_model(self.args.model_name)
        model = model_cls.build_model_from_args(self.args, graphs[0])

        self.logger.train_info(f"Computing HGSketch embeddings for {len(graphs)} graphs...")
        self.logger.train_info(f"Parameters: K={model.K}, R={model.R}, D={model.D}")

        # Compute embeddings for all graphs
        embeddings = []
        for i, g in enumerate(tqdm(graphs, desc="HGSketch")):
            x_g = model.compute_sketch(g)
            x_lin = model.linearize(x_g)
            embeddings.append(x_lin)

        # Pad embeddings to the same length (graphs may have different sizes)
        max_len = max(len(e) for e in embeddings)
        X = np.zeros((len(embeddings), max_len), dtype=np.float32)
        for i, e in enumerate(embeddings):
            X[i, :len(e)] = e

        y = np.array(labels)

        # Split into train/test
        train_mask, test_mask = self._get_split(dataset, len(graphs))

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        self.logger.train_info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
        self.logger.train_info(f"Feature dimension: {max_len}")

        # Train linear classifier
        clf = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='auto', C=1.0)
        clf.fit(X_train, y_train)

        # Evaluate
        y_pred_train = clf.predict(X_train)
        y_pred_test = clf.predict(X_test)

        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)
        test_f1_macro = f1_score(y_test, y_pred_test, average='macro')
        test_f1_micro = f1_score(y_test, y_pred_test, average='micro')

        self.logger.train_info(f"Train Accuracy: {train_acc:.4f}")
        self.logger.train_info(f"Test Accuracy: {test_acc:.4f}")
        self.logger.train_info(f"Test F1-Macro: {test_f1_macro:.4f}")
        self.logger.train_info(f"Test F1-Micro: {test_f1_micro:.4f}")

        return {
            'train_acc': train_acc,
            'test_acc': test_acc,
            'test_f1_macro': test_f1_macro,
            'test_f1_micro': test_f1_micro,
        }

    def _load_graph_dataset(self, dataset):
        """
        Load graphs and labels from the dataset.

        Returns
        -------
        graphs : list of DGLHeteroGraph
        labels : list of int
        """
        graphs = []
        labels = []

        if hasattr(dataset, 'graphs') and hasattr(dataset, 'labels'):
            graphs = dataset.graphs
            labels = dataset.labels
            if hasattr(labels, 'numpy'):
                labels = labels.numpy().tolist()
            elif isinstance(labels, np.ndarray):
                labels = labels.tolist()
        elif hasattr(dataset, '__len__') and hasattr(dataset, '__getitem__'):
            for i in range(len(dataset)):
                item = dataset[i]
                if isinstance(item, tuple) and len(item) == 2:
                    g, l = item
                    graphs.append(g)
                    labels.append(l.item() if hasattr(l, 'item') else int(l))

        return graphs, labels

    def _get_split(self, dataset, n):
        """
        Get train/test split masks.

        Returns
        -------
        train_mask : np.ndarray of bool
        test_mask : np.ndarray of bool
        """
        if hasattr(dataset, 'train_mask') and hasattr(dataset, 'test_mask'):
            train_mask = np.array(dataset.train_mask, dtype=bool)
            test_mask = np.array(dataset.test_mask, dtype=bool)
        elif hasattr(dataset, 'train_idx') and hasattr(dataset, 'test_idx'):
            train_mask = np.zeros(n, dtype=bool)
            test_mask = np.zeros(n, dtype=bool)
            train_mask[dataset.train_idx] = True
            test_mask[dataset.test_idx] = True
        else:
            # Default 80/20 split
            rng = np.random.RandomState(self.args.seed)
            indices = rng.permutation(n)
            split = int(0.8 * n)
            train_mask = np.zeros(n, dtype=bool)
            test_mask = np.zeros(n, dtype=bool)
            train_mask[indices[:split]] = True
            test_mask[indices[split:]] = True

        return train_mask, test_mask
