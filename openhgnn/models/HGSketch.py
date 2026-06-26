"""
HGSketch Model
==============
HGSketch maps heterogeneous graphs to low-dimensional Hamming space by extracting
simplicial complexes to capture higher-order structures and using Locality-Sensitive
Hashing (LSH) for ultra-fast dimensionality reduction.

Steps:
1. Extract k-simplices and build Hodge Laplacian matrices L_k
2. Initialize heterogeneous features via one-hot encoding of node types
3. Build local information amplification operator M^(k) = L_k ⊙ L_k
4. Build global structure enhancement operator N^(k) = (M^(k))^2
5. Iterated LSH: UPDATE -> TRANSFORM -> sgn binarization
6. Graph-level feature concatenation
7. Linearization for downstream linear classifiers
"""

import torch
import numpy as np
import networkx as nx
from itertools import combinations
from scipy import sparse as sp

from . import BaseModel, register_model


@register_model('HGSketch')
class HGSketch(BaseModel):
    r"""
    HGSketch model for heterogeneous graph-level representation.

    Parameters
    ----------
    K : int
        Maximum simplex dimension.
    R : int
        Number of LSH iterations.
    D : int
        Hash dimension (output dimension per iteration).
    num_node_types : int
        Number of node types in the heterogeneous graph.
    seed : int
        Random seed for reproducibility.
    """

    @classmethod
    def build_model_from_args(cls, args, hg):
        return cls(
            K=args.K,
            R=args.R,
            D=args.D,
            num_node_types=len(hg.ntypes),
            seed=args.seed,
        )

    def __init__(self, K=2, R=3, D=128, num_node_types=1, seed=0):
        super(HGSketch, self).__init__()
        self.K = K
        self.R = R
        self.D = D
        self.num_node_types = num_node_types
        self.seed = seed
        # Dummy parameter so PyTorch recognizes this as a module
        self._dummy = torch.nn.Parameter(torch.empty(0), requires_grad=False)

    def forward(self, hg):
        """
        Generate graph-level binary hash code for a single heterogeneous graph.

        Parameters
        ----------
        hg : dgl.DGLHeteroGraph
            A heterogeneous graph.

        Returns
        -------
        x_g : np.ndarray
            Graph-level binary feature vector.
        """
        return self.compute_sketch(hg)

    @torch.no_grad()
    def compute_sketch(self, hg):
        """Core HGSketch pipeline for a single graph."""
        rng = np.random.RandomState(self.seed)

        # Convert to undirected NetworkX graph (homogeneous view)
        nx_g = self._hg_to_nx(hg)
        num_nodes = nx_g.number_of_nodes()

        if num_nodes == 0:
            return np.zeros(0, dtype=np.float32)

        # Build node-type mapping: node_id -> type_index
        node_type_map = self._build_node_type_map(hg)

        # Step 1: Extract simplices of dimension 0..K
        simplices_by_dim = self._extract_simplices(nx_g, self.K)

        # Steps 1-6: For each dimension k, compute features
        all_features = []
        for k in range(self.K + 1):
            simplices_k = simplices_by_dim.get(k, [])
            if len(simplices_k) == 0:
                all_features.append(np.array([], dtype=np.float32))
                continue

            # Step 1: Build Hodge Laplacian L_k
            L_k = self._build_hodge_laplacian(simplices_by_dim, k)

            # Step 2: Initialize heterogeneous features
            H_in = self._init_hetero_features(simplices_k, node_type_map)

            # Step 3: Local information amplification M^(k) = L_k ⊙ L_k
            M_k = L_k.multiply(L_k)  # Hadamard product

            # Step 4: Global structure enhancement N^(k) = (M^(k))^2
            N_k = M_k.dot(M_k)

            # Step 5: Iterated LSH
            for r in range(self.R):
                # UPDATE: feature propagation with N^(k)
                H_temp = self._update(H_in, N_k)
                # TRANSFORM: random projection
                W = rng.randn(H_temp.shape[1], self.D)
                H_in = H_temp @ W
                # Binarize with sign function
                H_in = np.sign(H_in)
                # Replace 0s with -1 (edge case when value is exactly 0)
                H_in[H_in == 0] = 1.0

            all_features.append(H_in.flatten())

        # Step 6: Concatenate all dimensions
        x_g = np.concatenate([f for f in all_features if f.size > 0])

        return x_g

    def linearize(self, x_g):
        """
        Step 7: Linearize binary features for linear classifiers.
        Maps binary vector of length L to sparse vector of length 2L.

        Parameters
        ----------
        x_g : np.ndarray
            Binary feature vector with values in {-1, 1}.

        Returns
        -------
        x_lin : np.ndarray
            Linearized feature vector of length 2L.
        """
        L = len(x_g)
        if L == 0:
            return np.zeros(0, dtype=np.float32)
        # Indicator for +1 and -1
        pos = (x_g == 1).astype(np.float64)
        neg = (x_g == -1).astype(np.float64)
        x_lin = np.concatenate([pos, neg]) / np.sqrt(L)
        return x_lin

    # ==================== Helper Methods ====================

    def _hg_to_nx(self, hg):
        """Convert DGL heterogeneous graph to undirected NetworkX graph."""
        nx_g = nx.Graph()
        # Add all nodes
        for ntype in hg.ntypes:
            num = hg.num_nodes(ntype)
            # Use global node IDs
            start = self._get_node_offset(hg, ntype)
            for i in range(num):
                nx_g.add_node(start + i)

        # Add all edges (undirected)
        for etype in hg.canonical_etypes:
            src_type, _, dst_type = etype
            src, dst = hg.edges(etype=etype)
            src_offset = self._get_node_offset(hg, src_type)
            dst_offset = self._get_node_offset(hg, dst_type)
            for s, d in zip(src.numpy(), dst.numpy()):
                u = src_offset + s
                v = dst_offset + d
                if u != v:
                    nx_g.add_edge(u, v)
        return nx_g

    def _get_node_offset(self, hg, ntype):
        """Get the global node ID offset for a given node type."""
        offset = 0
        for nt in hg.ntypes:
            if nt == ntype:
                return offset
            offset += hg.num_nodes(nt)
        return offset

    def _build_node_type_map(self, hg):
        """Build a mapping from global node ID to node type index."""
        node_type_map = {}
        type_idx = 0
        offset = 0
        for ntype in hg.ntypes:
            num = hg.num_nodes(ntype)
            for i in range(num):
                node_type_map[offset + i] = type_idx
            type_idx += 1
            offset += num
        return node_type_map

    def _extract_simplices(self, nx_g, K):
        """
        Extract k-simplices (k=0..K) from the graph.
        A k-simplex is a (k+1)-clique.

        Returns
        -------
        simplices_by_dim : dict
            {k: list of tuples}, each tuple is a sorted k-simplex.
        """
        simplices_by_dim = {k: [] for k in range(K + 1)}

        # 0-simplices are just nodes
        for node in nx_g.nodes():
            simplices_by_dim[0].append((node,))

        if K >= 1:
            # Find all cliques up to size K+1
            all_cliques = list(nx.enumerate_all_cliques(nx_g))
            for clique in all_cliques:
                dim = len(clique) - 1  # k-simplex has k+1 nodes
                if dim > K:
                    break
                if dim >= 1:
                    simplices_by_dim[dim].append(tuple(sorted(clique)))

        return simplices_by_dim

    def _build_boundary_matrix(self, simplices_k, simplices_k_minus_1):
        """
        Build boundary matrix B_k mapping k-simplices to (k-1)-simplices.

        B_k has shape (num_(k-1)-simplices, num_k-simplices).
        """
        if len(simplices_k) == 0 or len(simplices_k_minus_1) == 0:
            return sp.csr_matrix((len(simplices_k_minus_1), len(simplices_k)))

        # Index lookup for (k-1)-simplices
        face_to_idx = {s: i for i, s in enumerate(simplices_k_minus_1)}

        rows, cols, vals = [], [], []
        for j, simplex in enumerate(simplices_k):
            # Each k-simplex has (k+1) faces of dimension (k-1)
            for i_face, _ in enumerate(simplex):
                face = tuple(simplex[:i_face] + simplex[i_face + 1:])
                if face in face_to_idx:
                    rows.append(face_to_idx[face])
                    cols.append(j)
                    vals.append((-1) ** i_face)

        B_k = sp.csr_matrix(
            (vals, (rows, cols)),
            shape=(len(simplices_k_minus_1), len(simplices_k))
        )
        return B_k

    def _build_hodge_laplacian(self, simplices_by_dim, k):
        """
        Build the Hodge Laplacian L_k.

        L_0 = B_1 @ B_1^T
        L_k = B_k^T @ B_k + B_{k+1} @ B_{k+1}^T  (for 1 <= k < K)
        L_K = B_K^T @ B_K
        """
        n = len(simplices_by_dim.get(k, []))
        if n == 0:
            return sp.csr_matrix((0, 0))

        K = self.K

        if k == 0:
            # L_0 = B_1 @ B_1^T
            simplices_1 = simplices_by_dim.get(1, [])
            if len(simplices_1) > 0:
                B_1 = self._build_boundary_matrix(simplices_1, simplices_by_dim[0])
                L_k = B_1.dot(B_1.T)
            else:
                L_k = sp.csr_matrix((n, n))
        elif k == K:
            # L_K = B_K^T @ B_K
            B_k = self._build_boundary_matrix(simplices_by_dim[k], simplices_by_dim.get(k - 1, []))
            L_k = B_k.T.dot(B_k)
        else:
            # L_k = B_k^T @ B_k + B_{k+1} @ B_{k+1}^T
            B_k = self._build_boundary_matrix(simplices_by_dim[k], simplices_by_dim.get(k - 1, []))
            down = B_k.T.dot(B_k)

            simplices_k_plus_1 = simplices_by_dim.get(k + 1, [])
            if len(simplices_k_plus_1) > 0:
                B_k_plus_1 = self._build_boundary_matrix(simplices_k_plus_1, simplices_by_dim[k])
                up = B_k_plus_1.dot(B_k_plus_1.T)
            else:
                up = sp.csr_matrix((n, n))

            L_k = down + up

        return L_k.tocsr().astype(np.float64)

    def _init_hetero_features(self, simplices_k, node_type_map):
        """
        Initialize features for k-simplices using one-hot encoding of node types.

        For each k-simplex, aggregate the one-hot vectors of its constituent nodes.

        Returns
        -------
        H_in : np.ndarray
            Shape (num_simplices, num_node_types).
        """
        num_types = self.num_node_types
        n = len(simplices_k)
        H_in = np.zeros((n, num_types), dtype=np.float64)

        for i, simplex in enumerate(simplices_k):
            for node in simplex:
                t = node_type_map.get(node, 0)
                H_in[i, t] = 1.0  # one-hot aggregation (union)

        return H_in

    def _update(self, H_in, N_k):
        """
        UPDATE step: propagate features using global operator N^(k).

        H_temp = N^(k) @ H_in + H_in  (with residual connection)
        """
        if sp.issparse(N_k):
            H_temp = N_k.dot(H_in) + H_in
        else:
            H_temp = N_k @ H_in + H_in
        return H_temp
