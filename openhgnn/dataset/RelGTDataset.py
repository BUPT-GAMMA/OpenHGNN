"""
RelGT dataset utilities.

Ported from relgt/utils.py with no algorithmic changes.
Provides neighbour sampling and the RelGTTokens Dataset class
that is consumed by RelGTTrainer.
"""
import gc
import os
import random
import time
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Optional, Tuple

import numpy as np
import h5py
import torch
from torch import Tensor
from torch.utils.data import Dataset

try:
    from torch_geometric.data import HeteroData
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
except Exception:
    SentenceTransformer = None
    HAS_ST = False

try:
    from relbench.base import Dataset as RelBenchDataset, EntityTask
    from relbench.modeling.graph import get_node_train_table_input
    HAS_RELBENCH = True
except ImportError:
    HAS_RELBENCH = False


# ---------------------------------------------------------------------------
# Module-level globals for multiprocessing workers
# ---------------------------------------------------------------------------
GLOBAL_ADJ = None
GLOBAL_ALL_NODES = None


# ---------------------------------------------------------------------------
# Text embedding
# ---------------------------------------------------------------------------

class GloveTextEmbedding:
    """Wraps sentence-transformers GloVe model for text column encoding."""

    def __init__(self, device: torch.device):
        if not HAS_ST:
            raise ImportError(
                "sentence-transformers is required for GloveTextEmbedding. "
                "Install with: pip install sentence-transformers"
            )
        self.model = SentenceTransformer(
            "sentence-transformers/average_word_embeddings_glove.6B.300d",
            device=device,
        )

    def __call__(self, sentences: List[str]) -> Tensor:
        return torch.from_numpy(self.model.encode(sentences))


# ---------------------------------------------------------------------------
# Adjacency construction
# ---------------------------------------------------------------------------

def build_adjacency_hetero(hetero_data: "HeteroData",
                           undirected: bool = True) -> Dict:
    """
    Build a per-node-type adjacency dictionary from a HeteroData object.

    Returns
    -------
    adjacency : Dict[str, List[set]]
        adjacency[node_type][local_node_idx] = set of (neighbor_type, neighbor_idx)
    """
    adjacency = {
        node_type: [set() for _ in range(hetero_data[node_type].num_nodes)]
        for node_type in hetero_data.node_types
    }
    for edge_type in hetero_data.edge_types:
        src_type, _, dst_type = edge_type
        if "edge_index" not in hetero_data[edge_type]:
            continue
        edge_index = hetero_data[edge_type].edge_index
        src_list = edge_index[0].tolist()
        dst_list = edge_index[1].tolist()
        for s, d in zip(src_list, dst_list):
            adjacency[src_type][s].add((dst_type, d))
            if undirected:
                adjacency[dst_type][d].add((src_type, s))
    return adjacency


# ---------------------------------------------------------------------------
# Neighbour gathering helpers
# ---------------------------------------------------------------------------

def gather_1_and_2_hop_with_seed_time(
    adjacency: Dict,
    data: "HeteroData",
    node_type: str,
    node_idx: int,
    seed_time: float,
    max_1hop_threshold: int = 5000,
    max_2hop_threshold: int = 1000,
) -> List[Tuple]:
    """
    Gather 1-hop and 2-hop temporally-filtered neighbours.

    Returns
    -------
    List of tuples (nbr_type, nbr_idx, hop, relative_time_days, connecting_1hops)
    """
    # 1-hop
    n1_full = adjacency[node_type][node_idx]
    if len(n1_full) > max_1hop_threshold:
        n1_full = random.sample(list(n1_full), max_1hop_threshold)
    else:
        n1_full = list(n1_full)

    n1 = set()
    for (nbr_t, nbr_i) in n1_full:
        if hasattr(data[nbr_t], "time"):
            if data[nbr_t].time[nbr_i] <= seed_time:
                n1.add((nbr_t, nbr_i))
        else:
            n1.add((nbr_t, nbr_i))

    # 2-hop
    n2 = defaultdict(set)
    for (nbr_t, nbr_i) in n1:
        nbr2_full = adjacency[nbr_t][nbr_i]
        if len(nbr2_full) > max_2hop_threshold:
            nbr2_full = random.sample(list(nbr2_full), max_2hop_threshold)
        else:
            nbr2_full = list(nbr2_full)
        for (nbr2_t, nbr2_i) in nbr2_full:
            if (nbr2_t, nbr2_i) == (node_type, node_idx):
                continue
            if hasattr(data[nbr2_t], "time"):
                if data[nbr2_t].time[nbr2_i] <= seed_time:
                    n2[(nbr2_t, nbr2_i)].add((nbr_t, nbr_i))
            else:
                n2[(nbr2_t, nbr2_i)].add((nbr_t, nbr_i))

    n2 = {k: v for k, v in n2.items() if k not in n1}

    neighbors_with_time = []
    for (nbr_t, nbr_i) in n1:
        if hasattr(data[nbr_t], "time"):
            nbr_time = data[nbr_t].time[nbr_i].item()
            relative_time_days = (seed_time - nbr_time) / (60 * 60 * 24)
        else:
            relative_time_days = 0.0
        neighbors_with_time.append((nbr_t, nbr_i, 1, relative_time_days, None))

    for (nbr2_t, nbr2_i), connecting_1hops in n2.items():
        if hasattr(data[nbr2_t], "time"):
            nbr2_time = data[nbr2_t].time[nbr2_i].item()
            relative_time_days = (seed_time - nbr2_time) / (60 * 60 * 24)
        else:
            relative_time_days = 0.0
        neighbors_with_time.append(
            (nbr2_t, nbr2_i, 2, relative_time_days, connecting_1hops))

    return neighbors_with_time


def init_worker_globals(adj, all_nodes):
    """Initialiser for multiprocessing workers — injects shared globals."""
    global GLOBAL_ADJ, GLOBAL_ALL_NODES
    GLOBAL_ADJ = adj
    GLOBAL_ALL_NODES = all_nodes


def _process_one_seed(args):
    """
    Worker function: gather neighbours for one seed node, sample K tokens,
    build the local subgraph edge_index.
    """
    global GLOBAL_ADJ, GLOBAL_ALL_NODES

    (data, K, seed_node_type, seed_node_idx, seed_time, seed_val) = args
    random.seed(seed_val)

    T_hat = gather_1_and_2_hop_with_seed_time(
        GLOBAL_ADJ, data, seed_node_type, seed_node_idx, seed_time)
    T_hat_list = list(T_hat)
    size_th = len(T_hat_list)
    K_minus_1 = K - 1

    one_hop = [n for n in T_hat_list if n[2] == 1]
    two_hop = [n for n in T_hat_list if n[2] == 2]
    combined = one_hop + two_hop

    if size_th >= K_minus_1:
        chosen = random.sample(combined, K_minus_1)
    elif 0 < size_th < K_minus_1:
        chosen = random.choices(combined, k=K_minus_1)
    else:
        if K_minus_1 <= len(GLOBAL_ALL_NODES):
            fallback = random.sample(GLOBAL_ALL_NODES, K_minus_1)
        else:
            fallback = random.choices(GLOBAL_ALL_NODES, k=K_minus_1)
        chosen = []
        for (ft, fi) in fallback:
            if hasattr(data[ft], "time"):
                ft_time = data[ft].time[fi].item()
                rel_time = (seed_time - ft_time) / (60 * 60 * 24)
            else:
                rel_time = 0.0
            chosen.append((ft, fi, 3, rel_time, None))

    # Build final token list: seed first, then randomised neighbours
    final_tokens = [(seed_node_type, seed_node_idx, 0, 0.0, 0)]
    final_tokens.extend(chosen)
    if len(final_tokens) > 1:
        rest = final_tokens[1:]
        random.shuffle(rest)
        final_tokens = [final_tokens[0]] + rest

    # Build local subgraph adjacency
    local_map = {(t_str, i): j
                 for j, (t_str, i, *_) in enumerate(final_tokens)}
    edges = []
    for j_src, (t_str, i, *_) in enumerate(final_tokens):
        for (nbr_t, nbr_i) in GLOBAL_ADJ[t_str][i]:
            if (nbr_t, nbr_i) in local_map:
                edges.append((j_src, local_map[(nbr_t, nbr_i)]))

    if edges:
        edge_index = np.array(edges, dtype=np.int32).T   # [2, E]
    else:
        edge_index = np.zeros((2, 0), dtype=np.int32)

    return (seed_node_type, seed_node_idx, final_tokens, edge_index)


def local_nodes_hetero(
    data: "HeteroData",
    K: int,
    table_input_nodes: tuple,
    table_input_time: torch.Tensor,
    undirected: bool = True,
    num_workers: Optional[int] = None,
) -> Dict:
    """
    Parallel K-token neighbourhood sampling for a batch of seed nodes.

    Returns
    -------
    S : Dict[str, Dict[int, Tuple[List, np.ndarray]]]
        S[seed_node_type][seed_node_idx] = (final_tokens_list, edge_index)
    """
    global GLOBAL_ADJ, GLOBAL_ALL_NODES

    if GLOBAL_ADJ is None:
        GLOBAL_ADJ = build_adjacency_hetero(data, undirected=undirected)
    if GLOBAL_ALL_NODES is None:
        GLOBAL_ALL_NODES = [
            (nt, i)
            for nt in data.node_types
            for i in range(data[nt].num_nodes)
        ]

    seed_node_type, seed_node_idxs = table_input_nodes
    assert len(seed_node_idxs) == len(table_input_time)

    tasks = []
    for i, node_idx_t in enumerate(seed_node_idxs):
        node_idx = node_idx_t.item()
        seed_t = table_input_time[i].item()
        seed_val = hash((seed_node_type, node_idx, seed_t, K)) & 0xFFFFFFFF
        tasks.append((data, K, seed_node_type, node_idx, seed_t, seed_val))

    if num_workers is None:
        num_workers = max(1, min(cpu_count() - 2, len(tasks)))

    with Pool(
        processes=num_workers,
        initializer=init_worker_globals,
        initargs=(GLOBAL_ADJ, GLOBAL_ALL_NODES),
    ) as pool:
        results = pool.map(_process_one_seed, tasks)

    S = {seed_node_type: {}}
    for (nt, idx, final_list, edge_index) in results:
        S[nt][idx] = (final_list, edge_index)
    return S


# ---------------------------------------------------------------------------
# RelGTTokens — main Dataset class
# ---------------------------------------------------------------------------

class RelGTTokens(Dataset):
    """
    PyTorch Dataset that provides pre-sampled neighbour token sequences for
    each seed node in a RelBench split.

    Sampling is done once (optionally in parallel) and cached to HDF5.
    At runtime, __getitem__ reads from HDF5 and attaches raw TorchFrame
    objects for tabular feature encoding.

    Requires: relbench, pytorch-frame, h5py, sentence-transformers
    """

    def __init__(
        self,
        data: "HeteroData",
        task: "EntityTask",
        K: int,
        split: str = "train",
        undirected: bool = True,
        num_workers: Optional[int] = None,
        precompute: bool = True,
        precomputed_dir: Optional[str] = None,
        train_stage: str = "finetune",
    ):
        super().__init__()
        if not HAS_RELBENCH:
            raise ImportError(
                "relbench is required for RelGTTokens. "
                "Install with: pip install relbench"
            )

        self.data = data
        self.task = task
        self.split = split
        self.K = K
        self.undirected = undirected
        self.num_workers = num_workers
        self.precompute = precompute
        self.precomputed_dir = precomputed_dir
        self.train_stage = train_stage

        self.table = self.task.get_table(split=self.split)
        self.table_input = get_node_train_table_input(self.table, self.task)
        self.node_type, self.node_idxs = self.table_input.nodes
        self.target = (
            self.table_input.target
            if self.table_input.target is not None
            else None
        )
        self.time = getattr(self.table_input, "time", None)

        self.node_types = self.data.node_types
        self.node_type_to_index = {nt: idx for idx, nt in enumerate(self.node_types)}
        self.index_to_node_type = {idx: nt for idx, nt in enumerate(self.node_types)}

        self.max_neighbor_hop = 2 + 1   # 2-hop + 1 fallback token type

        self._create_global_mappings()
        self.precomputed_path = self._construct_precomputed_path()

        if self.precompute:
            if os.path.exists(self.precomputed_path):
                print(f"[{self.split}] Found existing HDF5 at {self.precomputed_path}")
            else:
                print(f"[{self.split}] Precomputing neighbour sampling (K={self.K})…")
                self._precompute_sampling()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _create_global_mappings(self):
        self.type_local_to_global: Dict[Tuple[int, int], int] = {}
        self.global_to_type_local: Dict[int, Tuple[int, int]] = {}
        global_index = 0
        for type_idx, node_type in self.index_to_node_type.items():
            if "x" in self.data[node_type]:
                num_nodes = self.data[node_type]["x"].size(0)
            else:
                num_nodes = self.data[node_type].num_nodes
            for local_idx in range(num_nodes):
                key = (type_idx, local_idx)
                self.type_local_to_global[key] = global_index
                self.global_to_type_local[global_index] = key
                global_index += 1

    def get_global_index(self, type_idxs: List[int],
                         local_idxs: List[int]) -> List[int]:
        return [self.type_local_to_global[(t, l)]
                for t, l in zip(type_idxs, local_idxs)]

    def _construct_precomputed_path(self) -> str:
        if not self.precomputed_dir:
            raise ValueError("'precomputed_dir' must be provided.")
        path = os.path.join(self.precomputed_dir, str(self.K), f"{self.split}.h5")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    def _create_datasets(self, h5file: h5py.File, total_samples: int) -> dict:
        chunk = min(total_samples, 10000)
        return {
            "types":   h5file.create_dataset("types",   shape=(total_samples, self.K),
                                              dtype="int16",   chunks=(chunk, self.K)),
            "indices": h5file.create_dataset("indices", shape=(total_samples, self.K),
                                              dtype="int32",   chunks=(chunk, self.K)),
            "hops":    h5file.create_dataset("hops",    shape=(total_samples, self.K),
                                              dtype="int8",    chunks=(chunk, self.K)),
            "times":   h5file.create_dataset("times",   shape=(total_samples, self.K),
                                              dtype="float32", chunks=(chunk, self.K)),
        }

    def _precompute_sampling(self):
        """Run local_nodes_hetero in chunks and persist to HDF5."""
        from tqdm import tqdm
        total = len(self.node_idxs)
        chunk_size = 10000

        with h5py.File(self.precomputed_path, "w") as hf:
            datasets = self._create_datasets(hf, total)
            adjacency_all = [None] * total

            with tqdm(total=total, desc=f"Precomputing '{self.split}'") as pbar:
                for start_idx in range(0, total, chunk_size):
                    end_idx = min(start_idx + chunk_size, total)
                    size_chunk = end_idx - start_idx

                    chunk_node_idxs = self.node_idxs[start_idx:end_idx]
                    chunk_times = (self.time[start_idx:end_idx]
                                   if self.time is not None else None)

                    S_chunk = local_nodes_hetero(
                        data=self.data.to("cpu"),
                        K=self.K,
                        table_input_nodes=(self.node_type, chunk_node_idxs),
                        table_input_time=chunk_times,
                        undirected=self.undirected,
                        num_workers=self.num_workers,
                    )

                    c_types   = np.zeros((size_chunk, self.K), dtype=np.int16)
                    c_indices = np.zeros((size_chunk, self.K), dtype=np.int32)
                    c_hops    = np.zeros((size_chunk, self.K), dtype=np.int8)
                    c_times   = np.zeros((size_chunk, self.K), dtype=np.float32)

                    for i, node_id in enumerate(chunk_node_idxs):
                        final_nodes, edge_index = S_chunk[self.node_type][int(node_id)]
                        for j, (t_str, nbr_loc_idx, hop, t_val, _) in enumerate(
                                final_nodes):
                            c_types[i, j]   = self.node_type_to_index[t_str]
                            c_indices[i, j] = nbr_loc_idx
                            c_hops[i, j]    = hop
                            c_times[i, j]   = t_val
                        adjacency_all[start_idx + i] = edge_index

                    datasets["types"][start_idx:end_idx]   = c_types
                    datasets["indices"][start_idx:end_idx] = c_indices
                    datasets["hops"][start_idx:end_idx]    = c_hops
                    datasets["times"][start_idx:end_idx]   = c_times
                    pbar.update(size_chunk)
                    del S_chunk
                    gc.collect()

            # Store variable-length edge_index arrays compactly
            offsets = np.zeros(total + 1, dtype=np.uint64)
            for i in range(total):
                E_i = adjacency_all[i].shape[1] if adjacency_all[i] is not None else 0
                offsets[i + 1] = offsets[i] + E_i

            total_edges = int(offsets[-1])
            edges_dset = hf.create_dataset("edges",
                                           shape=(2, total_edges), dtype="int16")
            for i in range(total):
                e_arr = adjacency_all[i]
                s, e_ = offsets[i], offsets[i + 1]
                if e_arr is not None and e_arr.size > 0:
                    edges_dset[:, s:e_] = e_arr
            hf.create_dataset("edges_offsets", data=offsets)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.node_idxs)

    def __getitem__(self, idx: int) -> Tuple[dict, Optional[torch.Tensor]]:
        with h5py.File(self.precomputed_path, "r") as hf:
            sample = {
                "types":   torch.from_numpy(hf["types"][idx]).long(),
                "indices": torch.from_numpy(hf["indices"][idx]).long(),
                "hops":    torch.from_numpy(hf["hops"][idx]).long(),
                "times":   torch.from_numpy(hf["times"][idx]),
            }
            offsets    = hf["edges_offsets"]
            edges_dset = hf["edges"]
            s, e_ = offsets[idx], offsets[idx + 1]
            if s == e_:
                eidx = torch.zeros((2, 0), dtype=torch.long)
            else:
                eidx = torch.from_numpy(edges_dset[:, s:e_]).long()
            sample["edge_index"] = eidx

        label = self.target[idx] if self.target is not None else None

        sample["first_type"]  = sample["types"][0].item()
        sample["first_index"] = sample["indices"][0].item()
        sample["tfs"] = [
            self.data[self.index_to_node_type[t.item()]].tf[i.item()]
            for t, i in zip(sample["types"], sample["indices"])
        ]
        sample["global_idx"] = idx
        return sample, label

    def collate(self, batch: List[Tuple[dict, Optional[torch.Tensor]]]) -> dict:
        samples, labels = zip(*batch)

        neighbor_types   = torch.stack([s["types"]   for s in samples])  # [B, K]
        neighbor_indices = torch.stack([s["indices"] for s in samples])
        neighbor_hops    = torch.stack([s["hops"]    for s in samples])
        neighbor_times   = torch.stack([s["times"]   for s in samples])

        out = {
            "neighbor_types":   neighbor_types,
            "neighbor_indices": neighbor_indices,
            "neighbor_hops":    neighbor_hops,
            "neighbor_times":   neighbor_times,
        }

        if self.target is not None:
            out["labels"] = torch.stack(list(labels))
        else:
            out["labels"] = None

        # Global seed-node indices (for VQ centroid tracking)
        first_types   = [s["first_type"]  for s in samples]
        first_indices = [s["first_index"] for s in samples]
        out["node_indices"] = torch.tensor(
            self.get_global_index(first_types, first_indices), dtype=torch.long)

        # Group TorchFrame objects by node type for batch encoding
        B, K = neighbor_types.shape
        grouped_tfs = {}
        grouped_positions = {}
        for t_id in range(len(self.node_types)):
            mask = (neighbor_types == t_id)
            if not mask.any():
                continue
            local_idxs  = neighbor_indices[mask]
            type_str    = self.index_to_node_type[t_id]
            offsets_list = torch.nonzero(mask, as_tuple=False).tolist()
            grouped_tfs[t_id]       = self.data[type_str].tf[local_idxs]
            grouped_positions[t_id] = [b * K + k for (b, k) in offsets_list]

        flat_batch_idx = torch.arange(B).unsqueeze(1).expand(B, K).reshape(-1).tolist()
        flat_nbr_idx   = torch.arange(K).repeat(B).tolist()

        out.update({
            "grouped_tfs":     grouped_tfs,
            "grouped_indices": grouped_positions,
            "flat_batch_idx":  flat_batch_idx,
            "flat_nbr_idx":    flat_nbr_idx,
            "global_idx":      torch.tensor([s["global_idx"] for s in samples],
                                            dtype=torch.long),
        })

        # Build batched subgraph edge_index + batch vector for GNN-PE
        batched_edges = []
        batch_vec     = []
        node_offset   = 0
        for i, sample in enumerate(samples):
            eidx = sample["edge_index"]
            K_i  = sample["types"].size(0)
            batched_edges.append(eidx + node_offset)
            batch_vec.append(torch.full((K_i,), i, dtype=torch.long))
            node_offset += K_i

        out["edge_index"] = (torch.cat(batched_edges, dim=1)
                             if batched_edges
                             else torch.zeros((2, 0), dtype=torch.long))
        out["batch"] = (torch.cat(batch_vec)
                        if batch_vec
                        else torch.zeros((0,), dtype=torch.long))
        return out
