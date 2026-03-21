"""
Implementation of HTGformer Layer
Strictly corresponds to each formula in the paper HTGformer (SIGIR 2025)
Corresponding formulas:
Formula (1): GCNAggregation.forward() — Non-parametric Graph Convolution Aggregation
Formula (2): NodeTypePrompt (LLM prompt construction) — Node Type Prompt
Formula (3): LLMTypeEncoder.encode() — LLM Generated Type Encoding
Formula (4): SinusoidalTemporalEncoding — Sine Temporal Encoding
Formula (5): HeteroTemporalEncoder.forward() — Splicing of Spatio-Temporal Encoding
Formula (6) (7): SpatioTemporalAttention — Transformer Attention
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as dglfn


# ===========================================================================
# Formula (1): Graph Embedding Layer
# H^t_{v,r} = A^t_r * H^t_{N^t_r(v)}
# Non-parametric graph convolution (without learnable parameters), symmetric normalized adjacency matrix
# ===========================================================================
class GCNAggregation(nn.Module):
    """
    Formula (1): Non-parametric Graph Convolution
    H^t_{v,r} = A^t_r * H^t_{N^t_r(v)}
    For each relationship type r, independently aggregate the neighbor features without including learnable parameters
    """
    def __init__(self):
        super().__init__()

    def forward(self, graph, feat_dict):
        """
        Args:
            graph: dgl.DGLGraph，Heterogeneous graph (at a single time step)
            feat_dict: {ntype: tensor [N, d]}，Characteristics of each node type
        Returns:
            agg_dict: {(ntype, etype, dtype): tensor [N_dst, d]}
                      The neighbor representations aggregated by the target node in each relationship context
        """
        agg_dict = {}
        for etype in graph.canonical_etypes:
            src_type, rel_type, dst_type = etype
            if src_type not in feat_dict:
                continue

            sub_g = graph[etype]
            src_feat = feat_dict[src_type]  # [N_src, d]

            # Calculate symmetric normalization: D^(-1/2) A D^(-1/2)
            # This is equivalent to message = src_feat / sqrt(deg_src),
            # Then sum, and finally divide by sqrt(deg_dst)
            src_deg = sub_g.out_degrees().float().clamp(min=1)
            dst_deg = sub_g.in_degrees().float().clamp(min=1)

            # Normalization of source node features
            norm_src = src_feat / src_deg.unsqueeze(-1).sqrt()

            sub_g.srcdata['h'] = norm_src
            sub_g.update_all(
                dglfn.copy_u('h', 'm'),
                dglfn.sum('m', 'h_agg')
            )
            h_agg = sub_g.dstdata['h_agg']  # [N_dst, d]

            # Normalized target node
            h_agg = h_agg / dst_deg.unsqueeze(-1).sqrt()
            agg_dict[etype] = h_agg

        return agg_dict


class GraphEmbeddingLayer(nn.Module):
    """
    Paper Section 3.1: Graph Embedding Layer
    For each time slice t and for each target node type v:
    1. For each relationship r, aggregate the neighbor features using formula (1) -> H^t_{v,r}
    2. Collect {H^t_{v,r} | r∈R(v)} ∪ {H^t_v} as the sequence input
    The final output sequence length is compressed from |V| to T * |T_n| (the number of node types).
    """

    def __init__(self):
        super().__init__()
        self.gcn = GCNAggregation()

    def forward(self, graphs, feat_dicts, category):
        """
        Args:
            graphs: List[DGLGraph], heterogeneous graphs over T time steps
            feat_dicts: List[dict], node features for T time steps
            category: str, target node type (e.g. 'paper')
        Returns:
            seq_list: List[dict]，每个时间步下目标节点的各视角表示
                      每个dict: {'self': [N,d], rel1: [N,d], ...}
        """
        seq_list = []
        for t, (graph, feat_dict) in enumerate(zip(graphs, feat_dicts)):
            t_repr = {'self': feat_dict[category]}  # H^t_v (自身特征)

            # Formula (1): Aggregate neighbors for each relationship targeted by category
            agg_dict = self.gcn(graph, feat_dict)
            for (src_type, rel_type, dst_type), h_agg in agg_dict.items():
                if dst_type == category:
                    t_repr[rel_type] = h_agg  # H^t_{v,r}

            seq_list.append(t_repr)
        return seq_list


# ===========================================================================
# Formula (2)(3): LLM Node Type Code
# Prompt(v) = {Introduction to type v; Instruction.}
# H^LLM_v = LLM(Prompt(v))
#
# Note: Due to the inability to invoke the LLM API in the experimental environment, two modes are provided:
#   - use_llm=True: Call the local LLM or load pre-computed LLM embeddings
#   - use_llm=False: Replace with randomly initialized learnable embeddings (ablation experiment without LLM)
# ===========================================================================
class LLMTypeEncoder(nn.Module):
    """

    Formula (2)(3): Semantic encoding of node types using
    LLM H^LLM_v = LLM(Prompt(v))

    The paper uses LLama3 to encode the textual descriptions of node types.
    The output dimension is the hidden layer dimension of LLM (4096 for LLama3), and then projected to hidden_dim.

   Practical strategy:
   Generate the embeddings for each node type in advance using LLM and save them. During training, load them directly (frozen).
   If there is no LLM resource, use learnable embeddings instead (corresponding to the ablation experiment w/o_LLM)

    """
    # The prompt templates of each node type in Figure 2 of the paper
    NODE_TYPE_PROMPTS = {
        'paper': (
            "Description: Academic paper is a type of node in the academic "
            "dynamic graph. These type of nodes are connected to Author nodes "
            "and Venues nodes. <Description of downstream tasks.>\n"
            "Instruction: Please output a summary of the information about "
            "this node type in the following format: "
            "{Introduction to node type:; Relevant relations analysis:}."
        ),
        'author': (
            "Description: Author is a type of node in the academic dynamic "
            "graph. These type of nodes are connected to Paper nodes. "
            "<Description of downstream tasks.>\n"
            "Instruction: Please output a summary of the information about "
            "this node type in the following format: "
            "{Introduction to node type:; Relevant relations analysis:}."
        ),
        'venue': (
            "Description: Venue is a type of node in the academic dynamic "
            "graph. These type of nodes represent publication venues such as "
            "conferences and journals. Connected to Paper nodes. "
            "<Description of downstream tasks.>\n"
            "Instruction: Please output a summary of the information about "
            "this node type in the following format: "
            "{Introduction to node type:; Relevant relations analysis:}."
        ),
        # OGBN-MAG Dataset node type
        'institution': (
            "Description: Institution is a type of node representing research "
            "institutions in the academic dynamic graph. Connected to Author nodes."
            "\nInstruction: Please output a summary in the format: "
            "{Introduction to node type:; Relevant relations analysis:}."
        ),
        'field_of_study': (
            "Description: Field of study is a type of node representing "
            "research fields in the academic dynamic graph. Connected to Paper nodes."
            "\nInstruction: Please output a summary in the format: "
            "{Introduction to node type:; Relevant relations analysis:}."
        ),
        # YELP Dataset
        'user': (
            "Description: User is a type of node in the review dynamic graph. "
            "Connected to Item nodes through review and tip relations."
            "\nInstruction: Please output a summary in the format: "
            "{Introduction to node type:; Relevant relations analysis:}."
        ),
        'item': (
            "Description: Item is a type of node in the review dynamic graph. "
            "Connected to User nodes through review and tip relations."
            "\nInstruction: Please output a summary in the format: "
            "{Introduction to node type:; Relevant relations analysis:}."
        ),
        # COVID-19 Dataset
        'state': (
            "Description: State is a type of node in the COVID-19 epidemic "
            "dynamic graph. Connected to County nodes and other State nodes."
            "\nInstruction: Please output a summary in the format: "
            "{Introduction to node type:; Relevant relations analysis:}."
        ),
        'county': (
            "Description: County is a type of node in the COVID-19 epidemic "
            "dynamic graph. Connected to State nodes and other County nodes."
            "\nInstruction: Please output a summary in the format: "
            "{Introduction to node type:; Relevant relations analysis:}."
        ),
    }

    def __init__(self, node_types, hidden_dim, llm_embed_dim=4096,
                 use_llm=False, llm_embed_path=None):
        """
        Args:
            node_types: List[str], List of node types
            hidden_dim: int, Output dimension d (in the paper, d = 64)
            llm_embed_dim: int, Output dimension of LLM (LLama3 = 4096, GPT3.5 = 1536)
            use_llm: bool, Whether to use pre-computed LLM embeddings
            llm_embed_path: str, File path for pre-computed LLM embeddings (.pt file)
        """
        super().__init__()
        self.node_types = node_types
        self.hidden_dim = hidden_dim
        self.use_llm = use_llm
        self.type2idx = {t: i for i, t in enumerate(node_types)}

        if use_llm and llm_embed_path is not None:
            # Load pre-computed LLM embeddings (frozen, not involved in gradient updates)
            llm_embeds = torch.load(llm_embed_path)  # {ntype: tensor[llm_dim]}
            embed_matrix = torch.stack(
                [llm_embeds[t] for t in node_types], dim=0
            )  # [num_types, llm_embed_dim]
            self.register_buffer('llm_embeds', embed_matrix)
            # Project onto hidden_dim
            self.proj = nn.Linear(llm_embed_dim, hidden_dim, bias=False)
            self.llm_loaded = True
        else:
            # Ablation experiment w/o_LLM: Replacing with learnable embeddings
            # Ablation experiment w/o_LLM: Replacing with learnable embeddings
            self.type_embedding = nn.Embedding(len(node_types), hidden_dim)
            self.llm_loaded = False

    def forward(self, node_type_ids):
        """
        Args:
            node_type_ids: tensor [N]，Node type index
        Returns:
            H_LLM: tensor [N, hidden_dim]，Node type code
        """
        if self.llm_loaded:
            # Formula (3): H^LLM_v = LLM(Prompt(v)), which has been pre-calculated
            raw = self.llm_embeds[node_type_ids]  # [N, llm_dim]
            return self.proj(raw)                  # [N, hidden_dim]
        else:
            # w/o_LLM
            return self.type_embedding(node_type_ids)  # [N, hidden_dim]

    @staticmethod
    def get_prompt(node_type):
        """Return the node type prompt in the format of Figure 2 from the paper."""
        return LLMTypeEncoder.NODE_TYPE_PROMPTS.get(
            node_type,
            f"Description: {node_type} is a type of node in the dynamic graph.\n"
            f"Instruction: Please output a summary in the format: "
            f"{{Introduction to node type:; Relevant relations analysis:}}."
        )


# ===========================================================================
# Formula (4)(5): Sine time sequence encoding + Spatiotemporal encoding
# p^t_i = sin/cos(t / 10000^{2i/d})
# H^{sp,t}_v = ||^d_{i=1} (H^LLM_v + p^t_i)
# ===========================================================================
class SinusoidalTemporalEncoding(nn.Module):
    """
    Formula (4): Sine time sequence encoding (the same as the original position encoding of Transformer)
    p^t_i = sin(t/10000^{2i/d})  if i=2k
          = cos(t/10000^{2i/d})  if i=2k+1
    """
    def __init__(self, hidden_dim, max_timestamps=1000):
        super().__init__()
        # 预计算所有时间步的编码
        pe = torch.zeros(max_timestamps, hidden_dim)
        position = torch.arange(0, max_timestamps, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2).float()
            * (-math.log(10000.0) / hidden_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)   # 偶数维度 sin
        pe[:, 1::2] = torch.cos(position * div_term)   # 奇数维度 cos
        self.register_buffer('pe', pe)  # [max_T, d]

    def forward(self, t):
        """
        Args:
            t: int，Time step index
        Returns:
            p_t: tensor [d]，Encoding at time step t
        """
        return self.pe[t]  # [d]


class HeteroTemporalEncoder(nn.Module):
    """
    Paper Section 3.2: Hetero-Temporal Encoder


    Formula (5): H^{sp,t}_v = ||^d_{i=1} (H^LLM_v + p^t_i)
    Note: The formula in the paper uses || (concatenate across dimensions)
    The actual meaning is: For each dimension i, concatenate H^LLM_v[i] + p^t_i
    This is equivalent to: H^{sp,t}_v = H^LLM_v + p^t (element-wise addition, with the same dimension d)
    
    Finally, the time-space encoding will be concatenated with the node features.：
    Z^t_v = concat(H^t_v, H^{sp,t}_v)  — 论文Section 3.2最后一段
    """
    def __init__(self, node_types, feat_dim_dict, hidden_dim,
                 use_llm=False, llm_embed_path=None, llm_embed_dim=4096):
        """
        Args:
            node_types: List[str]
            feat_dim_dict: {ntype: int}，The original feature dimensions of each node type
            hidden_dim: int，d=64（Paper hyperparameters）
            use_llm: bool
            llm_embed_path: str
        """
        super().__init__()
        self.node_types = node_types
        self.hidden_dim = hidden_dim
        self.type2idx = {t: i for i, t in enumerate(node_types)}

        # Formula (3): LLM Type Encoding
        self.llm_encoder = LLMTypeEncoder(
            node_types=node_types,
            hidden_dim=hidden_dim,
            use_llm=use_llm,
            llm_embed_dim=llm_embed_dim,
            llm_embed_path=llm_embed_path
        )

        # Formula (4): Sine waveform encoding
        self.temporal_enc = SinusoidalTemporalEncoding(hidden_dim)

        # Feature projection: Project the original features of different dimensions to hidden_dim
        # Establish projection layers for the original features of each node type
        self.feat_proj = nn.ModuleDict({
            ntype: nn.Linear(feat_dim_dict[ntype], hidden_dim, bias=True)
            for ntype in node_types if ntype in feat_dim_dict
        })
        # General Projection Layer: Used to handle neighbor features with different dimensions after GCN aggregation
        # The dimension of the aggregated features is equal to the original feature dimension of the source node. Dynamic projection is required.
        self._proj_cache = nn.ModuleDict()
        self._feat_dim_dict = dict(feat_dim_dict)

    def _get_proj(self, in_dim, device):
        """
        Dynamically obtain or create projection layers corresponding to the input dimensions
        Used to handle the situation where the dimensions after GCN aggregation are different from the original feature dimensions
        """
        key = str(in_dim)
        if key not in self._proj_cache:
            proj = nn.Linear(in_dim, self.hidden_dim, bias=True).to(device)
            nn.init.xavier_uniform_(proj.weight)
            nn.init.zeros_(proj.bias)
            self._proj_cache[key] = proj
        return self._proj_cache[key]

    def forward(self, feat_dict, timestep, category):
        """
        For the target node type "category", generate the enhanced node representation Z^t_v

        Args:
            feat_dict: {ntype: tensor [N, feat_dim]}, where feat_dim can be of any dimension
            timestep: int, the current time step t
            category: str, the target node type
        Returns:
            Z_t: tensor [N_cat, 2*hidden_dim]
                 = concat(H^t_v_proj, H^{sp,t}_v)
                 H^{sp,t}_v = H^LLM_v + p^t（公式5）
        """
        feat = feat_dict[category]
        N = feat.shape[0]
        in_dim = feat.shape[1]
        device = feat.device

        # 1. Project the original features to hidden_dim -> H^t_v
        # If it is the feature of the target node itself, use the pre-built feat_proj
        # If it is the neighbor features aggregated by GCN (with a different dimension), use dynamic projection
        if category in self.feat_proj and in_dim == self._feat_dim_dict.get(category, -1):
            H_t = self.feat_proj[category](feat)  # [N, d]
        else:
            H_t = self._get_proj(in_dim, device)(feat)  # [N, d]

        # 2. Formula (3): LLM type encoding H^LLM_v
        type_idx = self.type2idx[category]
        type_ids = torch.full((N,), type_idx, dtype=torch.long, device=device)
        H_LLM = self.llm_encoder(type_ids)  # [N, d]

        # 3. Formula (4): Sine time-sequential encoding p^t
        p_t = self.temporal_enc(timestep).to(device)  # [d]

        # 4. Formula (5): H^{sp,t}_v = H^LLM_v + p^t
        H_sp_t = H_LLM + p_t.unsqueeze(0)  # [N, d]

        # 5. Paper： Section 3.2: Z^t_v = concat(H^t_v, H^{sp,t}_v)
        Z_t = torch.cat([H_t, H_sp_t], dim=-1)  # [N, 2*d]

        return Z_t  # [N, 2*d]


# ===========================================================================
# Formula (6)(7): Spatio-Temporal Attention（StandardTransformer）
# Q = W_Q Z_v,  K = W_K Z,  V = W_V Z
# Z'_v = Softmax(QK^T / sqrt(d_K)) V
# Key point: W_K, W_Q, W_V are shared among all node types (as shown in Figure 1(d) of the paper)
# ===========================================================================
class SpatioTemporalAttention(nn.Module):
    """
    Formula (6) (7): Spatio-Temporal Attention

    Key design of the paper:
    1. The QKV parameters are shared across all node types ("Shared across all node type")
    2. The input Z already contains temporal and spatial encoding, so the attention naturally captures the temporal and spatial relationships
    3. The output Z'_v = [Z^{T+1}_v, ..., Z^{2T}_v] represents the predicted future node representation
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        """
        Args:
            embed_dim: int，Input dimension ( = 2 * hidden_dim, as the features and spatiotemporal encoding are concatenated)
            num_heads: int，Number of attention points
            dropout: float
        """
        super().__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        # Formula (6): Shared QKV linear transformation (shared among all node types)
        self.W_Q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_K = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_V = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, Z_v, Z, mask=None):
        """
        Args:
            Z_v: tensor [N, T, embed_dim]，The sequence (query) of the target node
            Z: tensor [N, L, embed_dim]，The full sequence (key/value)
               L = T*(1 + |relations|)，Including oneself + all related neighbors
            mask: Optional tensor，Attention mask
        Returns:
            Z_prime: tensor [N, T, embed_dim]，Predicted node representation
        """
        N, T, _ = Z_v.shape
        L = Z.shape[1]

        # Formula (6): Q = W_Q Z_v, K = W_K Z, V = W_V Z
        Q = self.W_Q(Z_v)  # [N, T, d]
        K = self.W_K(Z)    # [N, L, d]
        V = self.W_V(Z)    # [N, L, d]

        # Horizontal split
        Q = Q.view(N, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(N, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(N, L, self.num_heads, self.head_dim).transpose(1, 2)
        # Shape : [N, num_heads, seq_len, head_dim]

        # Formula (7): Softmax(QK^T / sqrt(d_K)) V
        attn = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, V)  # [N, num_heads, T, head_dim]
        out = out.transpose(1, 2).contiguous().view(N, T, self.embed_dim)
        Z_prime = self.out_proj(out)  # [N, T, embed_dim]

        return Z_prime


class HTGformerEncoderLayer(nn.Module):
    """
    The complete Transformer encoding layer:
    Spatio-Temporal Attention + Add&Norm + FFN + Add&Norm
    (The "Add & Norm" structure in Figure 1(a) of the paper)
    Use Pre-LayerNorm (for more stable training)
    """
    def __init__(self, embed_dim, num_heads, ffn_dim=None, dropout=0.1):
        super().__init__()
        ffn_dim = ffn_dim or embed_dim * 4
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = SpatioTemporalAttention(embed_dim, num_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, Z_v, Z):
        # Pre-LN + 残差
        Z_v_norm = self.norm1(Z_v)
        Z_norm = self.norm1(Z)
        Z_v = Z_v + self.attn(Z_v_norm, Z_norm)
        Z_v = Z_v + self.ffn(self.norm2(Z_v))
        return Z_v
