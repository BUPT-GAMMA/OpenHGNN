"""
HTGformer Layer 实现
严格对应论文 HTGformer (SIGIR 2025) 各公式

公式对应关系：
  公式(1): GCNAggregation.forward()       — 非参数图卷积聚合
  公式(2): NodeTypePrompt (LLM prompt构造) — 节点类型prompt
  公式(3): LLMTypeEncoder.encode()         — LLM生成类型编码
  公式(4): SinusoidalTemporalEncoding      — 正弦时序编码
  公式(5): HeteroTemporalEncoder.forward() — 时空编码拼接
  公式(6)(7): SpatioTemporalAttention      — Transformer注意力
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as dglfn


# ===========================================================================
# 公式(1): Graph Embedding Layer
# H^t_{v,r} = A^t_r * H^t_{N^t_r(v)}
# 非参数图卷积（无可学习参数），对称归一化邻接矩阵
# ===========================================================================
class GCNAggregation(nn.Module):
    """
    公式(1): 非参数图卷积
    H^t_{v,r} = A^t_r * H^t_{N^t_r(v)}
    对每种关系类型r独立聚合邻居特征，不含可学习参数
    """
    def __init__(self):
        super().__init__()

    def forward(self, graph, feat_dict):
        """
        Args:
            graph: dgl.DGLGraph，异质图（单个时间步）
            feat_dict: {ntype: tensor [N, d]}，各节点类型的特征
        Returns:
            agg_dict: {(ntype, etype, dtype): tensor [N_dst, d]}
                      每种关系下目标节点聚合到的邻居表示
        """
        agg_dict = {}
        for etype in graph.canonical_etypes:
            src_type, rel_type, dst_type = etype
            if src_type not in feat_dict:
                continue

            sub_g = graph[etype]
            src_feat = feat_dict[src_type]  # [N_src, d]

            # 计算对称归一化：D^{-1/2} A D^{-1/2}
            # 等价于 message = src_feat / sqrt(deg_src), 
            # 再 sum, 再除以 sqrt(deg_dst)
            src_deg = sub_g.out_degrees().float().clamp(min=1)
            dst_deg = sub_g.in_degrees().float().clamp(min=1)

            # 归一化源节点特征
            norm_src = src_feat / src_deg.unsqueeze(-1).sqrt()

            sub_g.srcdata['h'] = norm_src
            sub_g.update_all(
                dglfn.copy_u('h', 'm'),
                dglfn.sum('m', 'h_agg')
            )
            h_agg = sub_g.dstdata['h_agg']  # [N_dst, d]

            # 归一化目标节点
            h_agg = h_agg / dst_deg.unsqueeze(-1).sqrt()
            agg_dict[etype] = h_agg

        return agg_dict


class GraphEmbeddingLayer(nn.Module):
    """
    论文 Section 3.1: Graph Embedding Layer
    
    对每个时间切片t，对目标节点类型v：
      1. 对每种关系r，用公式(1)聚合邻居特征 -> H^t_{v,r}
      2. 收集 {H^t_{v,r} | r∈R(v)} ∪ {H^t_v} 作为序列输入
    
    最终输出序列长度从 |V| 压缩到 T * |T_n|（节点类型数）
    """
    def __init__(self):
        super().__init__()
        self.gcn = GCNAggregation()

    def forward(self, graphs, feat_dicts, category):
        """
        Args:
            graphs: List[DGLGraph]，T个时间步的异质图
            feat_dicts: List[dict]，T个时间步的节点特征
            category: str，目标节点类型（如'paper'）
        Returns:
            seq_list: List[dict]，每个时间步下目标节点的各视角表示
                      每个dict: {'self': [N,d], rel1: [N,d], ...}
        """
        seq_list = []
        for t, (graph, feat_dict) in enumerate(zip(graphs, feat_dicts)):
            t_repr = {'self': feat_dict[category]}  # H^t_v (自身特征)

            # 公式(1): 对每种以category为目标的关系聚合邻居
            agg_dict = self.gcn(graph, feat_dict)
            for (src_type, rel_type, dst_type), h_agg in agg_dict.items():
                if dst_type == category:
                    t_repr[rel_type] = h_agg  # H^t_{v,r}

            seq_list.append(t_repr)
        return seq_list


# ===========================================================================
# 公式(2)(3): LLM节点类型编码
# Prompt(v) = {Introduction to type v; Instruction.}
# H^LLM_v = LLM(Prompt(v))
#
# 注意：由于实验环境无法调用LLM API，提供两种模式：
#   - use_llm=True: 调用本地LLM或加载预计算的LLM嵌入
#   - use_llm=False: 用随机初始化的可学习嵌入替代（消融实验w/o_LLM）
# ===========================================================================
class LLMTypeEncoder(nn.Module):
    """
    公式(2)(3): 节点类型的LLM语义编码
    H^LLM_v = LLM(Prompt(v))
    
    论文使用 LLama3 对节点类型的文本描述进行编码
    输出维度为LLM隐层维度（LLama3为4096），再投影到hidden_dim
    
    实用策略：
      预先用LLM生成各节点类型的嵌入并保存，训练时直接加载（frozen）
      若无LLM资源，用可学习嵌入替代（对应消融实验 w/o_LLM）
    """
    # 论文Figure 2中各节点类型的prompt模板
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
        # OGBN-MAG数据集节点类型
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
        # YELP数据集
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
        # COVID-19数据集
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
            node_types: List[str]，节点类型列表
            hidden_dim: int，输出维度d（论文中d=64）
            llm_embed_dim: int，LLM输出维度（LLama3=4096, GPT3.5=1536）
            use_llm: bool，是否使用预计算LLM嵌入
            llm_embed_path: str，预计算LLM嵌入的文件路径（.pt文件）
        """
        super().__init__()
        self.node_types = node_types
        self.hidden_dim = hidden_dim
        self.use_llm = use_llm
        self.type2idx = {t: i for i, t in enumerate(node_types)}

        if use_llm and llm_embed_path is not None:
            # 加载预计算的LLM嵌入（frozen，不参与梯度更新）
            llm_embeds = torch.load(llm_embed_path)  # {ntype: tensor[llm_dim]}
            embed_matrix = torch.stack(
                [llm_embeds[t] for t in node_types], dim=0
            )  # [num_types, llm_embed_dim]
            self.register_buffer('llm_embeds', embed_matrix)
            # 投影到hidden_dim
            self.proj = nn.Linear(llm_embed_dim, hidden_dim, bias=False)
            self.llm_loaded = True
        else:
            # 消融实验 w/o_LLM：用可学习嵌入替代
            # 论文消融实验证明LLM编码有帮助但模型仍可运行
            self.type_embedding = nn.Embedding(len(node_types), hidden_dim)
            self.llm_loaded = False

    def forward(self, node_type_ids):
        """
        Args:
            node_type_ids: tensor [N]，节点类型索引
        Returns:
            H_LLM: tensor [N, hidden_dim]，节点类型编码
        """
        if self.llm_loaded:
            # 公式(3): H^LLM_v = LLM(Prompt(v))，已预计算
            raw = self.llm_embeds[node_type_ids]  # [N, llm_dim]
            return self.proj(raw)                  # [N, hidden_dim]
        else:
            # w/o_LLM 替代方案
            return self.type_embedding(node_type_ids)  # [N, hidden_dim]

    @staticmethod
    def get_prompt(node_type):
        """返回论文Figure 2格式的节点类型prompt"""
        return LLMTypeEncoder.NODE_TYPE_PROMPTS.get(
            node_type,
            f"Description: {node_type} is a type of node in the dynamic graph.\n"
            f"Instruction: Please output a summary in the format: "
            f"{{Introduction to node type:; Relevant relations analysis:}}."
        )


# ===========================================================================
# 公式(4)(5): 正弦时序编码 + 时空编码
# p^t_i = sin/cos(t / 10000^{2i/d})
# H^{sp,t}_v = ||^d_{i=1} (H^LLM_v + p^t_i)
# ===========================================================================
class SinusoidalTemporalEncoding(nn.Module):
    """
    公式(4): 正弦时序编码（与Transformer原始位置编码相同）
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
            t: int，时间步索引
        Returns:
            p_t: tensor [d]，时间步t的编码
        """
        return self.pe[t]  # [d]


class HeteroTemporalEncoder(nn.Module):
    """
    论文 Section 3.2: Hetero-Temporal Encoder
    
    公式(5): H^{sp,t}_v = ||^d_{i=1} (H^LLM_v + p^t_i)
    注意：论文公式(5)用的是 || (concat across dimensions)
    实际含义是：对每个维度i，将H^LLM_v[i] + p^t_i 拼接
    等价于：H^{sp,t}_v = H^LLM_v + p^t（element-wise add，维度d相同）
    
    最终将时空编码与节点特征concat：
    Z^t_v = concat(H^t_v, H^{sp,t}_v)  — 论文Section 3.2最后一段
    """
    def __init__(self, node_types, feat_dim_dict, hidden_dim,
                 use_llm=False, llm_embed_path=None):
        """
        Args:
            node_types: List[str]
            feat_dim_dict: {ntype: int}，各节点类型原始特征维度
            hidden_dim: int，d=64（论文超参数）
            use_llm: bool
            llm_embed_path: str
        """
        super().__init__()
        self.node_types = node_types
        self.hidden_dim = hidden_dim
        self.type2idx = {t: i for i, t in enumerate(node_types)}

        # 公式(3): LLM类型编码
        self.llm_encoder = LLMTypeEncoder(
            node_types=node_types,
            hidden_dim=hidden_dim,
            use_llm=use_llm,
            llm_embed_path=llm_embed_path
        )

        # 公式(4): 正弦时序编码
        self.temporal_enc = SinusoidalTemporalEncoding(hidden_dim)

        # 特征投影：将不同维度的原始特征投影到hidden_dim
        # 为每种节点类型的原始特征建立投影层
        self.feat_proj = nn.ModuleDict({
            ntype: nn.Linear(feat_dim_dict[ntype], hidden_dim, bias=True)
            for ntype in node_types if ntype in feat_dim_dict
        })
        # 通用投影层：用于处理GCN聚合后维度不同的邻居特征
        # 聚合后特征维度等于源节点原始特征维度，需动态投影
        self._proj_cache = nn.ModuleDict()
        self._feat_dim_dict = dict(feat_dim_dict)

    def _get_proj(self, in_dim, device):
        """
        动态获取或创建对应输入维度的投影层
        用于处理GCN聚合后维度与原始特征维度不同的情况
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
        对目标节点类型category，生成增强后的节点表示 Z^t_v

        Args:
            feat_dict: {ntype: tensor [N, feat_dim]}，feat_dim可以是任意维度
            timestep: int，当前时间步t
            category: str，目标节点类型
        Returns:
            Z_t: tensor [N_cat, 2*hidden_dim]
                 = concat(H^t_v_proj, H^{sp,t}_v)
                 其中 H^{sp,t}_v = H^LLM_v + p^t（公式5）
        """
        feat = feat_dict[category]
        N = feat.shape[0]
        in_dim = feat.shape[1]
        device = feat.device

        # 1. 投影原始特征到 hidden_dim -> H^t_v
        # 如果是目标节点自身特征，用预建的feat_proj
        # 如果是GCN聚合后的邻居特征（维度可能不同），用动态投影
        if category in self.feat_proj and in_dim == self._feat_dim_dict.get(category, -1):
            H_t = self.feat_proj[category](feat)  # [N, d]
        else:
            H_t = self._get_proj(in_dim, device)(feat)  # [N, d]

        # 2. 公式(3): LLM类型编码 H^LLM_v
        type_idx = self.type2idx[category]
        type_ids = torch.full((N,), type_idx, dtype=torch.long, device=device)
        H_LLM = self.llm_encoder(type_ids)  # [N, d]

        # 3. 公式(4): 正弦时序编码 p^t
        p_t = self.temporal_enc(timestep).to(device)  # [d]

        # 4. 公式(5): H^{sp,t}_v = H^LLM_v + p^t
        H_sp_t = H_LLM + p_t.unsqueeze(0)  # [N, d]

        # 5. 论文Section 3.2: Z^t_v = concat(H^t_v, H^{sp,t}_v)
        Z_t = torch.cat([H_t, H_sp_t], dim=-1)  # [N, 2*d]

        return Z_t  # [N, 2*d]


# ===========================================================================
# 公式(6)(7): Spatio-Temporal Attention（标准Transformer）
# Q = W_Q Z_v,  K = W_K Z,  V = W_V Z
# Z'_v = Softmax(QK^T / sqrt(d_K)) V
# 关键：W_K, W_Q, W_V 在所有节点类型间共享（论文Figure 1(d)）
# ===========================================================================
class SpatioTemporalAttention(nn.Module):
    """
    公式(6)(7): Spatio-Temporal Attention
    
    论文关键设计：
    1. QKV参数在所有节点类型间共享（"Shared across all node type"）
    2. 输入Z已包含时空编码，因此注意力自然捕获时空关系
    3. 输出 Z'_v = [Z^{T+1}_v, ..., Z^{2T}_v] 为预测的未来节点表示
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        """
        Args:
            embed_dim: int，输入维度（= 2*hidden_dim，因为concat了特征和时空编码）
            num_heads: int，注意力头数
            dropout: float
        """
        super().__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        # 公式(6): 共享的QKV线性变换（所有节点类型共享）
        self.W_Q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_K = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_V = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, Z_v, Z, mask=None):
        """
        Args:
            Z_v: tensor [N, T, embed_dim]，目标节点的序列（query）
            Z: tensor [N, L, embed_dim]，全序列（key/value）
               L = T*(1 + |relations|)，包含自身+所有关系邻居
            mask: Optional tensor，注意力mask
        Returns:
            Z_prime: tensor [N, T, embed_dim]，预测的节点表示
        """
        N, T, _ = Z_v.shape
        L = Z.shape[1]

        # 公式(6): Q = W_Q Z_v, K = W_K Z, V = W_V Z
        Q = self.W_Q(Z_v)  # [N, T, d]
        K = self.W_K(Z)    # [N, L, d]
        V = self.W_V(Z)    # [N, L, d]

        # 多头拆分
        Q = Q.view(N, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(N, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(N, L, self.num_heads, self.head_dim).transpose(1, 2)
        # 形状: [N, num_heads, seq_len, head_dim]

        # 公式(7): Softmax(QK^T / sqrt(d_K)) V
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
    完整的Transformer编码层：
    Spatio-Temporal Attention + Add&Norm + FFN + Add&Norm
    （论文Figure 1(a)中的 "Add & Norm" 结构）
    使用Pre-LayerNorm（训练更稳定）
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
