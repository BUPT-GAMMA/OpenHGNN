import torch
from scipy.sparse import csr_matrix
import numpy as np
import dgl


def initialize(target, msg, d_e, d_r, B):
    init_emb_ent = torch.zeros((target.num_ent, d_e)).cuda()
    init_emb_rel = torch.zeros((2 * target.num_rel, d_r)).cuda()
    gain = torch.nn.init.calculate_gain('relu')
    torch.nn.init.xavier_normal_(init_emb_ent, gain=gain)
    torch.nn.init.xavier_normal_(init_emb_rel, gain=gain)
    relation_triplets = generate_relation_triplets(msg, target.num_ent, target.num_rel, B)
    relation_triplets = torch.tensor(relation_triplets).cuda()
    return init_emb_ent, init_emb_rel, relation_triplets


def create_relation_graph(triplet, num_ent, num_rel):
    ind_h = triplet[:, :2]
    ind_t = triplet[:, 1:]
    E_h = csr_matrix((np.ones(len(ind_h)), (ind_h[:, 0], ind_h[:, 1])), shape=(num_ent, 2 * num_rel))
    E_t = csr_matrix((np.ones(len(ind_t)), (ind_t[:, 1], ind_t[:, 0])), shape=(num_ent, 2 * num_rel))
    diag_vals_h = E_h.sum(axis=1).A1
    diag_vals_h[diag_vals_h != 0] = 1 / (diag_vals_h[diag_vals_h != 0] ** 2)
    diag_vals_t = E_t.sum(axis=1).A1
    diag_vals_t[diag_vals_t != 0] = 1 / (diag_vals_t[diag_vals_t != 0] ** 2)
    D_h_inv = csr_matrix((diag_vals_h, (np.arange(num_ent), np.arange(num_ent))), shape=(num_ent, num_ent))
    D_t_inv = csr_matrix((diag_vals_t, (np.arange(num_ent), np.arange(num_ent))), shape=(num_ent, num_ent))
    A_h = E_h.transpose() @ D_h_inv @ E_h
    A_t = E_t.transpose() @ D_t_inv @ E_t
    return A_h + A_t


def get_rank(triplet, scores, filters, target=0):
    thres = scores[triplet[0, target]].item()
    scores[filters] = thres - 1
    rank = (scores > thres).sum() + (scores == thres).sum() // 2 + 1
    return rank.item()


def get_metrics(rank):
    rank = np.array(rank, dtype=np.int)
    mr = np.mean(rank)
    mrr = np.mean(1 / rank)
    hit10 = np.sum(rank < 11) / len(rank)
    hit3 = np.sum(rank < 4) / len(rank)
    hit1 = np.sum(rank < 2) / len(rank)
    return mr, mrr, hit10, hit3, hit1


def generate_neg(triplets, num_ent, num_neg=1):
    neg_triplets = triplets.unsqueeze(dim=1).repeat(1, num_neg, 1)
    rand_result = torch.rand((len(triplets), num_neg)).cuda()
    perturb_head = rand_result < 0.5
    perturb_tail = rand_result >= 0.5
    rand_idxs = torch.randint(low=0, high=num_ent - 1, size=(len(triplets), num_neg)).cuda()
    rand_idxs[perturb_head] += rand_idxs[perturb_head] >= neg_triplets[:, :, 0][perturb_head]
    rand_idxs[perturb_tail] += rand_idxs[perturb_tail] >= neg_triplets[:, :, 2][perturb_tail]
    neg_triplets[:, :, 0][perturb_head] = rand_idxs[perturb_head]
    neg_triplets[:, :, 2][perturb_tail] = rand_idxs[perturb_tail]
    neg_triplets = torch.cat(torch.split(neg_triplets, 1, dim=1), dim=0).squeeze(dim=1)
    return neg_triplets


def generate_relation_triplets(triplet, num_ent, num_rel, B):
    A = create_relation_graph(triplet, num_ent, num_rel)
    A_sparse = csr_matrix(A)
    G_rel = dgl.from_scipy(A_sparse)  # 这里用dgl.from_scipy()函数创建一个图对象
    G_rel.edata['weight'] = torch.from_numpy(A.data)  # 这里用A.data获取稀疏矩阵的非零元素，作为边的权重
    relation_triplets = get_relation_triplets(G_rel, B)
    return relation_triplets


def get_relation_triplets(G_rel, B):
    src, dst = G_rel.edges()  # 获取边的源节点和目标节点
    w = G_rel.edata['weight']  # 获取边的权重
    nnz = len(w)  # 获取边的数量
    temp = torch.argsort(-w)  # 对边的权重进行降序排序，并返回排序后的索引
    weight_ranks = torch.empty_like(temp)  # 创建一个空的张量，用于存储权重的排名
    weight_ranks[temp] = torch.arange(nnz) + 1  # 根据排序后的索引，给每个权重赋予一个排名
    rk = torch.floor(weight_ranks / nnz * B) - 1  # 把权重的排名映射到一个区间[0, B-1]
    rk = rk.int()  # 把映射后的权重转换为整数
    relation_triplets = torch.stack([src, dst, rk], dim=1)  # 按列拼接三个张量
    relation_triplets = relation_triplets.numpy()  # 把张量转换为numpy数组
    return relation_triplets
