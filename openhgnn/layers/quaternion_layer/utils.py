import torch


# 定义一个函数，将输入数据转换为PyTorch的张量类型，如果输入已经是张量则直接返回
def to_tensor(X):
    if type(X) is torch.Tensor:
        return X
    return torch.Tensor(X)


# 定义一个函数，用于从权重矩阵中计算图的拉普拉斯矩阵，通过计算节点的度并进行归一化处理
def get_Laplacian_from_weights(weights):
    degree = torch.sum(weights, dim=1).pow(-0.5)
    return (weights * degree).t() * degree


# 定义一个函数，用于计算图的拉普拉斯矩阵，通过添加自环、计算度矩阵并进行对称归一化
def get_Laplacian(A):
    device = A.device
    dim = A.shape[0]
    L = A + torch.eye(dim).to(device)
    D = L.sum(dim=1)
    sqrt_D = D.pow(-1 / 2)
    Laplacian = sqrt_D * (sqrt_D * L).t()
    return Laplacian
