import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticAttention(nn.Module):

    def __init__(self, in_ft: int, out_ft: int):
        super().__init__()
        self.in_ft = in_ft
        self.out_ft = out_ft

        self.proj = nn.Linear(in_ft, out_ft, bias=True)
        self.score = nn.Linear(out_ft, 1, bias=False)
        self.tanh = nn.Tanh()

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)
        nn.init.xavier_uniform_(self.score.weight)

    def forward(self, x, nb_rel: int = None) -> torch.Tensor:
        """
        返回：
            [N, F]
        """
        # 情况 1：输入是 list[Tensor]，每个元素 [N, F]
        if isinstance(x, (list, tuple)):
            if len(x) == 0:
                raise ValueError("SemanticAttention received an empty list.")

            h = torch.stack(x, dim=0)  # [R, N, F]

        # 情况 2：输入是 Tensor
        elif isinstance(x, torch.Tensor):
            if x.dim() == 3:
                # [R, N, F]
                h = x
            elif x.dim() == 2:
                # [R*N, F] -> 需要 nb_rel
                if nb_rel is None:
                    raise ValueError(
                        "When x is 2D tensor [R*N, F], nb_rel must be provided."
                    )
                total_nodes = x.size(0)
                if total_nodes % nb_rel != 0:
                    raise ValueError(
                        f"x.size(0)={total_nodes} is not divisible by nb_rel={nb_rel}."
                    )
                n = total_nodes // nb_rel
                h = x.view(nb_rel, n, x.size(1))  # [R, N, F]
            else:
                raise ValueError(
                    f"Unsupported tensor shape for SemanticAttention: {tuple(x.shape)}"
                )
        else:
            raise TypeError("SemanticAttention input must be a Tensor or a list/tuple of Tensors.")

        # h: [R, N, F]
        r, n, f = h.shape
        if f != self.in_ft:
            raise ValueError(
                f"Expected input feature dim {self.in_ft}, but got {f}."
            )

        # 每个 relation 的全局语义分数
        z = self.tanh(self.proj(h))        # [R, N, out_ft]
        s = self.score(z).mean(dim=1)      # [R, 1]
        beta = F.softmax(s, dim=0)         # [R, 1]
        beta = beta.view(r, 1, 1)          # [R, 1, 1]

        out = (beta * h).sum(dim=0)        # [N, F]
        return out