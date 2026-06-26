import torch
import torch.nn as nn
import torch.nn.functional as F


class FullyConnect(nn.Module):

    def __init__(
        self,
        in_ft: int,
        out_ft: int,
        act: nn.Module = None,
        drop_prob: float = 0.0,
        isBias: bool = False
    ):
        super().__init__()

        self.fc = nn.Linear(in_ft, out_ft, bias=False)

        if isBias:
            self.bias = nn.Parameter(torch.zeros(out_ft))
        else:
            self.register_parameter("bias", None)

        self.act = act if act is not None else nn.Identity()
        self.drop_prob = float(drop_prob)
        self.isBias = isBias

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        emb = F.dropout(emb, p=self.drop_prob, training=self.training)
        out = self.fc(emb)

        if self.bias is not None:
            out = out + self.bias

        out = self.act(out)
        return out


class FullyConnect2(nn.Module):

    def __init__(
        self,
        in_ft: int,
        hid_unit: int,
        out_ft: int,
        drop_prob: float = 0.0,
        isBias: bool = False
    ):
        super().__init__()
        self.fc1 = FullyConnect(
            in_ft, hid_unit,
            act=nn.PReLU(),
            drop_prob=drop_prob,
            isBias=isBias
        )
        self.fc2 = FullyConnect(
            hid_unit, out_ft,
            act=nn.PReLU(),
            drop_prob=drop_prob,
            isBias=isBias
        )
        self.out_proj = FullyConnect(
            out_ft, 1,
            act=nn.Identity(),
            drop_prob=drop_prob,
            isBias=isBias
        )

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        x = self.fc1(emb)
        x = self.fc2(x)
        x = self.out_proj(x)
        return x