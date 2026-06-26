import torch
import torch.nn as nn


class Discriminator(nn.Module):

    def __init__(self, v_ft: int, u_ft: int):
        super().__init__()
        self.bilinear = nn.Bilinear(v_ft, u_ft, 1)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.bilinear.weight)
        if self.bilinear.bias is not None:
            nn.init.zeros_(self.bilinear.bias)

    def forward(self, v_h: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        if c.dim() == 1:
            c = c.unsqueeze(0).expand(v_h.size(0), -1)

        out = self.bilinear(v_h, c)
        return out.squeeze(-1)