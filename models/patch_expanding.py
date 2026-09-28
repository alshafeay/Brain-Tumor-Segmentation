import torch
import torch.nn as nn


class PatchExpanding(nn.Module):
    def __init__(self, dim, dim_scale=2):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, dim_scale * dim, bias=False)
        self.norm = nn.LayerNorm(dim // dim_scale)

    def forward(self, x):
        # x shape: (B, H, W, C)
        B, H, W, C = x.shape

        x = self.expand(x)  # (B, H, W, dim_scale * C)

        x = x.view(B, H, W, 2, 2, C // 2)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, H * 2, W * 2, C // 2)

        x = self.norm(x)
        return x
