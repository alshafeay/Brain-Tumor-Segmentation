import torch
import torch.nn as nn
from timm.models.swin_transformer import SwinTransformerBlock
from models.patch_expanding import PatchExpanding


class DecoderBlock(nn.Module):
    def __init__(self, in_dim, skip_dim, out_dim, input_resolution, num_heads, window_size=7):
        super().__init__()

        self.patch_expand = PatchExpanding(dim=in_dim)

        self.reduce_channels = nn.Linear(in_dim // 2 + skip_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

        self.swin_block1 = SwinTransformerBlock(
            dim=out_dim,
            input_resolution=input_resolution,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=0,
        )

        self.swin_block2 = SwinTransformerBlock(
            dim=out_dim,
            input_resolution=input_resolution,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=window_size // 2,
        )

    def forward(self, x, skip):
        x = self.patch_expand(x)

        x = torch.cat([x, skip], dim=-1)
        x = self.reduce_channels(x)
        x = self.norm(x)

        B, H, W, C = x.shape
        x = x.view(B, H * W, C)

        x = self.swin_block1(x)
        x = self.swin_block2(x)

        x = x.view(B, H, W, C)
        return x
