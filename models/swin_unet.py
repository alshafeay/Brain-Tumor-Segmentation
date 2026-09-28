import torch
import torch.nn as nn
from models.swin_encoder import SwinEncoder
from models.decoder_block import DecoderBlock
from models.patch_expanding import PatchExpanding


class FinalExpanding(nn.Module):
    def __init__(self, dim, num_classes):
        super().__init__()
        self.expand = nn.Sequential(
            PatchExpanding(dim=dim),
            PatchExpanding(dim=dim // 2),
        )
        self.head = nn.Conv2d(dim // 4, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.expand(x)
        x = x.permute(0, 3, 1, 2)
        x = self.head(x)
        return x


class SwinUNet(nn.Module):
    def __init__(self, in_channels=4, num_classes=1, window_size=7):
        super().__init__()

        self.encoder = SwinEncoder(in_channels=in_channels, pretrained=True)

        self.decoder1 = DecoderBlock(
            in_dim=768, skip_dim=384, out_dim=384,
            input_resolution=(14, 14), num_heads=12, window_size=window_size,
        )
        self.decoder2 = DecoderBlock(
            in_dim=384, skip_dim=192, out_dim=192,
            input_resolution=(28, 28), num_heads=6, window_size=window_size,
        )
        self.decoder3 = DecoderBlock(
            in_dim=192, skip_dim=96, out_dim=96,
            input_resolution=(56, 56), num_heads=3, window_size=window_size,
        )

        self.final_expand = FinalExpanding(dim=96, num_classes=num_classes)

    def forward(self, x):
        features = self.encoder(x)
        stage0, stage1, stage2, stage3 = features

        x = self.decoder1(stage3, stage2)
        x = self.decoder2(x, stage1)
        x = self.decoder3(x, stage0)

        out = self.final_expand(x)
        return out
