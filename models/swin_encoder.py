import torch
import torch.nn as nn
import timm


class SwinEncoder(nn.Module):
    def __init__(self, model_name="swin_small_patch4_window7_224", in_channels=4, pretrained=True):
        super().__init__()

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )

        self._modify_patch_embed(in_channels)

    def _modify_patch_embed(self, in_channels):
        old_proj = self.backbone.patch_embed.proj
        old_weight = old_proj.weight.data
        out_channels = old_proj.out_channels
        kernel_size = old_proj.kernel_size
        stride = old_proj.stride

        new_proj = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)

        with torch.no_grad():
            mean_weight = old_weight.mean(dim=1, keepdim=True)
            new_weight = mean_weight.repeat(1, in_channels, 1, 1)
            new_proj.weight.copy_(new_weight)

            if old_proj.bias is not None:
                new_proj.bias.copy_(old_proj.bias)

        self.backbone.patch_embed.proj = new_proj

    def forward(self, x):
        features = self.backbone(x)
        return features
