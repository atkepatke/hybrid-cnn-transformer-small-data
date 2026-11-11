# src/model.py
import torch
import torch.nn as nn
from attention import ConvAttention, ConvControl
from vit import SmallViT

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

        if stride != 1 or in_ch != out_ch:   # shapes differ -> 1x1 conv to match
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + self.shortcut(x))   # the residual connection


class SmallResNet(nn.Module):
    def __init__(self, num_classes=10, in_channels=3,
                 widths=(32, 64, 128, 256), blocks_per_stage=2,
                 attn_after=None, attn_depth=1, num_heads=4, image_size=32,attn_type="attention"):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, widths[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(widths[0]),
            nn.ReLU(),
        )

        stages = []
        in_ch = widths[0]
        for i, width in enumerate(widths):
            stage_stride = 1 if i == 0 else 2          # stage 1 keeps 32x32
            blocks = []
            for j in range(blocks_per_stage):
                stride = stage_stride if j == 0 else 1  # only first block downsamples
                blocks.append(ResidualBlock(in_ch, width, stride))
                in_ch = width
            if attn_after == i + 1:                    # extra block at the end of this stage
                if attn_type == "attention":
                    side = image_size // (2 ** i)
                    blocks.append(ConvAttention(width, side * side, num_heads, attn_depth))
                else:
                    blocks.append(ConvControl(width))
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.ModuleList(stages)

        self.pool = nn.AdaptiveAvgPool2d(1)             # average each channel's map
        self.fc = nn.Linear(widths[-1], num_classes)

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.pool(x).flatten(1)    # B x C x 1 x 1  ->  B x C
        return self.fc(x)              # B x num_classes

def build_model(cfg):
    if cfg["model"] == "vit":
        return SmallViT(cfg["num_classes"], cfg["in_channels"], cfg["image_size"],
                        cfg["patch_size"], cfg["vit_dim"], cfg["vit_depth"], cfg["num_heads"])
    return SmallResNet(cfg["num_classes"], cfg["in_channels"], cfg["widths"],
                       cfg["blocks_per_stage"], cfg["attn_after"], cfg["attn_depth"],
                       cfg["num_heads"], cfg["image_size"], cfg["attn_type"])

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)