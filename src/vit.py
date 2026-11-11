import torch
import torch.nn as nn
from attention import TransformerBlock


class SmallViT(nn.Module):
    def __init__(self, num_classes=10, in_channels=3, image_size=32,
                 patch_size=4, dim=256, depth=5, num_heads=4):
        super().__init__()
        assert image_size % patch_size == 0, "image size must divide into patches"
        num_tokens = (image_size // patch_size) ** 2
        # cutting into patches + linear projection, done as one strided convolution
        self.patch_embed = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks = nn.Sequential(*[TransformerBlock(dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):                        # B x 3 x 32 x 32
        x = self.patch_embed(x)                  # B x dim x 8 x 8
        x = x.flatten(2).transpose(1, 2)         # B x 64 x dim
        x = x + self.pos_embed
        x = self.blocks(x)                       # B x 64 x dim
        x = self.norm(x).mean(dim=1)             # average over tokens: B x dim
        return self.fc(x)                        # B x num_classes