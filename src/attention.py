import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    """Multi-head self-attention. Input and output: B x N x C (N tokens, C dims)."""
    def __init__(self, dim, num_heads):
        super().__init__()
        assert dim % num_heads == 0, "channels must divide evenly into heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)   # learned layers that make Q, K, V
        self.proj = nn.Linear(dim, dim)      # mixes the heads back together

    def forward(self, x):                                   # B x N x C
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)                    # 3 x B x heads x N x head_dim
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = q @ k.transpose(-2, -1) / (self.head_dim ** 0.5)  # B x heads x N x N
        weights = scores.softmax(dim=-1)                    # each row sums to 1
        out = weights @ v                                   # B x heads x N x head_dim
        out = out.transpose(1, 2).reshape(B, N, C)          # join heads: B x N x C
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=2, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SelfAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim), nn.Dropout(dropout),
        )

    def forward(self, x):                      # shape in == shape out
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ConvAttention(nn.Module):
    """Feature map -> tokens -> Transformer blocks -> feature map (same shape)."""
    def __init__(self, channels, num_tokens, num_heads=4, depth=1):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, channels))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks = nn.Sequential(
            *[TransformerBlock(channels, num_heads) for _ in range(depth)]
        )

    def forward(self, x):                                   # B x C x H x W
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)               # B x (H*W) x C
        tokens = tokens + self.pos_embed
        tokens = self.blocks(tokens)
        return tokens.transpose(1, 2).reshape(B, C, H, W)   # back to a feature map

class ConvControl(nn.Module):
    """Parameter-matched control: one extra 3x3 conv block with a residual connection."""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()

    def forward(self, x):                       # shape in == shape out
        return self.relu(x + self.bn(self.conv(x)))