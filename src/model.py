import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal timestep embeddings implemented from scratch in PyTorch."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps):
        half_dim = (self.dim + 1) // 2
        exponent = -math.log(10000.0) * torch.arange(
            half_dim,
            device=timesteps.device,
            dtype=torch.float32,
        ) / max(half_dim - 1, 1)
        frequencies = torch.exp(exponent)
        angles = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
        embeddings = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
        return embeddings[:, :self.dim]


def _num_groups(channels):
    """Choose a GroupNorm group count that divides the channel count."""
    for groups in [8, 4, 2, 1]:
        if channels % groups == 0:
            return groups
    return 1


class ConvBlock(nn.Module):
    """Two-layer convolution block with optional timestep conditioning."""

    def __init__(self, in_channels, out_channels, time_emb_dim=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(_num_groups(out_channels), out_channels)
        self.act1 = nn.SiLU()
        self.time_proj = nn.Linear(time_emb_dim, out_channels) if time_emb_dim is not None else None
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(_num_groups(out_channels), out_channels)
        self.act2 = nn.SiLU()

    def forward(self, x, time_emb=None):
        h = self.conv1(x)
        h = self.norm1(h)

        if self.time_proj is not None and time_emb is not None:
            time_features = self.time_proj(time_emb).unsqueeze(-1).unsqueeze(-1)
            h = h + time_features

        h = self.act1(h)
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act2(h)
        return h


class DownBlock(nn.Module):
    """U-Net encoder block that returns the downsampled tensor and a skip tensor."""

    def __init__(self, in_channels, out_channels, time_emb_dim=None):
        super().__init__()
        self.block = ConvBlock(in_channels, out_channels, time_emb_dim)
        self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x, time_emb=None):
        skip = self.block(x, time_emb)
        x = self.downsample(skip)
        return x, skip


class UpBlock(nn.Module):
    """U-Net decoder block that upsamples, concatenates a skip tensor, and refines."""

    def __init__(self, in_channels, out_channels, time_emb_dim=None):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1)
        self.block = ConvBlock(in_channels * 2, out_channels, time_emb_dim)

    def forward(self, x, skip, time_emb=None):
        x = self.upsample(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
        x = torch.cat([x, skip], dim=1)
        return self.block(x, time_emb)


class SimpleUNet(nn.Module):
    """Small U-Net noise predictor for 32x32 CIFAR-10 diffusion experiments."""

    def __init__(self, image_channels=3, base_channels=64, time_emb_dim=256):
        super().__init__()
        self.init_conv = nn.Conv2d(image_channels, base_channels, kernel_size=3, padding=1)

        self.time_embedding = SinusoidalTimeEmbedding(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        self.down1 = DownBlock(base_channels, base_channels * 2, time_emb_dim)
        self.down2 = DownBlock(base_channels * 2, base_channels * 4, time_emb_dim)

        self.bottleneck1 = ConvBlock(base_channels * 4, base_channels * 4, time_emb_dim)
        self.bottleneck2 = ConvBlock(base_channels * 4, base_channels * 4, time_emb_dim)

        self.up1 = UpBlock(base_channels * 4, base_channels * 2, time_emb_dim)
        self.up2 = UpBlock(base_channels * 2, base_channels, time_emb_dim)

        self.final_conv = nn.Conv2d(base_channels, image_channels, kernel_size=1)

    def forward(self, x, timesteps):
        # Time embeddings condition the model on the current noise level.
        time_emb = self.time_mlp(self.time_embedding(timesteps))

        x = self.init_conv(x)
        x, skip1 = self.down1(x, time_emb)
        x, skip2 = self.down2(x, time_emb)

        x = self.bottleneck1(x, time_emb)
        x = self.bottleneck2(x, time_emb)

        x = self.up1(x, skip2, time_emb)
        x = self.up2(x, skip1, time_emb)

        # The model predicts epsilon_theta(x_t, t), not a denoised image.
        # Output shape matches input because the loss compares predicted noise with true noise.
        return self.final_conv(x)


def count_parameters(model):
    """Return the number of trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
