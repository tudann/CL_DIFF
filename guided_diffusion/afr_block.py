"""Anisotropic Feature Representation (AFR).

Models the direction-dependent blur of computed laminography with a pair of
asymmetric convolutions and a spatial gate.  The block is residual, so a
closed gate leaves the input unchanged.
"""
import torch as th
import torch.nn as nn
import torch.nn.functional as F


class AnisotropicFeatureRepresentation(nn.Module):
    """Full-resolution directional residual correction.

    Horizontal 1xK and vertical Kx1 depthwise convolutions capture anisotropic
    streaking.  A 1x1 mixer fuses the two axes, and a sigmoid gate scales the
    correction per location before it is added back to the input.
    """

    def __init__(self, channels, kernel_size=7):
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError(
                f"afr_kernel_size must be a positive odd integer, got {kernel_size}."
            )
        pad = kernel_size // 2
        self.channels = channels
        self.kernel_size = kernel_size

        self.conv_h = nn.Conv2d(
            channels, channels, (1, kernel_size),
            padding=(0, pad), groups=channels, bias=False,
        )
        self.conv_v = nn.Conv2d(
            channels, channels, (kernel_size, 1),
            padding=(pad, 0), groups=channels, bias=False,
        )
        self.mix = nn.Conv2d(channels * 2, channels, 1, bias=False)
        self.norm = nn.GroupNorm(min(32, channels), channels)
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        horiz = self.conv_h(x)
        vert = self.conv_v(x)
        correction = self.mix(th.cat([horiz, vert], dim=1))
        correction = F.silu(self.norm(correction))
        correction = self.gate(correction) * correction
        return x + correction
