"""Interlayer Leakage Decoupling (ILD).

Rewrites a 2.5D CL condition into an explicit current / leakage / common
split so the U-Net does not have to discover out-of-plane ghosts from raw
neighbor concatenation.  A residual spatial gate then mixes those semantic
channels; a closed gate leaves the decomposed condition unchanged.

No tilt angle or projection data is used.  Neighbor slices already carry
the leaked ghosts after FDK reconstruction.
"""
import torch as th
import torch.nn as nn
import torch.nn.functional as F


class InterlayerLeakageDecoupling(nn.Module):
    """Decompose [..., z-1, z, z+1, ...] into current, leakage, and common.

    For the default 3-slice stack the output channels are
    ``[CL_z, CL_z - 0.5(CL_{z-1}+CL_{z+1}), 0.5(CL_{z-1}+CL_{z+1})]``.
    Outer slices are kept when ``condition_channels > 3``.  Channel count
    is unchanged, so the U-Net stem is identical to the baseline.
    """

    def __init__(self, channels):
        super().__init__()
        if channels < 3 or channels % 2 == 0:
            raise ValueError(
                "ILD requires odd condition_channels >= 3, "
                f"got {channels}."
            )
        self.channels = channels
        self.center = channels // 2

        self.mix = nn.Conv2d(channels, channels, 1, bias=False)
        nn.init.zeros_(self.mix.weight)
        self.norm = nn.GroupNorm(min(32, channels), channels)
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def _decompose(self, cond):
        center = self.center
        current = cond[:, center : center + 1]
        common = 0.5 * (cond[:, center - 1 : center] + cond[:, center + 1 : center + 2])
        leak = current - common

        parts = []
        if center > 1:
            parts.append(cond[:, : center - 1])
        parts.extend([current, leak, common])
        if center + 2 < self.channels:
            parts.append(cond[:, center + 2 :])
        return th.cat(parts, dim=1)

    def forward(self, cond):
        if cond.shape[1] != self.channels:
            raise ValueError(
                f"ILD expected {self.channels} condition channels, "
                f"got {cond.shape[1]}."
            )
        decomposed = self._decompose(cond)
        correction = F.silu(self.norm(self.mix(decomposed)))
        correction = self.gate(correction) * correction
        return decomposed + correction
