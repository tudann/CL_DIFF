"""
Transformer blocks for the v2 backbone.

Two block types are provided, both conditioned on the diffusion timestep through
adaLN-Zero (DiT, Peebles & Xie 2023):

* ``DiTBlock2d``   -- dense global self-attention over the whole feature map.
                     Cost is O(N^2) in the number of pixels, so it is only
                     affordable on the coarse levels of the UNet.
* ``FreqBlock2d``  -- frequency-domain attention (FSAS) plus a frequency-gated
                     feed-forward network (DFFN), after FFTformer
                     (Kong et al., CVPR 2023). Cost is O(N log N), so it can be
                     placed on the mid/high resolution levels where the
                     limited-angle streak artifacts live.

Both blocks zero-initialise their adaLN gates, so a freshly built block is an
exact identity. That is what makes ``load_v1_state_dict`` worthwhile: every
weight carried over from v1 keeps the behaviour it learned, and the new blocks
contribute nothing until training opens their gates. The transfer is not a
perfect function match, because v1's middle block used the old ``AttentionBlock``
and has no counterpart here.
"""

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .nn import checkpoint


class TimestepBlock(nn.Module):
    """A module whose forward takes the timestep embedding as a second argument."""

    def forward(self, x, emb):
        raise NotImplementedError


def _modulate_tokens(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def _modulate_map(x, shift, scale):
    return x * (1 + scale[..., None, None]) + shift[..., None, None]


class ChannelLayerNorm(nn.Module):
    """
    LayerNorm over the channel dimension of an ``[N, C, H, W]`` tensor.

    Statistics are always accumulated in float32 and the result is cast back to
    the input dtype, mirroring ``GroupNorm32`` in ``nn.py``. This keeps the block
    usable under autocast without the normalisation itself becoming a source of
    numerical noise.
    """

    def __init__(self, channels, eps=1e-5, affine=True):
        super().__init__()
        self.eps = eps
        if affine:
            self.weight = nn.Parameter(th.ones(channels))
            self.bias = nn.Parameter(th.zeros(channels))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        orig_dtype = x.dtype
        xf = x.float()
        mean = xf.mean(dim=1, keepdim=True)
        var = xf.var(dim=1, keepdim=True, unbiased=False)
        xf = (xf - mean) / th.sqrt(var + self.eps)
        if self.weight is not None:
            xf = xf * self.weight[None, :, None, None] + self.bias[None, :, None, None]
        return xf.to(orig_dtype)


class AdaLNZero(nn.Module):
    """
    Produce ``n_out`` groups of per-channel modulation parameters from the
    timestep embedding. The final projection is zero-initialised, so shifts and
    scales start at 0 and residual gates start closed.
    """

    def __init__(self, emb_channels, channels, n_out=6):
        super().__init__()
        self.n_out = n_out
        self.channels = channels
        self.proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, n_out * channels),
        )
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, emb):
        return self.proj(emb).chunk(self.n_out, dim=-1)


def build_2d_sincos_pos_emb(h, w, dim, device, dtype):
    """
    Fixed 2D sine-cosine positional embedding of shape ``[h * w, dim]``.

    Computed on the fly from ``(h, w)`` rather than stored as a parameter, so the
    same weights work at any feature-map size. This matters here because the
    sampling script may run on crops that differ from the training crop.
    """
    if dim % 4 != 0:
        raise ValueError(f"positional embedding dim must be divisible by 4, got {dim}")
    quarter = dim // 4
    omega = th.arange(quarter, device=device, dtype=th.float32) / quarter
    omega = 1.0 / (10000.0 ** omega)

    grid_y = th.arange(h, device=device, dtype=th.float32)
    grid_x = th.arange(w, device=device, dtype=th.float32)
    out_y = grid_y[:, None] * omega[None, :]
    out_x = grid_x[:, None] * omega[None, :]
    pos_y = th.cat([out_y.sin(), out_y.cos()], dim=1)
    pos_x = th.cat([out_x.sin(), out_x.cos()], dim=1)

    emb = th.cat(
        [
            pos_y[:, None, :].expand(h, w, dim // 2),
            pos_x[None, :, :].expand(h, w, dim // 2),
        ],
        dim=-1,
    )
    return emb.reshape(h * w, dim).to(dtype)


class DiTBlock2d(TimestepBlock):
    """
    Pre-norm transformer block over flattened spatial positions.

    Unlike a plain DiT block this consumes and returns an ``[N, C, H, W]``
    feature map, so it can be dropped into the UNet in place of an
    ``AttentionBlock`` while the surrounding convolutions keep providing the
    local inductive bias and the long skip connections keep carrying high
    frequencies.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        num_heads=4,
        num_head_channels=-1,
        mlp_ratio=4.0,
        dropout=0.0,
        use_checkpoint=False,
        use_pos_emb=True,
    ):
        super().__init__()
        if num_head_channels != -1:
            if channels % num_head_channels != 0:
                raise ValueError(
                    f"channels {channels} not divisible by num_head_channels "
                    f"{num_head_channels}"
                )
            self.num_heads = channels // num_head_channels
        else:
            if channels % num_heads != 0:
                raise ValueError(
                    f"channels {channels} not divisible by num_heads {num_heads}"
                )
            self.num_heads = num_heads

        self.channels = channels
        self.use_checkpoint = use_checkpoint
        self.use_pos_emb = use_pos_emb and channels % 4 == 0
        self.dropout = dropout

        # Only the adaLN gates are zero-initialised. Zeroing the output
        # projections as well would make both the gate gradient and the branch
        # gradient identically zero, leaving the block permanently dead.
        self.norm1 = nn.LayerNorm(channels, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(channels, channels * 3, bias=True)
        self.proj = nn.Linear(channels, channels, bias=True)

        self.norm2 = nn.LayerNorm(channels, elementwise_affine=False, eps=1e-6)
        hidden = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Dropout(p=dropout),
            nn.Linear(hidden, channels, bias=True),
        )

        self.ada = AdaLNZero(emb_channels, channels, n_out=6)
        self._pos_cache = {}

    def _pos_emb(self, h, w, device, dtype):
        key = (h, w, device, dtype)
        if key not in self._pos_cache:
            self._pos_cache[key] = build_2d_sincos_pos_emb(
                h, w, self.channels, device, dtype
            )
        return self._pos_cache[key]

    def forward(self, x, emb):
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _attn(self, tokens):
        n, seq, c = tokens.shape
        qkv = self.qkv(tokens)
        qkv = qkv.reshape(n, seq, 3, self.num_heads, c // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(n, seq, c)
        return self.proj(out)

    def _forward(self, x, emb):
        n, c, h, w = x.shape
        shift1, scale1, gate1, shift2, scale2, gate2 = (
            m.to(x.dtype) for m in self.ada(emb)
        )

        tokens = x.reshape(n, c, h * w).transpose(1, 2)

        # The positional embedding is injected into the attention branch only,
        # never into the residual stream, so a closed gate leaves the block an
        # exact identity.
        normed = _modulate_tokens(self.norm1(tokens), shift1, scale1)
        if self.use_pos_emb:
            normed = normed + self._pos_emb(h, w, x.device, x.dtype)[None]
        tokens = tokens + gate1.unsqueeze(1) * self._attn(normed)

        normed = _modulate_tokens(self.norm2(tokens), shift2, scale2)
        tokens = tokens + gate2.unsqueeze(1) * self.mlp(normed)

        return tokens.transpose(1, 2).reshape(n, c, h, w)


class FSAS(nn.Module):
    """
    Frequency-domain self-attention.

    Correlation between q and k is computed as an element-wise product of their
    patch-wise 2D FFTs, which is the convolution theorem applied in reverse: one
    element-wise product in frequency space stands in for the full spatial
    correlation. Cost is O(N log N) instead of the O(N^2) of dot-product
    attention, which is what makes it usable at 96x96 and above.

    Adapted from FFTformer (https://github.com/kkkls/FFTformer). Two changes
    relative to the copy in ``FFT_Transformer.py``: the per-forward
    ``torch.cuda.empty_cache()`` calls are gone (they synchronise the device and
    drop the caching allocator on every step), and inputs whose spatial size is
    not a multiple of ``patch_size`` are reflect-padded instead of crashing.
    """

    def __init__(self, channels, patch_size=8, bias=False):
        super().__init__()
        self.patch_size = patch_size
        self.to_hidden = nn.Conv2d(channels, channels * 6, 1, bias=bias)
        self.to_hidden_dw = nn.Conv2d(
            channels * 6, channels * 6, 3, padding=1, groups=channels * 6, bias=bias
        )
        self.norm = ChannelLayerNorm(channels * 2)
        self.project_out = nn.Conv2d(channels * 2, channels, 1, bias=bias)

    def _patch_fft_product(self, q, k):
        p = self.patch_size
        n, c, h, w = q.shape
        q = q.reshape(n, c, h // p, p, w // p, p).permute(0, 1, 2, 4, 3, 5)
        k = k.reshape(n, c, h // p, p, w // p, p).permute(0, 1, 2, 4, 3, 5)
        # FFT is always taken in float32: rfft2 has no half-precision kernel on
        # most backends, and the product of two spectra is numerically touchy.
        q_fft = th.fft.rfft2(q.float())
        k_fft = th.fft.rfft2(k.float())
        out = th.fft.irfft2(q_fft * k_fft, s=(p, p))
        out = out.permute(0, 1, 2, 4, 3, 5).reshape(n, c, h, w)
        return out.to(q.dtype)

    def forward(self, x):
        p = self.patch_size
        _, _, h0, w0 = x.shape
        pad_h = (p - h0 % p) % p
        pad_w = (p - w0 % p) % p
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        hidden = self.to_hidden_dw(self.to_hidden(x))
        q, k, v = hidden.chunk(3, dim=1)
        out = self.norm(self._patch_fft_product(q, k))
        out = self.project_out(v * out)

        if pad_h or pad_w:
            out = out[..., :h0, :w0]
        return out


class DFFN(nn.Module):
    """
    Frequency-gated feed-forward network from FFTformer.

    A learnable per-frequency mask is applied inside each patch before the
    depthwise convolution and GELU gating, which lets the block suppress or
    amplify specific spatial frequencies directly. That is a good match for
    limited-angle reconstruction, where the corruption is a missing wedge in
    Fourier space rather than an additive spatial pattern.
    """

    def __init__(self, channels, expansion=2.66, patch_size=8, bias=False):
        super().__init__()
        hidden = int(channels * expansion)
        self.patch_size = patch_size
        self.project_in = nn.Conv2d(channels, hidden * 2, 1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden * 2, hidden * 2, 3, padding=1, groups=hidden * 2, bias=bias
        )
        self.fft_mask = nn.Parameter(
            th.ones(hidden * 2, 1, 1, patch_size, patch_size // 2 + 1)
        )
        self.project_out = nn.Conv2d(hidden, channels, 1, bias=bias)

    def forward(self, x):
        p = self.patch_size
        _, _, h0, w0 = x.shape
        pad_h = (p - h0 % p) % p
        pad_w = (p - w0 % p) % p
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        x = self.project_in(x)
        n, c, h, w = x.shape
        patches = x.reshape(n, c, h // p, p, w // p, p).permute(0, 1, 2, 4, 3, 5)
        spec = th.fft.rfft2(patches.float()) * self.fft_mask.unsqueeze(0)
        patches = th.fft.irfft2(spec, s=(p, p)).to(x.dtype)
        x = patches.permute(0, 1, 2, 4, 3, 5).reshape(n, c, h, w)

        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        out = self.project_out(F.gelu(x1) * x2)

        if pad_h or pad_w:
            out = out[..., :h0, :w0]
        return out


class FreqBlock2d(TimestepBlock):
    """
    FSAS + DFFN with adaLN-Zero timestep conditioning.

    Note that the upstream ``TransformerBlock`` in ``FFT_Transformer.py`` writes
    ``x = self.attn(self.norm1(x))``, dropping the residual around the attention
    branch. That is restored here, and both branches are gated so the block
    starts as an identity.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        expansion=2.66,
        patch_size=8,
        bias=False,
        use_checkpoint=False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.norm1 = ChannelLayerNorm(channels, affine=False)
        self.attn = FSAS(channels, patch_size=patch_size, bias=bias)
        self.norm2 = ChannelLayerNorm(channels, affine=False)
        self.ffn = DFFN(channels, expansion=expansion, patch_size=patch_size, bias=bias)
        self.ada = AdaLNZero(emb_channels, channels, n_out=6)

    def forward(self, x, emb):
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
        shift1, scale1, gate1, shift2, scale2, gate2 = (
            m.to(x.dtype) for m in self.ada(emb)
        )
        h = _modulate_map(self.norm1(x), shift1, scale1)
        x = x + gate1[..., None, None] * self.attn(h)
        h = _modulate_map(self.norm2(x), shift2, scale2)
        x = x + gate2[..., None, None] * self.ffn(h)
        return x
