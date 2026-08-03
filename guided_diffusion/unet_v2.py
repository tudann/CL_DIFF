"""
v2 backbone: the v1 UNet skeleton with transformer blocks on selected levels.

The multi-scale encoder/decoder and the long skip connections are kept
deliberately. The task is conditional restoration -- the output slice is
spatially aligned with the conditioning slices -- so the skips are the cheapest
possible path for high-frequency content, and a patchify-based isotropic
transformer would have to relearn that path through a bottleneck.

Two things changed relative to ``unet.py``:

1. ``attention_resolutions`` and ``freq_resolutions`` are now interpreted as
   feature-map sizes and compared against ``image_size // ds``. In v1 the
   comparison was ``ds in {image_size // res}``, and since ``ds`` only ever takes
   powers of two, a 768 image with ``"16,8"`` produced ``{48, 96}`` and matched
   nothing -- the encoder and decoder ended up with no attention at all.
2. The matched levels get ``DiTBlock2d`` (dense global attention, coarse levels)
   or ``FreqBlock2d`` (frequency-domain attention, mid levels) instead of the
   plain ``AttentionBlock``.
"""

import torch as th
import torch.nn as nn

from .fp16_util import convert_module_to_f16, convert_module_to_f32
from .nn import conv_nd, linear, normalization, timestep_embedding, zero_module
from .transformer_blocks import DiTBlock2d, FreqBlock2d
from .transformer_blocks import TimestepBlock as TransformerTimestepBlock
from .unet import Downsample, ResBlock, Upsample
from .unet import TimestepBlock as ConvTimestepBlock


class TimestepEmbedSequential(nn.Sequential):
    """
    Sequential container that forwards the timestep embedding to any child that
    wants it. Unlike the version in ``unet.py`` this recognises both the
    convolutional ``TimestepBlock`` and the transformer one.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, (ConvTimestepBlock, TransformerTimestepBlock)):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


def parse_resolutions(spec, image_size):
    """
    Turn ``"96,48"`` into the set of downsample factors that produce those
    feature-map sizes. Empty or ``"none"`` yields an empty set.
    """
    if spec is None:
        return set()
    if isinstance(spec, (set, frozenset)):
        return set(spec)
    if isinstance(spec, (list, tuple)):
        items = list(spec)
    else:
        spec = str(spec).strip()
        if spec == "" or spec.lower() == "none":
            return set()
        items = [s for s in spec.split(",") if s.strip() != ""]

    factors = set()
    for item in items:
        res = int(item)
        if res <= 0 or image_size % res != 0:
            raise ValueError(
                f"resolution {res} does not evenly divide image_size {image_size}"
            )
        factors.add(image_size // res)
    return factors


def load_v1_state_dict(model, state_dict, verbose=True):
    """
    Warm-start a v2 backbone from a v1 checkpoint.

    Worth doing: the adaLN gates start closed, so every transferred weight keeps
    exactly the behaviour it learned in v1 and the new blocks only add to it from
    zero.

    Two name families need care. Inserting a transformer block into a decoder
    container pushes the ``Upsample`` that follows it from index 1 to index 2, so
    those convolutions are remapped by position. The v1 middle block used the old
    ``AttentionBlock``, which has no counterpart here; those tensors are dropped
    and the replacement ``DiTBlock2d`` starts from scratch (harmlessly, since its
    gate is closed).

    :return: ``(loaded, skipped)`` lists of source key names.
    """
    remapped = dict(state_dict)

    for j, container in enumerate(model.output_blocks):
        ups = [i for i, m in enumerate(container) if isinstance(m, Upsample)]
        if not ups or ups[0] == 1:
            continue
        target_idx = ups[0]
        for suffix in ("conv.weight", "conv.bias"):
            src = f"output_blocks.{j}.1.{suffix}"
            dst = f"output_blocks.{j}.{target_idx}.{suffix}"
            if src in remapped and dst not in remapped:
                remapped[dst] = remapped.pop(src)

    # A v1 AttentionBlock and a v2 DiTBlock2d both expose a ``qkv`` submodule
    # whose bias has the same shape, so a plain name-and-shape match would quietly
    # copy a legacy attention bias into a transformer block. Exclude the new
    # blocks from the transfer entirely.
    transformer_prefixes = tuple(
        f"{name}."
        for name, module in model.named_modules()
        if isinstance(module, (DiTBlock2d, FreqBlock2d))
    )

    own = model.state_dict()
    loaded, skipped = [], []
    to_load = {}
    for key, value in remapped.items():
        if transformer_prefixes and key.startswith(transformer_prefixes):
            skipped.append(key)
        elif key in own and own[key].shape == value.shape:
            to_load[key] = value
            loaded.append(key)
        else:
            skipped.append(key)

    missing = [k for k in own if k not in to_load]
    model.load_state_dict(to_load, strict=False)

    if verbose:
        n_loaded = sum(own[k].numel() for k in loaded)
        n_total = sum(t.numel() for t in own.values())
        print(
            f"[warm start] loaded {len(loaded)} tensors covering "
            f"{n_loaded / n_total:.1%} of the v2 parameters"
        )
        if skipped:
            print(f"[warm start] dropped {len(skipped)} v1 tensors: {skipped}")
        print(f"[warm start] {len(missing)} v2 tensors left at their init values")

    return loaded, skipped


class CL_IMG_Transformer_UNet(nn.Module):
    """
    Conditional diffusion backbone for limited-angle CL reconstruction.

    Input is a list ``[x_t, limited_img]``: the noised target slice and the
    2.5D stack of conditioning CL-FDK slices. They are concatenated on the
    channel axis, so the conditioning stays pixel-aligned with the target at
    full resolution rather than being squeezed through a patch embedding.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        condition_channels,
        out_channels,
        model_channels,
        num_res_blocks,
        attention_resolutions=(),
        freq_resolutions=(),
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=4,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        transformer_depth=1,
        freq_depth=1,
        mlp_ratio=4.0,
        freq_expansion=2.66,
        freq_patch_size=8,
        middle_block_attn=True,
        w=1.50,
        cond_prob=0.2,
        weighted_condition=False,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.image_size = image_size
        self.in_channels = in_channels
        self.condition_channels = condition_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.use_fp16 = use_fp16
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        self.attn_ds = parse_resolutions(attention_resolutions, image_size)
        self.freq_ds = parse_resolutions(freq_resolutions, image_size)
        overlap = self.attn_ds & self.freq_ds
        if overlap:
            raise ValueError(
                "a level cannot host both dense and frequency attention; "
                f"conflicting downsample factors: {sorted(overlap)}"
            )

        time_embed_dim = model_channels * 4
        self.cond_prob = th.tensor(cond_prob)
        self.weighted_condition = weighted_condition
        self.w = w

        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        ch = input_ch = int(channel_mult[0] * model_channels)
        model_input_channels = in_channels + condition_channels

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, model_input_channels, ch, 3, padding=1)
                )
            ]
        )
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                layers.extend(
                    self._make_transformer_layers(
                        ch, time_embed_dim, ds, num_heads, transformer_depth,
                        freq_depth, mlp_ratio, freq_expansion, freq_patch_size,
                        dropout, use_checkpoint,
                    )
                )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)

            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2

        middle_layers = [
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            )
        ]
        if middle_block_attn:
            middle_layers.append(
                DiTBlock2d(
                    ch,
                    time_embed_dim,
                    num_heads=num_heads,
                    num_head_channels=num_head_channels,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    use_checkpoint=use_checkpoint,
                )
            )
        middle_layers.append(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            )
        )
        self.middle_block = TimestepEmbedSequential(*middle_layers)

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                layers.extend(
                    self._make_transformer_layers(
                        ch, time_embed_dim, ds, num_heads_upsample,
                        transformer_depth, freq_depth, mlp_ratio, freq_expansion,
                        freq_patch_size, dropout, use_checkpoint,
                    )
                )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, ch, out_channels, 3, padding=1)),
        )

    def _make_transformer_layers(
        self,
        ch,
        time_embed_dim,
        ds,
        num_heads,
        transformer_depth,
        freq_depth,
        mlp_ratio,
        freq_expansion,
        freq_patch_size,
        dropout,
        use_checkpoint,
    ):
        layers = []
        if ds in self.attn_ds:
            for _ in range(transformer_depth):
                layers.append(
                    DiTBlock2d(
                        ch,
                        time_embed_dim,
                        num_heads=num_heads,
                        num_head_channels=self.num_head_channels,
                        mlp_ratio=mlp_ratio,
                        dropout=dropout,
                        use_checkpoint=use_checkpoint,
                    )
                )
        elif ds in self.freq_ds:
            for _ in range(freq_depth):
                layers.append(
                    FreqBlock2d(
                        ch,
                        time_embed_dim,
                        expansion=freq_expansion,
                        patch_size=freq_patch_size,
                        use_checkpoint=use_checkpoint,
                    )
                )
        return layers

    def convert_to_fp16(self):
        """
        Cast the convolutional torso to fp16.

        ``convert_module_to_f16`` only touches conv weights, which would leave
        the transformer blocks' linear layers and norms in fp32 and produce dtype
        mismatches mid-graph. Use ``--use_bf16`` instead, which wraps the forward
        in autocast and handles every layer type.
        """
        if self.attn_ds or self.freq_ds:
            raise RuntimeError(
                "the v2 transformer backbone does not support --use_fp16; "
                "use --use_bf16 True instead"
            )
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def _forward(self, ipt, timesteps, weighted_condition):
        x, limited_img = ipt

        c_ = (
            1.0
            if weighted_condition
            else th.bernoulli(self.cond_prob).to(x.device).type(self.dtype)
        )

        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        h = th.cat([x.type(self.dtype), c_ * limited_img.type(self.dtype)], dim=1)

        hs = []
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)

        h = self.middle_block(h, emb)

        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.type(x.dtype)

        return self.out(h)

    def forward(self, ipt, timesteps, **kwargs):
        if self.weighted_condition:
            return self.w * self._forward(ipt, timesteps, weighted_condition=True) + (
                1 - self.w
            ) * self._forward(ipt, timesteps, weighted_condition=False)
        return self._forward(ipt, timesteps, weighted_condition=True)
