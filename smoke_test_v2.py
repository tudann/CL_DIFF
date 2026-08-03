"""
Structural checks for the v2 backbone. Runs on CPU in a few seconds.

    python smoke_test_v2.py

Covers four things:

1. every arch produces the (B, 2, H, W) output that ``gaussian_diffusion``
   asserts on;
2. the adaLN-Zero gates really do make a freshly built transformer block an
   identity, which is what allows warm-starting from a conv-only checkpoint;
3. those gates receive non-zero gradients, i.e. the blocks are not dead
   (zero-initialising both the gate and the branch output projection would make
   both gradients vanish);
4. the resolution spec resolves to the levels we intend, including a regression
   guard on the v1 semantics that silently matched nothing.
"""

import sys
import traceback

import torch as th

from guided_diffusion.script_util import CL_IMG_create_model_and_diffusion
from guided_diffusion.transformer_blocks import DiTBlock2d, FreqBlock2d
from guided_diffusion.transformer_blocks import TimestepBlock as TransformerTimestepBlock
from guided_diffusion.unet_v2 import load_v1_state_dict, parse_resolutions

DEVICE = th.device("cpu")
IMAGE_SIZE = 64
BATCH = 2
COND_CH = 3

SMALL = dict(
    image_size=IMAGE_SIZE,
    num_channels=32,
    num_res_blocks=1,
    channel_mult="1,1,2",
    condition_channels=COND_CH,
    attention_resolutions="16",
    freq_resolutions="32",
    num_heads=4,
    num_head_channels=-1,
    num_heads_upsample=-1,
    dropout=0.0,
    use_checkpoint=False,
    use_scale_shift_norm=True,
    resblock_updown=False,
    use_fp16=False,
    use_new_attention_order=False,
    learn_sigma=True,
    diffusion_steps=50,
    noise_schedule="linear",
    timestep_respacing="",
    use_kl=False,
    predict_xstart=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
)


class _Passthrough(TransformerTimestepBlock):
    def forward(self, x, emb):
        return x


def build(arch, **overrides):
    cfg = dict(SMALL)
    cfg["arch"] = arch
    if arch == "v1":
        # v1 only understands the legacy spec.
        cfg["attention_resolutions"] = "16,8"
    cfg.update(overrides)
    return CL_IMG_create_model_and_diffusion(device=DEVICE, **cfg)


def sample_inputs():
    g = th.Generator().manual_seed(0)
    x = th.randn(BATCH, 1, IMAGE_SIZE, IMAGE_SIZE, generator=g)
    cond = th.randn(BATCH, COND_CH, IMAGE_SIZE, IMAGE_SIZE, generator=g)
    t = th.randint(0, 50, (BATCH,), generator=g)
    return x, cond, t


def test_forward_shapes():
    for arch in ("v1", "v2a", "v2b"):
        model, _ = build(arch)
        x, cond, t = sample_inputs()
        with th.no_grad():
            out = model([x, cond], t)
        expected = (BATCH, 2, IMAGE_SIZE, IMAGE_SIZE)
        assert out.shape == expected, f"{arch}: got {tuple(out.shape)}, want {expected}"
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  {arch:4s} output {tuple(out.shape)}  params {n_params / 1e6:.3f} M")


def test_transformer_blocks_present():
    model, _ = build("v2b")
    kinds = {}
    for container in [*model.input_blocks, model.middle_block, *model.output_blocks]:
        for layer in container:
            if isinstance(layer, (DiTBlock2d, FreqBlock2d)):
                kinds[type(layer).__name__] = kinds.get(type(layer).__name__, 0) + 1
    assert "DiTBlock2d" in kinds, "no dense attention block was inserted"
    assert "FreqBlock2d" in kinds, "no frequency block was inserted"
    print(f"  inserted blocks: {kinds}")


def test_adaln_zero_is_identity():
    """A freshly built v2b must match the same net with its transformer blocks bypassed."""
    th.manual_seed(1234)
    model, _ = build("v2b")
    model.eval()
    x, cond, t = sample_inputs()
    with th.no_grad():
        ref = model([x, cond], t)

    replaced = 0
    for container in [*model.input_blocks, model.middle_block, *model.output_blocks]:
        for i, layer in enumerate(container):
            if isinstance(layer, (DiTBlock2d, FreqBlock2d)):
                container[i] = _Passthrough()
                replaced += 1
    assert replaced > 0

    with th.no_grad():
        bypassed = model([x, cond], t)

    delta = (ref - bypassed).abs().max().item()
    assert delta < 1e-5, f"blocks are not identity at init, max |delta| = {delta:.3e}"
    print(f"  bypassed {replaced} blocks, max |delta| = {delta:.3e}")


def test_gates_receive_gradient():
    """
    The adaLN gate must not be stuck at zero gradient.

    The final output conv is ``zero_module``-initialised, which makes the
    gradient of every parameter in the network body exactly zero on step 0. That
    is intended in guided-diffusion, but it means a gradient check at init tells
    us nothing, so we give that conv a small non-zero weight first -- the state
    it is in from step 1 onward.
    """
    th.manual_seed(7)
    model, diffusion = build("v2b")
    th.nn.init.normal_(model.out[-1].weight, std=0.01)
    x, cond, t = sample_inputs()

    losses = diffusion.training_losses(model, x, cond, t, 0, device=DEVICE)
    loss = losses["loss"].mean()
    assert th.isfinite(loss), f"loss is not finite: {loss}"
    loss.backward()

    checked = 0
    for name, param in model.named_parameters():
        if "ada.proj" not in name or not name.endswith(".weight"):
            continue
        checked += 1
        assert param.grad is not None, f"{name}: no gradient"
        gnorm = param.grad.abs().max().item()
        assert gnorm > 0, f"{name}: gate gradient is exactly zero (dead block)"
    assert checked > 0, "no adaLN gate parameters found"
    print(f"  loss {loss.item():.4f}, {checked} adaLN gates all received gradient")


LEGACY_ATTN_SUFFIXES = (
    "norm.weight", "norm.bias", "qkv.weight", "qkv.bias",
    "proj_out.weight", "proj_out.bias",
)


def test_warm_start_from_v1():
    """A v1 checkpoint must transfer into v2 apart from the legacy attention blocks."""
    # Scenario 1 reproduces the production case: at 768 the v1 spec matches no
    # level, so v1 has no attention outside the middle block and its decoder
    # Upsample sits at index 1, where v2 has it at index 2. At image_size=64 the
    # equivalent no-match spec is "8", since 64 // 8 = 8 and three levels only
    # reach ds = 4.
    v1_plain, _ = build("v1", attention_resolutions="8")
    v2, _ = build("v2a")
    sd_plain = v1_plain.state_dict()
    loaded, skipped = load_v1_state_dict(v2, sd_plain, verbose=False)
    assert loaded, "nothing transferred"
    assert all(k.startswith("middle_block.1.") for k in skipped), (
        f"only the middle attention should be dropped here, got {skipped}"
    )

    sd2 = v2.state_dict()
    moved = 0
    for key, value in sd_plain.items():
        if not key.startswith("output_blocks.") or ".1.conv.weight" not in key:
            continue
        target = key.replace(".1.conv.weight", ".2.conv.weight")
        if target in sd2:
            assert th.equal(sd2[target], value), f"{key} did not land in {target}"
            moved += 1
    assert moved > 0, "no upsample conv was remapped"

    # Scenario 2: a v1 spec that does hit a level produces legacy AttentionBlock
    # tensors with no v2 counterpart. They must be dropped rather than partially
    # matched into the new transformer block.
    v1_attn, _ = build("v1", attention_resolutions="16")
    v2_again, _ = build("v2a")
    _, skipped_attn = load_v1_state_dict(
        v2_again, v1_attn.state_dict(), verbose=False
    )
    unexpected = [k for k in skipped_attn if not k.endswith(LEGACY_ATTN_SUFFIXES)]
    assert not unexpected, f"unexpected tensors failed to transfer: {unexpected}"
    assert len(skipped_attn) > len(skipped), (
        "the extra legacy attention tensors were not detected"
    )

    print(
        f"  no-attn v1: {len(loaded)} tensors in, {moved} upsample conv(s) remapped, "
        f"{len(skipped)} dropped; with-attn v1: {len(skipped_attn)} dropped"
    )


def test_resolution_parsing():
    # 768 with the v2 spec: feature maps 24 and 12 sit at ds 32 and 64.
    assert parse_resolutions("24,12", 768) == {32, 64}
    assert parse_resolutions("96,48", 768) == {8, 16}
    assert parse_resolutions("", 768) == set()
    assert parse_resolutions("none", 768) == set()

    # Regression guard: the v1 formula produced downsample factors that a
    # power-of-two ds can never equal, so no level ever got attention.
    legacy = {768 // int(r) for r in "16,8".split(",")}
    reachable = {2 ** i for i in range(7)}
    assert legacy.isdisjoint(reachable), (
        "the v1 attention spec unexpectedly matches a level; "
        "the regression guard needs updating"
    )
    print(f"  v1 legacy spec resolved to {sorted(legacy)}, reachable ds {sorted(reachable)} -> no overlap")


def report_production_config(run_forward=False):
    """Build the real 768 config for each arch and report size. Opt-in: slow on CPU."""
    cfg = dict(SMALL)
    cfg.update(
        image_size=768,
        num_channels=64,
        num_res_blocks=2,
        channel_mult="",
        attention_resolutions="24,12",
        freq_resolutions="96,48",
        diffusion_steps=1000,
    )
    print("\nproduction config (768, num_channels=64, channel_mult=(0.5,1,1,2,2,4,4)):")
    for arch in ("v1", "v2a", "v2b"):
        c = dict(cfg)
        c["arch"] = arch
        if arch == "v1":
            c["attention_resolutions"] = "16,8"
        model, _ = CL_IMG_create_model_and_diffusion(device=DEVICE, **c)
        n = sum(p.numel() for p in model.parameters())
        line = f"  {arch:4s} params {n / 1e6:7.2f} M"
        if run_forward:
            x = th.randn(1, 1, 768, 768)
            cond = th.randn(1, COND_CH, 768, 768)
            with th.no_grad():
                out = model([x, cond], th.tensor([500]))
            line += f"  forward out {tuple(out.shape)}"
        print(line)


def report_warm_start_overlap():
    """
    How much of a v1 checkpoint transfers into v2a at the production config?

    Relevant because the adaLN-Zero gates make the new blocks identities at init,
    so any weight that does transfer keeps its learned behaviour.
    """
    cfg = dict(SMALL)
    cfg.update(
        image_size=768,
        num_channels=64,
        num_res_blocks=2,
        channel_mult="",
        freq_resolutions="96,48",
        diffusion_steps=1000,
    )
    v1, _ = CL_IMG_create_model_and_diffusion(
        device=DEVICE, **{**cfg, "arch": "v1", "attention_resolutions": "16,8"}
    )
    v2, _ = CL_IMG_create_model_and_diffusion(
        device=DEVICE, **{**cfg, "arch": "v2a", "attention_resolutions": "24,12"}
    )

    sd1 = v1.state_dict()
    print("\nv1 -> v2a weight transfer:")
    loaded, skipped = load_v1_state_dict(v2, sd1)
    n_loaded = sum(sd1[k].numel() for k in loaded if k in sd1)
    n_total_v1 = sum(t.numel() for t in sd1.values())
    print(f"  covers {n_loaded / n_total_v1:.1%} of the v1 parameters")
    for k in skipped:
        print(f"    dropped {k}  {tuple(sd1[k].shape)}")


def main():
    if "--production" in sys.argv:
        report_production_config(run_forward="--forward" in sys.argv)
        return 0
    if "--warm-start" in sys.argv:
        report_warm_start_overlap()
        return 0

    tests = [
        ("forward shapes", test_forward_shapes),
        ("transformer blocks inserted", test_transformer_blocks_present),
        ("adaLN-Zero identity at init", test_adaln_zero_is_identity),
        ("adaLN gates get gradient", test_gates_receive_gradient),
        ("warm start from v1", test_warm_start_from_v1),
        ("resolution parsing", test_resolution_parsing),
    ]
    failures = 0
    for name, fn in tests:
        print(f"[ RUN ] {name}")
        try:
            fn()
        except Exception:
            failures += 1
            print(f"[ FAIL] {name}")
            traceback.print_exc()
        else:
            print(f"[ OK  ] {name}")
    print()
    if failures:
        print(f"{failures} of {len(tests)} checks failed")
        return 1
    print(f"all {len(tests)} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
