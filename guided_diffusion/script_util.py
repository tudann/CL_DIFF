import argparse
import os
import sys

try:
    import yaml
except ImportError:  # pragma: no cover - only needed when using a local YAML file
    yaml = None

from guided_diffusion import gaussian_diffusion as gd
from .respace import SpacedDiffusion, space_timesteps
from .unet import CL_IMG_Model_test

def CL_IMG_create_model_and_diffusion(
    image_size,
    learn_sigma,
    num_channels,
    num_res_blocks,
    channel_mult,
    num_heads,
    num_head_channels,
    num_heads_upsample,
    attention_resolutions,
    dropout,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    use_kl,
    predict_xstart,
    rescale_timesteps,
    rescale_learned_sigmas,
    use_checkpoint,
    use_scale_shift_norm,
    resblock_updown,
    use_fp16,
    use_new_attention_order,
    device,
    condition_channels=1,
):
    model = create_CL_IMG_model(
        image_size,
        num_channels,
        num_res_blocks,
        condition_channels=condition_channels,
        channel_mult=channel_mult,
        use_checkpoint=use_checkpoint,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
        resblock_updown=resblock_updown,
        use_fp16=use_fp16,
        use_new_attention_order=use_new_attention_order,
    )

    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
        device=device,
    )
    return model, diffusion

# 创建模型
def create_CL_IMG_model(
    image_size,
    num_channels,
    num_res_blocks,
    condition_channels=1,
    channel_mult="",
    use_checkpoint=False,
    attention_resolutions="16, 8",
    num_heads=1,
    num_head_channels=-1,
    num_heads_upsample=-1,
    use_scale_shift_norm=False,
    dropout=0,
    resblock_updown=False,
    use_fp16=False,
    use_new_attention_order=False,
   
):
###########  channel_mult模型每个 downsample 阶段的 channel 数 #########################################################################################
    if channel_mult == "":
        if image_size == 1024:
            channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
        elif image_size == 768:
            channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
        elif image_size == 256:
            channel_mult = (0.5, 1, 2, 4, 8)
        elif image_size == 512:
            channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
        else:
            raise ValueError(f" CT_IMG_model unsupported image size: {image_size}")
    else:
        channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(","))
    print("channel_mult: ", channel_mult)

############当特征图的分辨率是原图的 1/16 和 1/8 时，加入 attention 层#########################################################################
    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    return CL_IMG_Model_test(
        image_size=image_size,
        in_channels=1,
        condition_channels=condition_channels,
        out_channels=2,

        model_channels=num_channels,
        num_res_blocks=num_res_blocks,

        attention_resolutions=tuple(attention_ds),
        dropout=dropout,

        channel_mult=channel_mult,
        use_checkpoint=use_checkpoint,

        use_fp16=use_fp16,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        use_new_attention_order=use_new_attention_order,
    )


def create_gaussian_diffusion(
    *,
    steps=1000,
    learn_sigma=False,
    noise_schedule="linear",
    use_kl=False,
    predict_xstart=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    timestep_respacing="",
    device=None,
):
    print("Diffusion_step:",steps)
#######betas是一个 numpy 数组，控制每步加多少噪声############################################
    betas = gd.get_named_beta_schedule(noise_schedule, steps)

############L2 损失，直接对预测噪声或图像做均方误差#############################################
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE

###########timestep_respacing 默认空串，表示使用所有连续的 [0,1,…,steps-1],若指定如 "ddim50" 或 "100,50,25,10"，则可以跳跃式挑选子序列
    if not timestep_respacing:
        timestep_respacing = [steps]
        
    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        device=device,
    )


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def local_config_path(script_name=None, argv=None):
    """Return the local YAML path requested on CLI, env, or convention."""
    argv = sys.argv[1:] if argv is None else list(argv)
    for index, arg in enumerate(argv):
        if arg.startswith("--config="):
            return arg.split("=", 1)[1]
        if arg == "--config" and index + 1 < len(argv):
            return argv[index + 1]
    configured = os.environ.get("LAMINO_CONFIG")
    if configured:
        return configured
    candidates = []
    if script_name:
        candidates.extend(
            [
                f"{script_name}.ymal",
                f"{script_name}.yaml",
                f".{script_name}.local.yaml",
                f".{script_name}.local.ymal",
            ]
        )
        if script_name == "sample":
            candidates.insert(1, "sampl.ymal")
    candidates.extend([".local.yaml", ".local.ymal"])
    return next((candidate for candidate in candidates if os.path.isfile(candidate)), candidates[0])


def load_local_config(defaults, path=None):
    """Overlay defaults with values from an optional, untracked YAML file.

    Command-line arguments still take precedence because this function is called
    before the argparse parser is populated. Keys for another entry point are
    ignored so one shared file can contain both training and inference settings.
    """
    path = path or local_config_path()
    if not path or not os.path.isfile(path):
        return defaults
    if yaml is None:
        raise RuntimeError("PyYAML is required to read the local configuration file.")
    with open(path, "r", encoding="utf-8") as handle:
        values = yaml.safe_load(handle) or {}
    if not isinstance(values, dict):
        raise ValueError(f"Local config {path} must contain a YAML mapping.")
    merged = dict(defaults)
    merged.update({key: value for key, value in values.items() if key in defaults})
    return merged


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")
