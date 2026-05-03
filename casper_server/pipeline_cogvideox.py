"""CogVideoX-Fun-V1.5-5b-InP バックエンドの Casper パイプライン実装。

`vendor/gen-omnimatte-public/inference/cogvideox_fun/predict_v2v.py` の
`load_pipeline` / `run_inference` 相当を `absl` 非依存で再実装したもの。
"""
import logging
import os
import uuid

import torch

from casper_server.pipeline_common import ensure_repo_on_path, save_video_high_quality
from server.model import (
    CASPER_FPS,
    CASPER_MATTING_MODE,
    CASPER_NUM_INFERENCE_STEPS,
    CASPER_REPO_DIR,
    CASPER_TEMPORAL_WINDOW_SIZE,
    CASPER_TRANSFORMER_PATH,
)


logger = logging.getLogger(__name__)


def build_default_config():
    """`config/default_cogvideox.py` を読み、本仕様で必要な値を上書きした ConfigDict を返す。"""
    ensure_repo_on_path()
    from config.default_cogvideox import get_config

    cfg = get_config()
    cfg.experiment.matting_mode = CASPER_MATTING_MODE
    cfg.experiment.skip_if_exists = False
    cfg.data.fps = CASPER_FPS
    cfg.video_model.transformer_path = CASPER_TRANSFORMER_PATH
    cfg.video_model.num_inference_steps = CASPER_NUM_INFERENCE_STEPS
    cfg.video_model.temporal_window_size = CASPER_TEMPORAL_WINDOW_SIZE
    return cfg


def load_pipeline(cfg):
    """`predict_v2v.load_pipeline` 相当（CogVideoX 用）。"""
    ensure_repo_on_path()

    from diffusers import (CogVideoXDDIMScheduler, DDIMScheduler,
                           DPMSolverMultistepScheduler,
                           EulerAncestralDiscreteScheduler,
                           EulerDiscreteScheduler, PNDMScheduler)
    from videox_fun.dist import set_multi_gpus_devices
    from videox_fun.models import (AutoencoderKLCogVideoX,
                                   CogVideoXTransformer3DModel, T5EncoderModel,
                                   T5Tokenizer)
    from videox_fun.pipeline import (CogVideoXFunInpaintPipeline,
                                     CogVideoXFunPipeline)
    from videox_fun.utils.fp8_optimization import convert_weight_dtype_wrapper
    from videox_fun.utils.lora_utils import merge_lora

    model_name = cfg.video_model.model_name
    weight_dtype = cfg.system.weight_dtype
    device = set_multi_gpus_devices(cfg.system.ulysses_degree, cfg.system.ring_degree)

    # model_name はリポジトリ相対パスなので cwd を合わせる
    prev_cwd = os.getcwd()
    os.chdir(CASPER_REPO_DIR)
    try:
        transformer_dtype = (
            torch.float8_e4m3fn
            if cfg.system.gpu_memory_mode == "model_cpu_offload_and_qfloat8"
            else weight_dtype
        )
        transformer = CogVideoXTransformer3DModel.from_pretrained(
            model_name,
            subfolder="transformer",
            low_cpu_mem_usage=True,
            torch_dtype=transformer_dtype,
            use_vae_mask=cfg.video_model.use_vae_mask,
            stack_mask=cfg.video_model.stack_mask,
        ).to(weight_dtype)

        if cfg.video_model.transformer_path:
            logger.info("loading casper transformer from %s", cfg.video_model.transformer_path)
            if cfg.video_model.transformer_path.endswith("safetensors"):
                from safetensors.torch import load_file
                state_dict = load_file(cfg.video_model.transformer_path)
            else:
                state_dict = torch.load(cfg.video_model.transformer_path, map_location="cpu")
            state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

            # use_vae_mask / stack_mask 使用時は patch_embed の入力次元が変わるので
            # ckpt のままでは shape が合わない。predict_v2v.py のロジックを移植
            param_name = "patch_embed.proj.weight"
            if (
                (cfg.video_model.use_vae_mask or cfg.video_model.stack_mask) and
                state_dict[param_name].size(1) != transformer.state_dict()[param_name].size(1)
            ):
                logger.info("patch_embed.proj.weight size mismatch; remapping")
                latent_ch = 16
                feat_scale = 8
                feat_dim = int(latent_ch * feat_scale)
                new_total_dim = transformer.state_dict()[param_name].size(1)
                transformer.state_dict()[param_name][:, :feat_dim] = state_dict[param_name][:, :feat_dim]
                transformer.state_dict()[param_name][:, -feat_dim:] = state_dict[param_name][:, -feat_dim:]
                for i in range(feat_dim, new_total_dim - feat_dim, feat_scale):
                    transformer.state_dict()[param_name][:, i:i + feat_scale] = state_dict[param_name][:, feat_dim:-feat_dim]
                state_dict[param_name] = transformer.state_dict()[param_name]

            m, u = transformer.load_state_dict(state_dict, strict=False)
            logger.info("transformer missing=%d unexpected=%d", len(m), len(u))

        vae = AutoencoderKLCogVideoX.from_pretrained(
            model_name, subfolder="vae"
        ).to(weight_dtype)

        if cfg.video_model.vae_path:
            from safetensors.torch import load_file
            state_dict = (
                load_file(cfg.video_model.vae_path)
                if cfg.video_model.vae_path.endswith("safetensors")
                else torch.load(cfg.video_model.vae_path, map_location="cpu")
            )
            state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict
            vae.load_state_dict(state_dict, strict=False)

        tokenizer = T5Tokenizer.from_pretrained(model_name, subfolder="tokenizer")
        text_encoder = T5EncoderModel.from_pretrained(
            model_name, subfolder="text_encoder", torch_dtype=weight_dtype
        )

        scheduler_cls = {
            "Euler": EulerDiscreteScheduler,
            "Euler A": EulerAncestralDiscreteScheduler,
            "DPM++": DPMSolverMultistepScheduler,
            "PNDM": PNDMScheduler,
            "DDIM_Cog": CogVideoXDDIMScheduler,
            "DDIM_Origin": DDIMScheduler,
        }[cfg.video_model.sampler_name]
        scheduler = scheduler_cls.from_pretrained(model_name, subfolder="scheduler")

        # transformer の入力チャネルが VAE の latent と一致しなければ inpaint パイプライン
        if transformer.config.in_channels != vae.config.latent_channels:
            pipeline = CogVideoXFunInpaintPipeline(
                vae=vae,
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                transformer=transformer,
                scheduler=scheduler,
            )
        else:
            pipeline = CogVideoXFunPipeline(
                vae=vae,
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                transformer=transformer,
                scheduler=scheduler,
            )

        if cfg.system.ulysses_degree > 1 or cfg.system.ring_degree > 1:
            transformer.enable_multi_gpus_inference()

        if cfg.system.gpu_memory_mode == "sequential_cpu_offload":
            pipeline.enable_sequential_cpu_offload(device=device)
        elif cfg.system.gpu_memory_mode == "model_cpu_offload_and_qfloat8":
            convert_weight_dtype_wrapper(transformer, weight_dtype)
            pipeline.enable_model_cpu_offload(device=device)
        elif cfg.system.gpu_memory_mode == "model_cpu_offload":
            pipeline.enable_model_cpu_offload(device=device)
        else:
            pipeline.to(device=device)

        generator = torch.Generator(device=device).manual_seed(cfg.system.seed)

        if cfg.video_model.lora_path:
            pipeline = merge_lora(
                pipeline, cfg.video_model.lora_path,
                cfg.video_model.lora_weight, device=device,
            )
    finally:
        os.chdir(prev_cwd)

    return pipeline, vae, generator


@torch.no_grad()
def run_one_seq(
    cfg,
    pipeline,
    vae,
    generator,
    seq_dir: str,
    save_dir: str,
    sample_size: tuple[int, int],
) -> str:
    """1 シーケンスについて CogVideoX Casper を走らせ、出力 mp4 のパスを返す。"""
    ensure_repo_on_path()

    from videox_fun.utils.utils import get_video_mask_input

    seq_name = os.path.basename(os.path.normpath(seq_dir))
    data_rootdir = os.path.dirname(os.path.normpath(seq_dir))

    video_length = cfg.data.max_video_length
    video_length = (
        int((video_length - 1) // vae.config.temporal_compression_ratio
            * vae.config.temporal_compression_ratio) + 1
        if video_length != 1 else 1
    )

    keep_fg_ids = [-1]  # all_fg

    # CogVideoX 版 get_video_mask_input は 4 値返すが clip_image は使わない
    input_video, input_video_mask, prompt, _ = get_video_mask_input(
        seq_name,
        sample_size=sample_size,
        keep_fg_ids=keep_fg_ids,
        max_video_length=video_length,
        temporal_window_size=cfg.video_model.temporal_window_size,
        data_rootdir=data_rootdir,
        use_trimask=cfg.video_model.use_trimask,
        dilate_width=cfg.data.dilate_width,
    )

    sample = pipeline(
        prompt,
        num_frames=cfg.video_model.temporal_window_size,
        negative_prompt=cfg.video_model.negative_prompt,
        height=sample_size[0],
        width=sample_size[1],
        generator=generator,
        guidance_scale=cfg.video_model.guidance_scale,
        num_inference_steps=cfg.video_model.num_inference_steps,
        video=input_video,
        mask_video=input_video_mask,
        strength=cfg.video_model.denoise_strength,
        use_trimask=cfg.video_model.use_trimask,
        zero_out_mask_region=cfg.video_model.zero_out_mask_region,
        skip_unet=cfg.experiment.skip_unet,
        use_vae_mask=cfg.video_model.use_vae_mask,
        stack_mask=cfg.video_model.stack_mask,
    ).videos

    os.makedirs(save_dir, exist_ok=True)

    save_video_name = f"{seq_name}-fg=" + "_".join([f"{i:02d}" for i in keep_fg_ids])
    prefix = save_video_name + f"-{uuid.uuid4().hex[:8]}"
    video_path = os.path.join(save_dir, prefix + ".mp4")
    save_video_high_quality(sample, video_path, fps=cfg.data.fps)
    return video_path
