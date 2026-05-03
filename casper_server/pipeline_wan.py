"""Wan2.1-Fun-1.3B-InP バックエンドの Casper パイプライン実装。

`vendor/gen-omnimatte-public/inference/wan2.1_fun/predict_v2v.py` の
`load_pipeline` / `run_inference` 相当を `absl` 非依存で再実装したもの。
"""
import logging
import os
import uuid

import torch
from omegaconf import OmegaConf

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
    """`config/default_wan.py` を読み、本仕様で必要な値を上書きした ConfigDict を返す。"""
    ensure_repo_on_path()
    from config.default_wan import get_config

    cfg = get_config()
    # 本仕様の上書き値（sample_size はリクエスト毎に run_one_seq で上書き）
    cfg.experiment.matting_mode = CASPER_MATTING_MODE
    cfg.experiment.skip_if_exists = False
    cfg.data.fps = CASPER_FPS
    cfg.video_model.transformer_path = CASPER_TRANSFORMER_PATH
    cfg.video_model.num_inference_steps = CASPER_NUM_INFERENCE_STEPS
    cfg.video_model.temporal_window_size = CASPER_TEMPORAL_WINDOW_SIZE
    return cfg


def load_pipeline(cfg):
    """`predict_v2v.load_pipeline` 相当（Wan 用）。"""
    ensure_repo_on_path()

    from diffusers import FlowMatchEulerDiscreteScheduler
    from transformers import AutoTokenizer
    from videox_fun.dist import set_multi_gpus_devices
    from videox_fun.models import (AutoencoderKLWan, CLIPModel,
                                   WanT5EncoderModel, WanTransformer3DModel)
    from videox_fun.models.cache_utils import get_teacache_coefficients
    from videox_fun.pipeline import WanFunInpaintPipeline
    from videox_fun.utils.fp8_optimization import (
        convert_model_weight_to_float8, convert_weight_dtype_wrapper,
        replace_parameters_by_name)
    from videox_fun.utils.lora_utils import merge_lora
    from videox_fun.utils.utils import filter_kwargs

    model_name = cfg.video_model.model_name
    weight_dtype = cfg.system.weight_dtype
    config_model = OmegaConf.load(
        os.path.join(CASPER_REPO_DIR, cfg.video_model.config_path)
    )
    device = set_multi_gpus_devices(cfg.system.ulysses_degree, cfg.system.ring_degree)

    # 重みなどはリポジトリの相対パスで指定されているので、cwd を合わせる必要がある
    prev_cwd = os.getcwd()
    os.chdir(CASPER_REPO_DIR)
    try:
        transformer = WanTransformer3DModel.from_pretrained(
            os.path.join(model_name, config_model['transformer_additional_kwargs'].get('transformer_subpath', 'transformer')),
            transformer_additional_kwargs=OmegaConf.to_container(config_model['transformer_additional_kwargs']),
            low_cpu_mem_usage=True,
            torch_dtype=weight_dtype,
        )

        if cfg.video_model.transformer_path:
            logger.info("loading casper transformer from %s", cfg.video_model.transformer_path)
            if cfg.video_model.transformer_path.endswith("safetensors"):
                from safetensors.torch import load_file
                state_dict = load_file(cfg.video_model.transformer_path)
            else:
                state_dict = torch.load(cfg.video_model.transformer_path, map_location="cpu")
            state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict
            m, u = transformer.load_state_dict(state_dict, strict=False)
            logger.info("transformer missing=%d unexpected=%d", len(m), len(u))

        vae = AutoencoderKLWan.from_pretrained(
            os.path.join(model_name, config_model['vae_kwargs'].get('vae_subpath', 'vae')),
            additional_kwargs=OmegaConf.to_container(config_model['vae_kwargs']),
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

        tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(model_name, config_model['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')),
        )

        text_encoder = WanT5EncoderModel.from_pretrained(
            os.path.join(model_name, config_model['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
            additional_kwargs=OmegaConf.to_container(config_model['text_encoder_kwargs']),
            low_cpu_mem_usage=True,
            torch_dtype=weight_dtype,
        ).eval()

        clip_image_encoder = CLIPModel.from_pretrained(
            os.path.join(model_name, config_model['image_encoder_kwargs'].get('image_encoder_subpath', 'image_encoder')),
        ).to(weight_dtype).eval()

        scheduler_cls = {"Flow": FlowMatchEulerDiscreteScheduler}[cfg.video_model.sampler_name]
        scheduler = scheduler_cls(
            **filter_kwargs(scheduler_cls, OmegaConf.to_container(config_model['scheduler_kwargs']))
        )

        pipeline = WanFunInpaintPipeline(
            transformer=transformer,
            vae=vae,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            scheduler=scheduler,
            clip_image_encoder=clip_image_encoder,
        )
        if cfg.system.ulysses_degree > 1 or cfg.system.ring_degree > 1:
            transformer.enable_multi_gpus_inference()

        if cfg.system.gpu_memory_mode == "sequential_cpu_offload":
            replace_parameters_by_name(transformer, ["modulation"], device=device)
            transformer.freqs = transformer.freqs.to(device=device)
            pipeline.enable_sequential_cpu_offload(device=device)
        elif cfg.system.gpu_memory_mode == "model_cpu_offload_and_qfloat8":
            convert_model_weight_to_float8(transformer, exclude_module_name=["modulation"])
            convert_weight_dtype_wrapper(transformer, weight_dtype)
            pipeline.enable_model_cpu_offload(device=device)
        elif cfg.system.gpu_memory_mode == "model_cpu_offload":
            pipeline.enable_model_cpu_offload(device=device)
        else:
            pipeline.to(device=device)

        coefficients = get_teacache_coefficients(model_name) if cfg.system.enable_teacache else None
        if coefficients is not None:
            pipeline.transformer.enable_teacache(
                coefficients,
                cfg.video_model.num_inference_steps,
                cfg.system.teacache_threshold,
                num_skip_start_steps=cfg.system.num_skip_start_steps,
                offload=cfg.system.teacache_offload,
            )

        generator = torch.Generator(device=device).manual_seed(cfg.system.seed)

        if cfg.video_model.lora_path:
            pipeline = merge_lora(pipeline, cfg.video_model.lora_path, cfg.video_model.lora_weight)
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
    """1 シーケンスについて Wan Casper を走らせ、出力 mp4 のパスを返す。"""
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

    input_video, input_video_mask, prompt, clip_image = get_video_mask_input(
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
        use_trimask=cfg.video_model.use_trimask,
        zero_out_mask_region=cfg.video_model.zero_out_mask_region,
        clip_image=clip_image,
        skip_unet=cfg.experiment.skip_unet,
    ).videos

    os.makedirs(save_dir, exist_ok=True)

    save_video_name = f"{seq_name}-fg=" + "_".join([f"{i:02d}" for i in keep_fg_ids])
    prefix = save_video_name + f"-{uuid.uuid4().hex[:8]}"
    video_path = os.path.join(save_dir, prefix + ".mp4")
    save_video_high_quality(sample, video_path, fps=cfg.data.fps)
    return video_path
