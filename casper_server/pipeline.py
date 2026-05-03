"""gen-omnimatte-public のパイプラインを `absl` 非依存に再実装したラッパ。

`vendor/gen-omnimatte-public/inference/wan2.1_fun/predict_v2v.py` の
`load_pipeline` / `run_inference` 相当を、本サーバ・sidecar から import 可能な
プレーンな関数として提供する。`predict_v2v.py` は import するだけで
`absl.flags` を介してグローバル CLI フラグを登録するため、ここからは import しない。
"""
import logging
import os
import sys
import uuid

import numpy as np
import torch
from omegaconf import OmegaConf

from server.model import (
    CASPER_FPS,
    CASPER_MATTING_MODE,
    CASPER_NUM_INFERENCE_STEPS,
    CASPER_REPO_DIR,
    CASPER_TEMPORAL_WINDOW_SIZE,
    CASPER_TRANSFORMER_PATH,
    FG_REPLACE_DIFF_THRESHOLD,
    FG_REPLACE_MASK_DILATE,
    FOREGROUND_ONLY_REPLACE,
)


logger = logging.getLogger(__name__)


def _ensure_repo_on_path() -> None:
    """gen-omnimatte-public の videox_fun / config を import 可能にする。"""
    if CASPER_REPO_DIR not in sys.path:
        sys.path.insert(0, CASPER_REPO_DIR)


def build_default_config():
    """`config/default_wan.py` を読み、本仕様で必要な値を上書きした ConfigDict を返す。"""
    _ensure_repo_on_path()
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
    """`predict_v2v.load_pipeline` 相当。`absl` を使わず純粋に cfg だけで動かす。"""
    _ensure_repo_on_path()

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
    """1 シーケンスについて Casper を走らせ、出力 mp4 のパスを返す。

    `seq_dir` には `input_video.mp4`, `mask_00.mp4`, `prompt.json` が
    既に配置されている必要がある（呼び出し側で準備）。

    `sample_size` は `(height, width)` の int タプル。両次元とも 16 の倍数で
    なければならない（Wan2.1 の VAE / patch サイズ制約）。呼び出し側で
    丸めること。
    """
    _ensure_repo_on_path()

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

    # get_video_mask_input は内部で os.path.join(data_rootdir, input_video_name, ...) を組む
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

    if FOREGROUND_ONLY_REPLACE:
        # 前景部分のみ Casper 出力で書き換え、それ以外は base video の画素を保持。
        # 判定マスク = (SAM2 マスクを dilate) AND (Casper-base の per-pixel diff > 閾値)（案 D）
        casper_frames = _tensor_to_uint8_rgb_frames(sample)
        composite_frames = _compose_foreground_only(
            casper_frames=casper_frames,
            input_video_path=os.path.join(seq_dir, "input_video.mp4"),
            mask_video_path=os.path.join(seq_dir, "mask_00.mp4"),
            diff_threshold=FG_REPLACE_DIFF_THRESHOLD,
            mask_dilate=FG_REPLACE_MASK_DILATE,
        )
        _save_uint8_frames_high_quality(composite_frames, video_path, fps=cfg.data.fps)
    else:
        _save_video_high_quality(sample, video_path, fps=cfg.data.fps)
    return video_path


def _tensor_to_uint8_rgb_frames(sample) -> list[np.ndarray]:
    """`(B, C, T, H, W)` のテンソル（[0, 1]）を `(H, W, 3)` uint8 RGB フレーム列に変換。

    `videox_fun.utils.utils.save_videos_grid` のテンソル変換ロジックを抜き出したもの。
    """
    import torchvision
    from einops import rearrange
    from PIL import Image

    videos = rearrange(sample, "b c t h w -> t b c h w")
    frames: list[np.ndarray] = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=6)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        x = (x * 255).numpy().astype(np.uint8)
        frames.append(np.array(Image.fromarray(x)))
    return frames


def _save_uint8_frames_high_quality(frames, path: str, fps: float) -> None:
    """RGB uint8 フレーム列を H.264 / yuv420p / crf 15 で mp4 に書き出す。"""
    import imageio

    imageio.mimsave(
        path,
        frames,
        fps=fps,
        codec="libx264",
        quality=10,                # imageio 互換の品質指標 (0-10、10 が最高)
        macro_block_size=8,        # default 16 だと意図しない resize が入る
        output_params=[
            "-crf", "15",          # 視覚的にロスレス相当（小さいほど高品質、18 が一般的、15 は高品質寄り）
            "-preset", "medium",   # 圧縮効率と速度のバランス
            "-pix_fmt", "yuv420p", # 互換性確保
        ],
    )


def _save_video_high_quality(sample, path: str, fps: float) -> None:
    """`(B, C, T, H, W)` のテンソル（[0, 1]）を高品質 mp4 に書き出す（後方互換ラッパ）。"""
    frames = _tensor_to_uint8_rgb_frames(sample)
    _save_uint8_frames_high_quality(frames, path, fps)


def _compose_foreground_only(
    casper_frames: list[np.ndarray],
    input_video_path: str,
    mask_video_path: str,
    diff_threshold: int,
    mask_dilate: int,
) -> list[np.ndarray]:
    """Casper 出力フレームと base video を、判定マスクに基づいて合成する。

    判定マスク = (SAM2 マスクを `mask_dilate` ピクセル dilate) AND
                 (Casper と base の per-channel max diff > `diff_threshold`)

    判定マスクが立っている画素 → Casper 出力で書き換え
    判定マスクが立っていない画素 → base video の画素を保持

    引数:
      `casper_frames`: list of `(H, W, 3)` uint8 RGB（`_tensor_to_uint8_rgb_frames` の出力）
      `input_video_path`: base video の絶対パス
      `mask_video_path`: SAM2 が生成したバイナリマスク mp4（白=対象前景）
      `diff_threshold`: 0-255 の per-channel 差分しきい値
      `mask_dilate`: SAM2 マスクの dilate 半径ピクセル

    戻り値: list of `(H_base, W_base, 3)` uint8 RGB。base video 解像度に揃えて返す。
    """
    import cv2

    # 1. base video を全フレーム RGB で読み込む
    base_frames = _decode_video_rgb(input_video_path)
    if not base_frames:
        raise RuntimeError(f"failed to decode base video: {input_video_path}")
    base_h, base_w = base_frames[0].shape[:2]

    # 2. SAM2 マスクを bool で読み込み（白=前景）
    sam2_masks = _decode_video_mask(mask_video_path)

    # 3. フレーム数を揃える（最小に切る）
    n = min(len(casper_frames), len(base_frames), len(sam2_masks))
    if n == 0:
        raise RuntimeError("no frames to compose")

    # 4. dilate カーネル（楕円）
    dilate_kernel = None
    if mask_dilate > 0:
        k = 2 * mask_dilate + 1
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    out: list[np.ndarray] = []
    for i in range(n):
        casper_frame = casper_frames[i]
        base_frame = base_frames[i]
        sam2_mask = sam2_masks[i]

        # Casper を base 解像度に合わせる（解像度ミスマッチがある場合）
        if casper_frame.shape[:2] != (base_h, base_w):
            casper_frame = cv2.resize(
                casper_frame, (base_w, base_h), interpolation=cv2.INTER_LINEAR
            )

        # SAM2 マスクを base 解像度に合わせて dilate
        if sam2_mask.shape != (base_h, base_w):
            sam2_mask = cv2.resize(
                sam2_mask.astype(np.uint8), (base_w, base_h), interpolation=cv2.INTER_NEAREST
            ) > 0
        if dilate_kernel is not None:
            sam2_mask = cv2.dilate(sam2_mask.astype(np.uint8), dilate_kernel, iterations=1) > 0

        # diff > threshold（per-channel max）
        diff = np.abs(casper_frame.astype(np.int16) - base_frame.astype(np.int16)).max(axis=-1)
        diff_mask = diff > diff_threshold

        # 案 D: AND
        modified = sam2_mask & diff_mask

        # 合成
        composite = base_frame.copy()
        composite[modified] = casper_frame[modified]
        out.append(composite)

    return out


def _decode_video_rgb(path: str) -> list[np.ndarray]:
    """mp4 を全フレーム RGB (H, W, 3) uint8 として読み込む。"""
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return []
    frames: list[np.ndarray] = []
    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret or frame_bgr is None:
                break
            frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    return frames


def _decode_video_mask(path: str) -> list[np.ndarray]:
    """マスク mp4 を全フレーム bool (H, W) として読み込む。白(>127)=前景。"""
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return []
    frames: list[np.ndarray] = []
    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret or frame_bgr is None:
                break
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            frames.append(gray > 127)
    finally:
        cap.release()
    return frames
