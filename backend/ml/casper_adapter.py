"""gen-omnimatte-public のパイプラインを呼ぶ薄いラッパ。

gen-omnimatte-publicの `run_inference` は入出力が合わないので、
`load_pipeline` で得たパイプラインを使って独自に実装する。
"""
import logging
import os
import sys

import torch

from backend.config import CASPER_NUM_INFERENCE_STEPS, CASPER_TEMPORAL_WINDOW_SIZE
from backend.media.video_io import VideoMetadata


logger = logging.getLogger(__name__)


_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CASPER_REPO_DIR = os.path.join(_BACKEND_DIR, "vendor", "gen-omnimatte-public")
CASPER_TRANSFORMER_PATH = os.path.join(
    CASPER_REPO_DIR, "models", "Casper", "wan2.1_fun_1.3b_casper.safetensors"
)
CASPER_MATTING_MODE = "all_fg"


if CASPER_REPO_DIR not in sys.path:
    sys.path.insert(0, CASPER_REPO_DIR)


def _build_default_config():
    from config.default_wan import get_config
    cfg = get_config()
    cfg.experiment.matting_mode = CASPER_MATTING_MODE
    cfg.experiment.skip_if_exists = False
    cfg.video_model.transformer_path = CASPER_TRANSFORMER_PATH
    cfg.video_model.num_inference_steps = CASPER_NUM_INFERENCE_STEPS
    cfg.video_model.temporal_window_size = CASPER_TEMPORAL_WINDOW_SIZE
    return cfg


class CasperPipeline:
    def __init__(self) -> None:
        from inference.wan21_fun import predict_v2v
        self._cfg = _build_default_config()
        self._pipeline, self._vae, self._generator = predict_v2v.load_pipeline(self._cfg)

    @torch.no_grad()
    def run(
        self,
        seq_dir: str,
        save_dir: str,
        sample_size: tuple[int, int],
        meta: VideoMetadata,
    ) -> str:
        """1 シーケンスについて Casper を走らせ、出力 mp4 のパスを返す。

        `seq_dir` には `input_video.mp4`, `mask_00.mp4`, `prompt.json` が
        既に配置されている必要がある（呼び出し側で準備）。

        `sample_size` は16 の倍数でなければならない（Wan2.1 の VAE / patch サイズ制約）。
        呼び出し側で丸めること。
        """
        from videox_fun.utils.utils import get_video_mask_input

        cfg = self._cfg
        vae = self._vae

        seq_name = os.path.basename(os.path.normpath(seq_dir))
        data_rootdir = os.path.dirname(os.path.normpath(seq_dir))

        video_length = cfg.data.max_video_length
        video_length = (
            int((video_length - 1) // vae.config.temporal_compression_ratio
                * vae.config.temporal_compression_ratio) + 1
            if video_length != 1 else 1
        )

        input_video, input_video_mask, prompt, clip_image = get_video_mask_input(
            seq_name,
            sample_size=sample_size,
            keep_fg_ids=[-1],  # all_fg
            max_video_length=video_length,
            temporal_window_size=cfg.video_model.temporal_window_size,
            data_rootdir=data_rootdir,
            use_trimask=cfg.video_model.use_trimask,
            dilate_width=cfg.data.dilate_width,
        )

        sample = self._pipeline(
            prompt,
            num_frames=cfg.video_model.temporal_window_size,
            negative_prompt=cfg.video_model.negative_prompt,
            height=sample_size[0],
            width=sample_size[1],
            generator=self._generator,
            guidance_scale=cfg.video_model.guidance_scale,
            num_inference_steps=cfg.video_model.num_inference_steps,
            video=input_video,
            mask_video=input_video_mask,
            use_trimask=cfg.video_model.use_trimask,
            zero_out_mask_region=cfg.video_model.zero_out_mask_region,
            clip_image=clip_image,
            skip_unet=cfg.experiment.skip_unet,
        ).videos

        # reflection paddingの切り詰め
        if sample.shape[2] > meta.num_frames:
            logger.info(
                "trimming Casper output: %d -> %d frames (drop temporal padding)",
                sample.shape[2], meta.num_frames,
            )
            sample = sample[:, :, :meta.num_frames]

        os.makedirs(save_dir, exist_ok=True)
        video_path = os.path.join(save_dir, "output.mp4")
        _save_video_mp4(sample, video_path, fps=meta.fps)
        return video_path


def _save_video_mp4(sample, path: str, fps: float) -> None:
    import imageio
    frames = (sample[0].permute(1, 2, 3, 0) * 255).clamp(0, 255).byte().numpy()
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
