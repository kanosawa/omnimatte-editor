"""バックエンド（Wan / CogVideoX）共通のヘルパ。"""
import sys

import numpy as np

from server.model import CASPER_REPO_DIR


def ensure_repo_on_path() -> None:
    """gen-omnimatte-public の videox_fun / config を import 可能にする。"""
    if CASPER_REPO_DIR not in sys.path:
        sys.path.insert(0, CASPER_REPO_DIR)


def save_video_high_quality(sample, path: str, fps: float) -> None:
    """`(B, C, T, H, W)` のテンソル（[0, 1]）を H.264 / yuv420p / crf 15 で mp4 に書き出す。

    `videox_fun.utils.utils.save_videos_grid` が `imageio.mimsave()` をデフォルト
    引数で呼ぶため出力 mp4 のビットレートが低い。本関数は同じテンソル変換ロジックで
    エンコードパラメータだけ高品質側に振った差し替え版。
    """
    import imageio
    import torchvision
    from einops import rearrange
    from PIL import Image

    videos = rearrange(sample, "b c t h w -> t b c h w")
    frames = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=6)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        x = (x * 255).numpy().astype(np.uint8)
        frames.append(np.array(Image.fromarray(x)))

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
