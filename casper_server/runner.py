"""`/run` と `/preload` で共有する Casper 推論ヘルパ。

両エンドポイントとも「動画 + マスクを sidecar の一時ディレクトリに置いて Casper
を回し、出力 mp4 バイトを取り出す」処理を行う。違いは:
  - `/run`: 同期実行、結果を即返す
  - `/preload`: バックグラウンド実行、`output_cache` に保存して終わる
"""
import asyncio
import json
import logging
import os
import shutil
import tempfile

from casper_server.holder import casper_holder
from casper_server.pipeline import run_one_seq
from server.model import CASPER_DEFAULT_PROMPT


logger = logging.getLogger(__name__)


# 同時に 1 件しか pipeline を回さない（GPU メモリ・状態整合性）。
# /run と /preload の双方が共有する。
run_lock = asyncio.Lock()


def round_to_multiple_of_16(value: int) -> int:
    """Wan2.1 の VAE (8x) + transformer patch (2x) 制約に合わせ、16 の倍数に丸める。

    最近傍に丸める。最小値は 16。
    """
    if value <= 0:
        raise ValueError(f"invalid dimension: {value}")
    snapped = int(round(value / 16.0)) * 16
    return max(16, snapped)


def do_pipeline_run(
    input_video_path: str,
    mask_path: str,
    width: int,
    height: int,
    prompt: str,
) -> bytes:
    """既に一時ファイルとして配置された動画・マスクで Casper を回し、mp4 バイトを返す。

    呼び出し側で `run_lock` を取得しておくこと。GPU を 1 件ずつ使う前提。
    """
    snap_h = round_to_multiple_of_16(height)
    snap_w = round_to_multiple_of_16(width)
    sample_size = (snap_h, snap_w)
    logger.info(
        "casper pipeline run: input=%dx%d -> sample_size=%dx%d",
        width, height, snap_w, snap_h,
    )

    work_root = tempfile.mkdtemp(prefix="casper_pipe_")
    seq_name = f"seq_{os.path.basename(work_root).split('_', 2)[-1]}"
    seq_dir = os.path.join(work_root, "data", seq_name)
    save_dir = os.path.join(work_root, "out")
    os.makedirs(seq_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    try:
        # gen-omnimatte の get_video_mask_input は seq_dir 内の input_video.mp4 と
        # mask_*.mp4 を読むので、配置し直す
        shutil.copyfile(input_video_path, os.path.join(seq_dir, "input_video.mp4"))
        shutil.copyfile(mask_path, os.path.join(seq_dir, "mask_00.mp4"))

        with open(os.path.join(seq_dir, "prompt.json"), "w", encoding="utf-8") as f:
            json.dump({"bg": prompt or CASPER_DEFAULT_PROMPT}, f)

        out_path = run_one_seq(
            casper_holder.cfg,
            casper_holder.pipeline,
            casper_holder.vae,
            casper_holder.generator,
            seq_dir,
            save_dir,
            sample_size,
        )

        with open(out_path, "rb") as f:
            return f.read()
    finally:
        shutil.rmtree(work_root, ignore_errors=True)
