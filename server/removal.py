import asyncio
import glob
import json
import logging
import os
import shutil
import subprocess
import tempfile
import uuid

import numpy as np

from server.model import (
    CASPER_CONFIG_PATH,
    CASPER_DEFAULT_PROMPT,
    CASPER_FPS,
    CASPER_MATTING_MODE,
    CASPER_NUM_INFERENCE_STEPS,
    CASPER_PYTHON,
    CASPER_REPO_DIR,
    CASPER_SAMPLE_SIZE,
    CASPER_TEMPORAL_WINDOW_SIZE,
    CASPER_TRANSFORMER_PATH,
)
from server.video_io import write_mask_mp4


logger = logging.getLogger(__name__)


class CasperWeightMissingError(FileNotFoundError):
    """Casper 用の重みファイルが配置されていない場合に投げる。"""


class CasperRunError(RuntimeError):
    """Casper の subprocess が異常終了した場合に投げる。"""


async def run_foreground_removal(
    base_video_path: str,
    masks: np.ndarray,
    fps: float,
) -> str:
    """前景削除済みの mp4 ファイルパスを返す。失敗時は例外。

    呼び出し側が `os.unlink` で後片付けする責務を持つ。
    """
    if not os.path.exists(CASPER_TRANSFORMER_PATH):
        raise CasperWeightMissingError(
            f"casper model not found: {CASPER_TRANSFORMER_PATH}"
        )

    return await asyncio.to_thread(_run_sync, base_video_path, masks, fps)


def _run_sync(base_video_path: str, masks: np.ndarray, fps: float) -> str:
    work_root = tempfile.mkdtemp(prefix="casper_run_")
    seq_name = f"seq_{uuid.uuid4().hex[:8]}"
    seq_dir = os.path.join(work_root, "data", seq_name)
    save_dir = os.path.join(work_root, "out")
    os.makedirs(seq_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    try:
        # 1. 入力動画を input_video.mp4 として配置（コピー）
        shutil.copyfile(base_video_path, os.path.join(seq_dir, "input_video.mp4"))

        # 2. マスクを mask_00.mp4 として書き出し（白=前景, 黒=背景, base video と同じ解像度・fps）
        write_mask_mp4(
            masks=masks,
            fps=fps,
            out_path=os.path.join(seq_dir, "mask_00.mp4"),
        )

        # 3. prompt.json を生成
        with open(os.path.join(seq_dir, "prompt.json"), "w", encoding="utf-8") as f:
            json.dump({"bg": CASPER_DEFAULT_PROMPT}, f)

        # 4. predict_v2v.py を subprocess で起動
        cmd = [
            CASPER_PYTHON,
            "inference/wan2.1_fun/predict_v2v.py",
            f"--config={CASPER_CONFIG_PATH}",
            f"--config.experiment.matting_mode={CASPER_MATTING_MODE}",
            f"--config.experiment.run_seqs={seq_name}",
            f"--config.experiment.save_path={save_dir}",
            f"--config.data.data_rootdir={os.path.dirname(seq_dir)}",
            f"--config.data.sample_size={CASPER_SAMPLE_SIZE}",
            f"--config.data.fps={CASPER_FPS}",
            f"--config.video_model.transformer_path={CASPER_TRANSFORMER_PATH}",
            f"--config.video_model.num_inference_steps={CASPER_NUM_INFERENCE_STEPS}",
            f"--config.video_model.temporal_window_size={CASPER_TEMPORAL_WINDOW_SIZE}",
        ]
        logger.info("running casper: %s", " ".join(cmd))
        result = subprocess.run(
            cmd,
            cwd=CASPER_REPO_DIR,
            capture_output=True,
        )
        if result.returncode != 0:
            stderr = result.stderr.decode(errors="replace")
            tail = stderr[-2000:]
            raise CasperRunError(f"foreground removal failed: {tail}")

        # 5. 生成された mp4 を取得（*-fg=-1-XXXX.mp4 を選ぶ）
        candidates = sorted(
            p for p in glob.glob(os.path.join(save_dir, f"{seq_name}-fg=-1-*.mp4"))
            if not p.endswith("_tuple.mp4")
        )
        if not candidates:
            raise CasperRunError(
                f"casper finished but produced no mp4 in {save_dir}"
            )
        produced = candidates[-1]

        # 別の場所に取り出してから tmpdir を消す（tmpdir は丸ごと削除する）
        out_fd, out_path = tempfile.mkstemp(prefix="casper_out_", suffix=".mp4")
        os.close(out_fd)
        shutil.copyfile(produced, out_path)
        return out_path
    finally:
        shutil.rmtree(work_root, ignore_errors=True)
