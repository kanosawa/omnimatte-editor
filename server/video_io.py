import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Iterable

import cv2
import imageio_ffmpeg
import numpy as np


@dataclass
class VideoMetadata:
    width: int
    height: int
    fps: float
    num_frames: int

    @property
    def duration_sec(self) -> float:
        if self.fps <= 0:
            return 0.0
        return self.num_frames / self.fps


def probe_video(path: str) -> VideoMetadata:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {path}")
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    return VideoMetadata(width=width, height=height, fps=fps, num_frames=num_frames)


def composite_overlay_to_mp4(
    original_video_path: str,
    masks_in_order: Iterable[np.ndarray],
    fps: float,
    overlay_color_bgr: tuple[int, int, int] = (64, 64, 255),
    overlay_alpha: float = 0.5,
) -> bytes:
    """原動画の各フレームに対し、マスク領域を半透明色でアルファブレンドした mp4 を返す。

    フロントエンドは `<video>` 要素 1 つで合成済み動画を再生するだけになるので、
    原動画とマスク動画を別々に再生するときに発生するデコーダ間のドリフトが原理的に発生しない。
    """
    masks = list(masks_in_order)
    if not masks:
        raise ValueError("no mask frames provided")

    cap = cv2.VideoCapture(original_video_path)
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {original_video_path}")
    try:
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    except Exception:
        cap.release()
        raise

    # H.264 requires even dimensions
    enc_w = width + (width % 2)
    enc_h = height + (height % 2)

    overlay_color = np.array(overlay_color_bgr, dtype=np.float32)
    alpha = float(max(0.0, min(1.0, overlay_alpha)))

    tmpdir = tempfile.mkdtemp(prefix="omnimatte_composite_")
    try:
        for i, mask in enumerate(masks):
            ret, frame = cap.read()
            if not ret:
                break  # 原動画の方が短い場合は途中で打ち切り

            mask_bool = _normalize_mask(mask, width, height)

            composite = frame
            if mask_bool.any():
                composite = frame.copy()
                composite[mask_bool] = (
                    frame[mask_bool].astype(np.float32) * (1.0 - alpha)
                    + overlay_color * alpha
                ).astype(np.uint8)

            if enc_w != width or enc_h != height:
                composite = cv2.copyMakeBorder(
                    composite, 0, enc_h - height, 0, enc_w - width,
                    cv2.BORDER_CONSTANT, value=0,
                )
            cv2.imwrite(os.path.join(tmpdir, f"frame_{i:06d}.png"), composite)

        cap.release()

        out_path = os.path.join(tmpdir, "composite.mp4")
        safe_fps = max(1.0, float(fps))
        cmd = [
            imageio_ffmpeg.get_ffmpeg_exe(), "-y",
            "-framerate", str(safe_fps),
            "-i", os.path.join(tmpdir, "frame_%06d.png"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "18",
            "-preset", "fast",
            out_path,
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg encoding failed:\n{result.stderr.decode(errors='replace')}"
            )
        with open(out_path, "rb") as f:
            return f.read()
    finally:
        cap.release()
        shutil.rmtree(tmpdir, ignore_errors=True)


def write_mask_mp4(
    masks: np.ndarray,
    fps: float,
    out_path: str,
) -> None:
    """マスク (T, H, W) bool を 3ch の mp4 として書き出す。

    白 (255,255,255) が前景、黒 (0,0,0) が背景。Casper（gen-omnimatte-public）
    の `mask_*.mp4` フォーマットに合わせる。base video と同じ解像度・fps で
    エンコードする前提。
    """
    if masks.ndim != 3:
        raise ValueError(f"masks must be 3D (T, H, W), got shape {masks.shape}")
    t, height, width = masks.shape
    if t == 0:
        raise ValueError("masks has zero frames")

    enc_w = width + (width % 2)
    enc_h = height + (height % 2)

    tmpdir = tempfile.mkdtemp(prefix="omnimatte_mask_")
    try:
        for i in range(t):
            mono = (masks[i].astype(np.uint8)) * 255
            frame = np.stack([mono, mono, mono], axis=-1)  # (H, W, 3) BGR (gray は同値なのでOK)
            if enc_w != width or enc_h != height:
                frame = cv2.copyMakeBorder(
                    frame, 0, enc_h - height, 0, enc_w - width,
                    cv2.BORDER_CONSTANT, value=0,
                )
            cv2.imwrite(os.path.join(tmpdir, f"frame_{i:06d}.png"), frame)

        safe_fps = max(1.0, float(fps))
        cmd = [
            imageio_ffmpeg.get_ffmpeg_exe(), "-y",
            "-framerate", str(safe_fps),
            "-i", os.path.join(tmpdir, "frame_%06d.png"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "18",
            "-preset", "fast",
            out_path,
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg mask encoding failed:\n{result.stderr.decode(errors='replace')}"
            )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _normalize_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    """bool / uint8 のマスクを (height, width) の bool に揃える。"""
    if mask.dtype == bool:
        arr = mask
    elif mask.max() <= 1:
        arr = mask.astype(bool)
    else:
        arr = mask.astype(np.uint8) > 127
    if arr.shape != (height, width):
        arr = cv2.resize(arr.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
    return arr
