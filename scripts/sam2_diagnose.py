"""SAM2 を「実験スクリプトと同じ呼び出しパス」で動かすスタンドアロン診断ツール.

バックエンドの /segment 結果と直接比較するための切り分け用。
モデル差を排除するため、バックエンドと**同じ checkpoint・config**
（`server.model.SAM2_CFG` / `SAM2_CKPT`）を使う。呼び出し手順:

  - cv2.VideoCapture でフレーム抽出 → JPEG (q=95) 保存
  - predictor.init_state(jpeg_dir)
  - predictor.add_new_points_or_box(box=pixel_xyxy)
  - predictor.propagate_in_video()  順方向 + 逆方向
  - autocast / inference_mode の外側 wrap なし

これで「同じ動画 + 同じ keyframe + 同じ bbox」に対して、本スクリプトの
出力マスクとバックエンド /segment が返すマスクを並べて見れば、
- 一致 → バックエンドの SAM2 呼び出しパスは正しい。差は入力データ等にある。
- 不一致 → バックエンド側のどこか（_extract_full_foreground 等）に原因がある。

Usage:
    # backend ログから segment request の値を拾い、その動画パスをコピーしておく:
    #   "segment request: frame_idx=42 bbox=[120.0,80.0,640.0,520.0] ... base_video=C:\\..\\tmpXXX.mp4"
    python scripts\\sam2_diagnose.py --video C:\\path\\copied.mp4 --keyframe 42 --box 120,80,640,520
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# server.model から共通設定を取得（リポジトリルートを sys.path に追加）
_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root))
from server.model import SAM2_CFG, SAM2_CKPT, SAM2_DEVICE  # noqa: E402


def parse_box(s: str) -> np.ndarray:
    parts = [float(x) for x in s.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("box must be 'x0,y0,x1,y1'")
    return np.array(parts, dtype=np.float32)


def extract_frames_to_jpeg_dir(
    video_path: Path, out_dir: Path, jpeg_quality: int = 95,
) -> tuple[list[np.ndarray], float, int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_rgb: list[np.ndarray] = []
    idx = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        cv2.imwrite(
            str(out_dir / f"{idx:05d}.jpg"),
            frame_bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
        )
        frames_rgb.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        idx += 1
    cap.release()
    if idx == 0:
        raise RuntimeError(f"no frames extracted from: {video_path}")
    return frames_rgb, fps, W, H


def overlay(
    frame_rgb: np.ndarray,
    mask: np.ndarray | None,
    color: tuple[int, int, int] = (60, 180, 255),
    alpha: float = 0.55,
    box: np.ndarray | None = None,
) -> np.ndarray:
    out = frame_rgb.astype(np.float32)
    if mask is not None and mask.any():
        layer = np.zeros_like(out)
        layer[mask] = color
        a = alpha * mask[..., None].astype(np.float32)
        out = out * (1 - a) + layer * a
    out_u8 = out.astype(np.uint8)
    if box is not None:
        x0, y0, x1, y1 = (int(v) for v in box)
        cv2.rectangle(out_u8, (x0, y0), (x1, y1), (255, 255, 0), 2)
    return out_u8


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True, help="path to MP4 (typically backend's session.base_video_path)")
    p.add_argument("--keyframe", type=int, required=True, help="frame index where the box is given (matches backend req.frame_idx)")
    p.add_argument("--box", type=parse_box, required=True, help="xyxy pixel box: 'x0,y0,x1,y1' (matches backend req.bbox)")
    p.add_argument("--out", default="sam2_diagnose_out", help="output directory")
    args = p.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(video_path.resolve())

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"video    : {video_path}")
    print(f"keyframe : {args.keyframe}")
    print(f"box      : {args.box.tolist()}")
    print(f"SAM2_CFG : {SAM2_CFG}")
    print(f"SAM2_CKPT: {SAM2_CKPT}")
    print(f"out dir  : {out_dir}")

    tmp_root = Path(tempfile.mkdtemp(prefix="sam2_diagnose_"))
    frames_dir = tmp_root / "frames"
    try:
        print("\n[1] extracting frames to JPEG dir (cv2 + q=95)")
        frames_rgb, fps, W, H = extract_frames_to_jpeg_dir(video_path, frames_dir)
        n = len(frames_rgb)
        print(f"  frames={n}  size=({W},{H})  fps={fps:.2f}")
        if not (0 <= args.keyframe < n):
            raise ValueError(f"keyframe out of range [0, {n - 1}]")

        print("\n[2] building SAM2 with backend's CFG / CKPT")
        from sam2.build_sam import build_sam2_video_predictor
        predictor = build_sam2_video_predictor(SAM2_CFG, SAM2_CKPT, device=SAM2_DEVICE)

        print("\n[3] init_state on JPEG dir (no autocast wrap)")
        state = predictor.init_state(video_path=str(frames_dir))

        print("\n[4] add_new_points_or_box (box prompt direct, pixel xyxy)")
        predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=args.keyframe,
            obj_id=0,
            box=args.box,
        )

        print("\n[5] propagate forward + reverse")
        masks_per_frame: dict[int, np.ndarray] = {}
        for frame_idx, _, mask_logits in predictor.propagate_in_video(state):
            masks_per_frame[frame_idx] = (mask_logits[0, 0] > 0.0).cpu().numpy()
        for frame_idx, _, mask_logits in predictor.propagate_in_video(state, reverse=True):
            if frame_idx in masks_per_frame:
                continue
            masks_per_frame[frame_idx] = (mask_logits[0, 0] > 0.0).cpu().numpy()

        keyframe_mask = masks_per_frame[args.keyframe]
        print(f"\n[6] keyframe mask area: {int(keyframe_mask.sum())} px (of {H * W} total)")

        keyframe_overlay = overlay(frames_rgb[args.keyframe], keyframe_mask, box=args.box)
        keyframe_path = out_dir / "keyframe_mask.png"
        cv2.imwrite(str(keyframe_path), cv2.cvtColor(keyframe_overlay, cv2.COLOR_RGB2BGR))
        np.save(str(out_dir / "keyframe_mask.npy"), keyframe_mask)
        print(f"  saved: {keyframe_path}")
        print(f"  saved: {out_dir / 'keyframe_mask.npy'} (bool {H}x{W})")

        out_mp4 = out_dir / "overlay.mp4"
        print(f"\n[7] writing overlay video: {out_mp4}")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_mp4), fourcc, fps, (W, H))
        if not writer.isOpened():
            raise RuntimeError(f"cv2.VideoWriter failed to open: {out_mp4}")
        empty = np.zeros((H, W), dtype=bool)
        for i, frame_rgb in enumerate(tqdm(frames_rgb, desc="composing")):
            m = masks_per_frame.get(i, empty)
            box_to_show = args.box if i == args.keyframe else None
            ov = overlay(frame_rgb, m, box=box_to_show)
            writer.write(cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))
        writer.release()
        print(f"  saved: {out_mp4}")

        print(
            f"\nDone. Compare {keyframe_path} (and {out_mp4}) with the backend /segment "
            "output for the same video + keyframe + box."
        )
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
