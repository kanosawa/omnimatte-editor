"""SAM2 backend 実装.

`vendor/sam2` の `Sam2VideoPredictor` をラップする。
`add_bbox_prompt` 内で SAM2 image predictor を `multimask_output=True` で実行し、
各候補マスクの tight bbox（true ピクセルの min/max x,y）が入力 bbox に最も近い
ものを採用する。bbox crop+upscale（案 A）は精度確保のため維持する。
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Iterator

import cv2
import numpy as np
import torch

from server.model import SAM2_CFG, SAM2_CKPT, SAM2_DEVICE
from server.sam_backend.base import ModelState, PropagateItem, SamBackend
from server.video_io import read_frame_at


logger = logging.getLogger(__name__)


# bbox 周辺をクロップする時のマージン（bbox サイズに対する比率）
_CROP_MARGIN_RATIO = 0.5
# クロップ後の画像を upscale する目標サイズ（長辺）。SAM2 内部は 1024 で動くため、
# bbox 周辺の effective 解像度を上げて detail を引き出す
_UPSCALE_TARGET_LONG_SIDE = 1024


class Sam2Backend(SamBackend):
    def __init__(self) -> None:
        self._state: ModelState = "loading"
        self._predictor = None
        self._image_predictor = None
        self._error: str | None = None
        self._ready_event = asyncio.Event()

    @property
    def state(self) -> ModelState:
        return self._state

    @property
    def error(self) -> str | None:
        return self._error

    @property
    def version(self):
        return "sam2"

    def _load_sync(self):
        from sam2.build_sam import build_sam2_video_predictor

        logger.info(
            "loading SAM2 model: cfg=%s ckpt=%s device=%s",
            SAM2_CFG, SAM2_CKPT, SAM2_DEVICE,
        )
        predictor = build_sam2_video_predictor(SAM2_CFG, SAM2_CKPT, device=SAM2_DEVICE)
        logger.info("SAM2 model loaded")
        return predictor

    async def load(self) -> None:
        try:
            self._predictor = await asyncio.to_thread(self._load_sync)
            self._state = "ready"
        except Exception as exc:
            logger.exception("SAM2 model load failed")
            self._error = str(exc)
            self._state = "failed"
        self._ready_event.set()

    async def wait_ready(self, timeout: float | None = None) -> None:
        if self._state == "ready":
            return
        if self._state == "failed":
            raise RuntimeError(f"model failed to load: {self._error}")
        try:
            await asyncio.wait_for(self._ready_event.wait(), timeout=timeout)
        except asyncio.TimeoutError as exc:
            raise TimeoutError("model not ready (timeout)") from exc
        if self._state == "failed":
            raise RuntimeError(f"model failed to load: {self._error}")

    # ---------- セッション ----------

    def _require_predictor(self):
        if self._predictor is None:
            raise RuntimeError("SAM2 predictor not loaded")
        return self._predictor

    def init_state(self, video_path: str) -> Any:
        predictor = self._require_predictor()
        with torch.inference_mode(), torch.autocast(SAM2_DEVICE, dtype=torch.bfloat16):
            return predictor.init_state(video_path=video_path)

    def reset_state(self, state: Any) -> None:
        predictor = self._require_predictor()
        with torch.inference_mode(), torch.autocast(SAM2_DEVICE, dtype=torch.bfloat16):
            predictor.reset_state(state)

    # ---------- マスク登録 ----------

    def add_mask(self, state: Any, frame_idx: int, obj_id: int, mask: np.ndarray) -> None:
        predictor = self._require_predictor()
        with torch.inference_mode(), torch.autocast(SAM2_DEVICE, dtype=torch.bfloat16):
            predictor.add_new_mask(
                inference_state=state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                mask=mask,
            )

    def add_bbox_prompt(
        self,
        state: Any,
        frame_idx: int,
        obj_id: int,
        bbox: list[float],
        base_video_path: str,
        height: int,
        width: int,
    ) -> np.ndarray:
        # 1. crop+upscale + image predictor (multimask) → tight-bbox IoU で候補選択
        mask = self._predict_initial_mask_for_bbox(
            base_video_path=base_video_path,
            frame_idx=frame_idx,
            bbox=bbox,
        )
        # 2. video predictor に登録（伝播の起点）
        self.add_mask(state=state, frame_idx=frame_idx, obj_id=obj_id, mask=mask)
        return mask

    # ---------- 伝播 ----------

    def propagate(
        self,
        state: Any,
        start_frame_idx: int,
        num_frames: int,  # SAM2 では未使用
        reverse: bool = False,
    ) -> Iterator[PropagateItem]:
        predictor = self._require_predictor()
        with torch.inference_mode(), torch.autocast(SAM2_DEVICE, dtype=torch.bfloat16):
            for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(
                state, start_frame_idx=start_frame_idx, reverse=reverse,
            ):
                # mask_logits: (N, 1, H, W) tensor
                masks = [
                    (mask_logits[i, 0] > 0.0).cpu().numpy() for i in range(len(obj_ids))
                ]
                yield int(frame_idx), [int(x) for x in obj_ids], masks

    # ---------- bbox → 初期マスク（案 A crop+upscale + multimask 候補選択）----------

    def _get_image_predictor(self):
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        # video predictor は SAM2Base のサブクラスなので、image predictor の
        # ベースモデルとして共有できる（GPU メモリの二重ロード回避）
        if self._image_predictor is None:
            self._image_predictor = SAM2ImagePredictor(self._require_predictor())
        return self._image_predictor

    def _predict_initial_mask_for_bbox(
        self,
        base_video_path: str,
        frame_idx: int,
        bbox: list[float],
    ) -> np.ndarray:
        """SAM2 image predictor の複数候補から tight bbox が入力 bbox に最も近いものを採用する。

        アルゴリズム:
          1. bbox 周辺 + マージンでフレームをクロップ
          2. クロップを長辺 1024 に upscale（cv2.INTER_CUBIC）
          3. 拡大後のクロップ + 変換した bbox で `multimask_output=True` を実行（3 候補）
          4. 各候補マスクの tight bbox（true ピクセルの min/max x,y）を計算
          5. tight bbox と入力 bbox の IoU が最大の候補を採用
          6. 最終マスクを元解像度にダウンスケールし、原座標に貼り戻す

        この基準は「サブコンポーネント（マスクが小さく bbox を満たさない）」
        「反転（マスクが背景全体を覆う）」「leakage（マスクが bbox からはみ出す）」
        の 3 ケースを 1 つの指標で同時に弾く。
        """
        image_predictor = self._get_image_predictor()

        frame_bgr = read_frame_at(base_video_path, frame_idx)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h_orig, w_orig = frame_rgb.shape[:2]

        # 1. bbox 周辺 + マージンでクロップ
        x1, y1, x2, y2 = [float(v) for v in bbox]
        bw = x2 - x1
        bh = y2 - y1
        cx1 = max(0, int(round(x1 - bw * _CROP_MARGIN_RATIO)))
        cy1 = max(0, int(round(y1 - bh * _CROP_MARGIN_RATIO)))
        cx2 = min(w_orig, int(round(x2 + bw * _CROP_MARGIN_RATIO)))
        cy2 = min(h_orig, int(round(y2 + bh * _CROP_MARGIN_RATIO)))
        if cx2 <= cx1 or cy2 <= cy1:
            raise RuntimeError(f"invalid crop region: ({cx1},{cy1})-({cx2},{cy2})")
        crop = frame_rgb[cy1:cy2, cx1:cx2]
        crop_h, crop_w = crop.shape[:2]

        # 2. 長辺 1024 に upscale（小さい場合のみ）
        long_side = max(crop_h, crop_w)
        if long_side < _UPSCALE_TARGET_LONG_SIDE:
            scale = _UPSCALE_TARGET_LONG_SIDE / long_side
            new_w = int(round(crop_w * scale))
            new_h = int(round(crop_h * scale))
            crop_proc = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        else:
            scale = 1.0
            crop_proc = crop
        proc_h, proc_w = crop_proc.shape[:2]

        # 3. bbox を crop+scale 座標系に変換（IoU 用に float のままクリップ）
        bbox_proc = np.array(
            [(x1 - cx1) * scale, (y1 - cy1) * scale, (x2 - cx1) * scale, (y2 - cy1) * scale],
            dtype=np.float32,
        )
        input_box = (
            max(0.0, float(bbox_proc[0])),
            max(0.0, float(bbox_proc[1])),
            min(float(proc_w), float(bbox_proc[2])),
            min(float(proc_h), float(bbox_proc[3])),
        )

        logger.info(
            "select: frame=%dx%d crop=%dx%d → proc=%dx%d (scale=%.2f) bbox_proc=(%.0f,%.0f,%.0f,%.0f)",
            w_orig, h_orig, crop_w, crop_h, proc_w, proc_h, scale,
            input_box[0], input_box[1], input_box[2], input_box[3],
        )

        with torch.inference_mode(), torch.autocast(SAM2_DEVICE, dtype=torch.bfloat16):
            image_predictor.set_image(crop_proc)
            # multimask_output=True で 3 候補を取得
            masks, scores, _ = image_predictor.predict(
                box=bbox_proc,
                multimask_output=True,
            )
            if len(masks) == 0:
                raise RuntimeError("SAM2 image predictor returned no mask")

        # 4-5. 各候補の tight bbox と入力 bbox の IoU を計算 → 最大の候補を選ぶ
        best_idx = -1
        best_iou = -1.0
        for i in range(len(masks)):
            m = masks[i].astype(bool)
            tb = _mask_tight_bbox(m)
            iou = _bbox_iou(input_box, tb) if tb is not None else 0.0
            score = float(scores[i]) if i < len(scores) else 0.0
            tb_str = "empty" if tb is None else f"({tb[0]:.0f},{tb[1]:.0f},{tb[2]:.0f},{tb[3]:.0f})"
            logger.info(
                "select: candidate %d sam_score=%.3f tight_bbox=%s bbox_iou=%.3f",
                i, score, tb_str, iou,
            )
            if iou > best_iou:
                best_iou = iou
                best_idx = i

        if best_idx < 0:
            raise RuntimeError("no valid mask candidate")
        mask_proc = masks[best_idx].astype(bool)
        logger.info("select: chose candidate %d (bbox_iou=%.3f)", best_idx, best_iou)

        # 6. 元解像度にダウンスケールして原座標に貼り戻す
        if scale != 1.0:
            mask_crop = cv2.resize(
                mask_proc.astype(np.uint8), (crop_w, crop_h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        else:
            mask_crop = mask_proc

        full_mask = np.zeros((h_orig, w_orig), dtype=bool)
        full_mask[cy1:cy2, cx1:cx2] = mask_crop
        return full_mask


def _mask_tight_bbox(mask: np.ndarray) -> tuple[float, float, float, float] | None:
    """マスクの true ピクセルの tight bbox `(xmin, ymin, xmax, ymax)`（exclusive max）。

    全 false なら None を返す。
    """
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    return (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))


def _bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """xyxy 形式 2 つの bbox の IoU（0..1）。"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    a_area = max(0.0, (ax2 - ax1) * (ay2 - ay1))
    b_area = max(0.0, (bx2 - bx1) * (by2 - by1))
    union = a_area + b_area - inter
    if union <= 0.0:
        return 0.0
    return float(inter / union)
