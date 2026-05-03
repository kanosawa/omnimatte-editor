"""SAM2 backend 実装.

`vendor/sam2` の `Sam2VideoPredictor` をラップする。
`add_bbox_prompt` 内で SAM2 image predictor による bbox crop+upscale + 反復補正
（旧 `server/routes/segment.py::_refine_mask_iteratively_for_bbox`）を行う。
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


# 反復補正の閾値: bbox 内に対するマスク面積の比がこれ未満なら
# 「サブコンポーネント疑い」とみなして positive point を追加して再予測する
_REFINE_FILL_RATIO_THRESHOLD = 0.5
# 追加する positive point の最大数
_REFINE_MAX_POINTS = 3
# 追加候補とする「bbox 内かつマスク外」の連結成分の最小面積（bbox 面積比）
_REFINE_MIN_COMPONENT_AREA_RATIO = 0.02

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
        # 1. crop+upscale + image predictor で初期マスクを得る
        mask = self._refine_mask_iteratively_for_bbox(
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

    # ---------- bbox refinement (案 Y + 案 A) ----------

    def _get_image_predictor(self):
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        # video predictor は SAM2Base のサブクラスなので、image predictor の
        # ベースモデルとして共有できる（GPU メモリの二重ロード回避）
        if self._image_predictor is None:
            self._image_predictor = SAM2ImagePredictor(self._require_predictor())
        return self._image_predictor

    def _refine_mask_iteratively_for_bbox(
        self,
        base_video_path: str,
        frame_idx: int,
        bbox: list[float],
    ) -> np.ndarray:
        """SAM2 画像プレディクタで反復補正してマスクを得る（案 Y + 案 A: bbox crop + upscale）。

        アルゴリズム:
          1. bbox 周辺 + マージンでフレームをクロップ
          2. クロップを長辺 1024 に upscale（cv2.INTER_CUBIC）
          3. 拡大後のクロップ + 変換した bbox で SAM2 image predictor を実行 → M0
          4. M0 が bbox を埋める比率（fill_ratio）を計算
          5. fill_ratio >= threshold なら M0 を採用
          6. それ以外は「bbox 内かつ M0 外」の連結成分の重心を positive point として
             追加し再予測 → M1。M1 が M0 より良ければ M1、そうでなければ M0
          7. 最終マスクを元解像度にダウンスケールし、原座標に貼り戻す
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

        # 3. bbox を crop+scale 座標系に変換
        bbox_np = np.array(
            [(x1 - cx1) * scale, (y1 - cy1) * scale, (x2 - cx1) * scale, (y2 - cy1) * scale],
            dtype=np.float32,
        )
        bx1 = max(0, int(round(bbox_np[0])))
        by1 = max(0, int(round(bbox_np[1])))
        bx2 = min(proc_w, int(round(bbox_np[2])))
        by2 = min(proc_h, int(round(bbox_np[3])))
        bbox_clipped_proc = (bx1, by1, bx2, by2)
        bbox_area = max(1, (bx2 - bx1) * (by2 - by1))

        logger.info(
            "refine: frame=%dx%d crop=%dx%d (margin=%.1f) → proc=%dx%d (scale=%.2f)",
            w_orig, h_orig, crop_w, crop_h, _CROP_MARGIN_RATIO, proc_w, proc_h, scale,
        )

        with torch.inference_mode(), torch.autocast(SAM2_DEVICE, dtype=torch.bfloat16):
            image_predictor.set_image(crop_proc)

            # 4. 初回予測（bbox のみ、SAM2 既定の選択に任せる）
            masks_0, _scores_0, _ = image_predictor.predict(
                box=bbox_np,
                multimask_output=False,
            )
            if len(masks_0) == 0:
                raise RuntimeError("SAM2 image predictor returned no mask")
            mask_0_proc = masks_0[0].astype(bool)

            in_bbox_0 = int(mask_0_proc[by1:by2, bx1:bx2].sum())
            ratio_0 = in_bbox_0 / bbox_area
            logger.info(
                "refine: initial fill_ratio=%.3f (threshold=%.3f)",
                ratio_0, _REFINE_FILL_RATIO_THRESHOLD,
            )

            if ratio_0 >= _REFINE_FILL_RATIO_THRESHOLD:
                mask_proc = mask_0_proc
            else:
                # 5. サブコンポーネント疑い → 追加 positive point を探す
                refine_points = _find_refinement_points(
                    mask_0_proc, bbox_clipped_proc, bbox_area,
                )
                if not refine_points:
                    logger.info("refine: no refinement points found, using initial mask")
                    mask_proc = mask_0_proc
                else:
                    point_coords = np.array(refine_points, dtype=np.float32)
                    point_labels = np.ones(len(refine_points), dtype=np.int32)
                    logger.info(
                        "refine: re-predicting with %d positive points: %s",
                        len(refine_points),
                        [(f"{p[0]:.0f}", f"{p[1]:.0f}") for p in refine_points],
                    )
                    masks_1, _scores_1, _ = image_predictor.predict(
                        box=bbox_np,
                        point_coords=point_coords,
                        point_labels=point_labels,
                        multimask_output=False,
                    )
                    mask_1_proc = masks_1[0].astype(bool)

                    in_bbox_1 = int(mask_1_proc[by1:by2, bx1:bx2].sum())
                    ratio_1 = in_bbox_1 / bbox_area
                    logger.info("refine: refined fill_ratio=%.3f", ratio_1)

                    if ratio_1 < ratio_0:
                        logger.info(
                            "refine: refinement reduced fill_ratio (%.3f -> %.3f), falling back",
                            ratio_0, ratio_1,
                        )
                        mask_proc = mask_0_proc
                    else:
                        mask_proc = mask_1_proc

        # 7. 元解像度にダウンスケールして原座標に貼り戻す
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


def _find_refinement_points(
    mask: np.ndarray,
    bbox_clipped: tuple[int, int, int, int],
    bbox_area: int,
) -> list[tuple[float, float]]:
    """「bbox 内かつ mask 外」の領域から positive point の候補を抽出する。

    アルゴリズム:
      1. (in_bbox AND not mask) のバイナリを作る
      2. 連結成分に分割
      3. 面積上位、かつ bbox 面積比 >= 閾値の成分を最大 N 個選ぶ
      4. 各成分の重心を point として返す。重心が成分外（凹形状）なら成分内の
         代表ピクセルにフォールバック

    返り値: `[(x, y), ...]` のピクセル座標リスト
    """
    h, w = mask.shape
    x1, y1, x2, y2 = bbox_clipped

    bbox_region = np.zeros_like(mask, dtype=np.uint8)
    bbox_region[y1:y2, x1:x2] = 1
    candidate = bbox_region & (~mask).astype(np.uint8)
    if candidate.sum() == 0:
        return []

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        candidate, connectivity=8,
    )
    if n_labels <= 1:
        return []

    min_area = max(1, int(_REFINE_MIN_COMPONENT_AREA_RATIO * bbox_area))

    components = sorted(
        ((int(stats[i, cv2.CC_STAT_AREA]), i) for i in range(1, n_labels)),
        reverse=True,
    )

    points: list[tuple[float, float]] = []
    for area, idx in components:
        if area < min_area:
            break
        cx, cy = centroids[idx]
        cx_int = int(round(cx))
        cy_int = int(round(cy))
        if 0 <= cy_int < h and 0 <= cx_int < w and labels[cy_int, cx_int] == idx:
            points.append((float(cx), float(cy)))
        else:
            ys, xs = np.where(labels == idx)
            if len(xs) > 0:
                mid = len(xs) // 2
                points.append((float(xs[mid]), float(ys[mid])))
        if len(points) >= _REFINE_MAX_POINTS:
            break

    return points
