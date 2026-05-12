"""Detectron2 (COCO Mask R-CNN) を class-agnostic で使う物体検出ホルダ。

クラスラベルは無視し、インスタンスマスクのみを返す。SAM2 への propagation 用に
中間フレームでの 1 ショット推論で使う。
"""
import asyncio
import logging
from typing import Literal

import numpy as np

from backend.model import (
    DETECTRON2_CONFIG,
    DETECTRON2_DEVICE,
    DETECTRON2_MAX_DETECTIONS,
    DETECTRON2_MIN_AREA_RATIO,
    DETECTRON2_SCORE_THRESH,
)


logger = logging.getLogger(__name__)


DetectorState = Literal["loading", "ready", "failed"]


class Detectron2:
    """Detectron2 の DefaultPredictor をプリロードして保持する。

    `/health` で `detector_state` として状態を返す。
    """

    def __init__(self) -> None:
        self._state: DetectorState = "loading"
        self._predictor = None
        self._error: str | None = None
        self._ready_event = asyncio.Event()

    @property
    def state(self) -> DetectorState:
        return self._state

    @property
    def predictor(self):
        return self._predictor

    @property
    def error(self) -> str | None:
        return self._error

    def _load_sync(self):
        from detectron2 import model_zoo
        from detectron2.config import get_cfg
        from detectron2.engine import DefaultPredictor

        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(DETECTRON2_CONFIG))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(DETECTRON2_CONFIG)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = DETECTRON2_SCORE_THRESH
        cfg.MODEL.DEVICE = DETECTRON2_DEVICE
        logger.info("loading Detectron2: %s device=%s", DETECTRON2_CONFIG, DETECTRON2_DEVICE)
        predictor = DefaultPredictor(cfg)
        logger.info("Detectron2 model loaded")
        return predictor

    async def load(self) -> None:
        try:
            self._predictor = await asyncio.to_thread(self._load_sync)
            self._state = "ready"
        except Exception as exc:
            logger.exception("Detectron2 load failed")
            self._error = str(exc)
            self._state = "failed"
        self._ready_event.set()

    async def wait_ready(self, timeout: float | None = None) -> None:
        if self._state == "ready":
            return
        if self._state == "failed":
            raise RuntimeError(f"detector failed to load: {self._error}")
        try:
            await asyncio.wait_for(self._ready_event.wait(), timeout=timeout)
        except asyncio.TimeoutError as exc:
            raise TimeoutError("detector not ready (timeout)") from exc
        if self._state == "failed":
            raise RuntimeError(f"detector failed to load: {self._error}")

    def detect(self, frame_bgr: np.ndarray) -> list[np.ndarray]:
        """1 フレームに対して COCO Mask R-CNN を走らせ、class-agnostic にインスタンスマスクを返す。

        フィルタ:
        - score < DETECTRON2_SCORE_THRESH（cfg 設定で既に効いている）
        - 面積 < DETECTRON2_MIN_AREA_RATIO * H * W は除外
        - area 降順で上位 DETECTRON2_MAX_DETECTIONS 個を残す

        Returns:
            list of (H, W) bool マスク。area 降順
        """
        if self._predictor is None:
            raise RuntimeError("detector predictor not loaded")
        outputs = self._predictor(frame_bgr)
        instances = outputs["instances"]
        if len(instances) == 0:
            return []

        masks = instances.pred_masks.cpu().numpy()  # (N, H, W) bool
        areas = masks.sum(axis=(1, 2))              # (N,)
        h, w = masks.shape[1], masks.shape[2]
        min_area = DETECTRON2_MIN_AREA_RATIO * h * w

        order = np.argsort(-areas)  # area 降順
        result: list[np.ndarray] = []
        for idx in order:
            if areas[idx] < min_area:
                continue
            result.append(masks[idx].astype(bool))
            if len(result) >= DETECTRON2_MAX_DETECTIONS:
                break
        return result


detectron2 = Detectron2()
