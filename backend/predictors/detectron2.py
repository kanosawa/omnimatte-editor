import asyncio
import logging
import numpy as np

from backend.config import (
    DETECTRON2_MAX_DETECTIONS,
    DETECTRON2_MIN_AREA_RATIO,
    DETECTRON2_SCORE_THRESH,
)


logger = logging.getLogger(__name__)

DETECTRON2_CONFIG = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"


class Detectron2:

    def __init__(self) -> None:
        self._predictor = None
        self._error: str | None = None
        self._ready_event = asyncio.Event()

    def _load_sync(self):
        from detectron2 import model_zoo
        from detectron2.config import get_cfg
        from detectron2.engine import DefaultPredictor

        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(DETECTRON2_CONFIG))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(DETECTRON2_CONFIG)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = DETECTRON2_SCORE_THRESH
        cfg.MODEL.DEVICE = "cuda"
        logger.info("loading Detectron2: %s device=cuda", DETECTRON2_CONFIG)
        predictor = DefaultPredictor(cfg)
        logger.info("Detectron2 model loaded")
        return predictor

    async def load(self) -> None:
        try:
            self._predictor = await asyncio.to_thread(self._load_sync)
        except Exception as exc:
            logger.exception("Detectron2 load failed")
            self._error = str(exc)
        self._ready_event.set()

    async def wait_ready(self, timeout: float | None = None) -> None:
        if self._ready_event.is_set():
            if self._error is not None:
                raise RuntimeError(f"detector failed to load: {self._error}")
            return
        try:
            await asyncio.wait_for(self._ready_event.wait(), timeout=timeout)
        except asyncio.TimeoutError as exc:
            raise TimeoutError("detector not ready (timeout)") from exc
        if self._error is not None:
            raise RuntimeError(f"detector failed to load: {self._error}")

    def detect(self, frame_bgr: np.ndarray) -> list[list[float]]:
        """1 フレームに対して COCO Mask R-CNN を走らせ、class-agnostic に bbox を返す。

        フィルタ:
        - score < DETECTRON2_SCORE_THRESH（cfg 設定で既に効いている）
        - マスク面積 < DETECTRON2_MIN_AREA_RATIO * H * W は除外
        - マスク面積降順で上位 DETECTRON2_MAX_DETECTIONS 個を残す

        Returns:
            list of [x1, y1, x2, y2] (xyxy pixel float)。マスク面積降順
        """
        if self._predictor is None:
            raise RuntimeError("detector predictor not loaded")
        outputs = self._predictor(frame_bgr)
        instances = outputs["instances"]
        if len(instances) == 0:
            return []

        masks = instances.pred_masks.cpu().numpy()           # (N, H, W) bool
        boxes = instances.pred_boxes.tensor.cpu().numpy()    # (N, 4) xyxy float
        areas = masks.sum(axis=(1, 2))                       # (N,)
        h, w = masks.shape[1], masks.shape[2]
        min_area = DETECTRON2_MIN_AREA_RATIO * h * w

        order = np.argsort(-areas)  # area 降順
        result: list[list[float]] = []
        for idx in order:
            if areas[idx] < min_area:
                continue
            result.append(boxes[idx].tolist())
            if len(result) >= DETECTRON2_MAX_DETECTIONS:
                break
        return result


detectron2 = Detectron2()
