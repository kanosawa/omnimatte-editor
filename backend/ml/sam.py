from __future__ import annotations

import asyncio
import logging
import os
import numpy as np
import torch
from typing import Any

logger = logging.getLogger(__name__)


# モデル重みとアーキテクチャ定義は backend/scripts/setup.sh のダウンロード設定と
# 一対一で対応している。変更する場合は setup.sh も同時に更新すること。
_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAM2_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
SAM2_CKPT = os.path.join(_BACKEND_DIR, "models", "sam2", "sam2.1_hiera_large.pt")


class Sam2:
    def __init__(self) -> None:
        self._predictor = None
        self._state: Any = None
        self._error: str | None = None
        self._ready_event = asyncio.Event()

    def _load_sync(self):
        from sam2.build_sam import build_sam2_video_predictor

        logger.info(
            "loading SAM2 model: cfg=%s ckpt=%s device=cuda",
            SAM2_CFG, SAM2_CKPT,
        )
        predictor = build_sam2_video_predictor(SAM2_CFG, SAM2_CKPT, device="cuda")
        logger.info("SAM2 model loaded")
        return predictor

    async def load(self) -> None:
        try:
            self._predictor = await asyncio.to_thread(self._load_sync)
        except Exception as exc:
            logger.exception("SAM2 model load failed")
            self._error = str(exc)
        self._ready_event.set()

    async def wait_ready(self, timeout: float | None = None) -> None:
        if self._ready_event.is_set():
            if self._error is not None:
                raise RuntimeError(f"model failed to load: {self._error}")
            return
        try:
            await asyncio.wait_for(self._ready_event.wait(), timeout=timeout)
        except asyncio.TimeoutError as exc:
            raise TimeoutError("model not ready (timeout)") from exc
        if self._error is not None:
            raise RuntimeError(f"model failed to load: {self._error}")

    def open_session(self, video_path: str) -> None:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            self._state = self._predictor.init_state(video_path=video_path)

    def segment_from_bboxes(
        self,
        bboxes: list[list[float]],
        keyframe_idx: int,
    ) -> list[np.ndarray]:
        
        num_frames, height, width = self._dims()
        object_masks: list[np.ndarray] = [
            np.zeros((num_frames, height, width), dtype=bool) for _ in bboxes
        ]
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            self._predictor.reset_state(self._state)
            for obj_id, bbox in enumerate(bboxes):
                self._predictor.add_new_points_or_box(
                    inference_state=self._state,
                    frame_idx=keyframe_idx,
                    obj_id=obj_id,
                    box=bbox,
                )
            for reverse in (False, True):
                for frame_idx, obj_ids, mask_logits in self._predictor.propagate_in_video(
                    self._state, start_frame_idx=keyframe_idx, reverse=reverse,
                ):
                    for i, oid in enumerate(obj_ids):
                        object_masks[oid][frame_idx] = (mask_logits[i, 0] > 0.0).cpu().numpy()
            self._predictor.reset_state(self._state)
        return object_masks

    def _dims(self) -> tuple[int, int, int]:
        s = self._state
        return int(s["num_frames"]), int(s["video_height"]), int(s["video_width"])


sam2 = Sam2()
