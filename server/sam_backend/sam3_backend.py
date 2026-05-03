"""SAM3 backend 実装.

`facebookresearch/sam3` の `Sam3Tracker`（`build_sam3_video_model().tracker`）を
ラップする。SAM3 video predictor は SAM2 と概ね同じ API（`init_state`,
`add_new_points_or_box`, `add_new_mask`, `propagate_in_video`,
`clear_all_points_in_video`）を提供するが、以下が異なる:

- 座標系: 既定で normalized [0, 1]。`rel_coordinates=False` で pixel 切替可
- `propagate_in_video` の戻り値が 5-tuple、`max_frame_num_to_track` が必須
- `reset_state` は無く `clear_all_points_in_video` を使う
- image predictor はテキスト/コンセプト前提（bbox 用ではない）→ SAM3 backend では
  `add_bbox_prompt` で video predictor に直接 bbox を渡し、SAM2 で行っていた
  crop+upscale + 反復補正は不要（SAM3 は低解像度に強い前提）
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Iterator

import numpy as np
import torch

from server.model import SAM3_CKPT, SAM3_DEVICE
from server.sam_backend.base import ModelState, PropagateItem, SamBackend


logger = logging.getLogger(__name__)


class Sam3Backend(SamBackend):
    def __init__(self) -> None:
        self._state: ModelState = "loading"
        self._predictor = None  # tracker
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
        return "sam3"

    def _load_sync(self):
        from sam3.model_builder import build_sam3_video_model

        logger.info("loading SAM3 model: ckpt=%s device=%s", SAM3_CKPT, SAM3_DEVICE)
        sam3_model = build_sam3_video_model(checkpoint_path=SAM3_CKPT)
        # tracker と detector は backbone を共有する（メモリ削減）
        predictor = sam3_model.tracker
        predictor.backbone = sam3_model.detector.backbone
        logger.info("SAM3 model loaded")
        return predictor

    async def load(self) -> None:
        try:
            self._predictor = await asyncio.to_thread(self._load_sync)
            self._state = "ready"
        except Exception as exc:
            logger.exception("SAM3 model load failed")
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
            raise RuntimeError("SAM3 predictor not loaded")
        return self._predictor

    def init_state(self, video_path: str) -> Any:
        predictor = self._require_predictor()
        with torch.inference_mode(), torch.autocast(SAM3_DEVICE, dtype=torch.bfloat16):
            return predictor.init_state(video_path=video_path)

    def reset_state(self, state: Any) -> None:
        predictor = self._require_predictor()
        with torch.inference_mode(), torch.autocast(SAM3_DEVICE, dtype=torch.bfloat16):
            predictor.clear_all_points_in_video(state)

    # ---------- マスク登録 ----------

    def add_mask(self, state: Any, frame_idx: int, obj_id: int, mask: np.ndarray) -> None:
        predictor = self._require_predictor()
        with torch.inference_mode(), torch.autocast(SAM3_DEVICE, dtype=torch.bfloat16):
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
        base_video_path: str,  # SAM3 では使わない（image predictor refinement なし）
        height: int,
        width: int,
    ) -> np.ndarray:
        predictor = self._require_predictor()
        x1, y1, x2, y2 = [float(v) for v in bbox]
        # SAM2 と仕様統一のため、コードは pixel 座標で受けて normalized に変換する
        rel_box = np.array(
            [[x1 / width, y1 / height, x2 / width, y2 / height]], dtype=np.float32,
        )
        with torch.inference_mode(), torch.autocast(SAM3_DEVICE, dtype=torch.bfloat16):
            _, out_obj_ids, _low_res_masks, video_res_masks = predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                box=rel_box,
            )
        # video_res_masks: tensor (N, 1, H, W) または (N, H, W) の logits
        # obj_id に対応する index を取り出す
        try:
            i = list(int(x) for x in out_obj_ids).index(int(obj_id))
        except ValueError:
            i = 0
        m = video_res_masks[i]
        if m.ndim == 3:
            m = m[0]
        return (m > 0.0).cpu().numpy().astype(bool)

    # ---------- 伝播 ----------

    def propagate(
        self,
        state: Any,
        start_frame_idx: int,
        num_frames: int,
        reverse: bool = False,
    ) -> Iterator[PropagateItem]:
        predictor = self._require_predictor()
        # SAM3 は max_frame_num_to_track 必須。動画の全フレームをカバーする値を渡す
        max_track = max(1, int(num_frames))
        with torch.inference_mode(), torch.autocast(SAM3_DEVICE, dtype=torch.bfloat16):
            for (
                frame_idx, obj_ids, _low_res, video_res_masks, _scores,
            ) in predictor.propagate_in_video(
                state,
                start_frame_idx=start_frame_idx,
                max_frame_num_to_track=max_track,
                reverse=reverse,
                propagate_preflight=True,
            ):
                masks: list[np.ndarray] = []
                for i in range(len(obj_ids)):
                    m = video_res_masks[i]
                    if m.ndim == 3:
                        m = m[0]
                    masks.append((m > 0.0).cpu().numpy().astype(bool))
                yield int(frame_idx), [int(x) for x in obj_ids], masks
