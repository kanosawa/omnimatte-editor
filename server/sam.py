"""SAM2 video predictor wrapper.

`sam2` パッケージ（pip install 経由）の `Sam2VideoPredictor` をラップする。
`add_bbox_prompt` は video predictor の `add_new_points_or_box` に bbox を直接渡し、
返ってきた `video_res_masks`（元解像度 logits）を bool 化して返す。SAM2 が想定する
正規の box prompt パスを使うことで、prompt embedding が memory bank に正しく入る。

各メソッドは `torch.inference_mode()` + `torch.autocast(bf16)` で囲む。
SAM2 公式の notebook と同じ設定で、Ampere 以降の GPU で bf16 Tensor Core を使い
matmul を約 2 倍高速化する（image encoder / memory attention / propagate の支配的コスト）。
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Iterator, Literal

import numpy as np
import torch

from server.model import SAM2_CFG, SAM2_CKPT, SAM2_DEVICE


logger = logging.getLogger(__name__)


ModelState = Literal["loading", "ready", "failed"]


PropagateItem = tuple[int, list[int], list[np.ndarray]]
"""propagate() の yield 単位。

- frame_idx: フレーム番号
- obj_ids: そのフレームで結果が得られた object_id のリスト
- masks: 各 obj_id に対応する bool[H, W] マスク（base video 解像度）
"""


class Sam2:
    def __init__(self) -> None:
        self._state: ModelState = "loading"
        self._predictor = None
        self._error: str | None = None
        self._ready_event = asyncio.Event()

    @property
    def state(self) -> ModelState:
        return self._state

    @property
    def error(self) -> str | None:
        return self._error

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
        base_video_path: str,  # 直接 box を渡すので未使用
        height: int,
        width: int,
    ) -> np.ndarray:
        predictor = self._require_predictor()
        box = np.asarray(bbox, dtype=np.float32)  # xyxy pixel
        with torch.inference_mode(), torch.autocast(SAM2_DEVICE, dtype=torch.bfloat16):
            _, out_obj_ids, video_res_masks = predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                box=box,
            )
        # video_res_masks: tensor (N, 1, H, W) の logits（base video 解像度）
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


sam2 = Sam2()
