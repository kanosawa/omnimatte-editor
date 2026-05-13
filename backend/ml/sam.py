"""SAM2 video predictor wrapper.

`sam2` パッケージ(pip install 経由)の `Sam2VideoPredictor` をラップする。
`add_bbox_prompt` は video predictor の `add_new_points_or_box` に bbox を直接渡し、
返ってきた `video_res_masks`(元解像度 logits)を bool 化して返す。

アプリ全体で同時に存在する SAM2 セッションは常に最大 1 件なので、`inference_state`
はこのクラス内部に持ち、呼び出し側には不透明ハンドルとして渡さない。
"""
from __future__ import annotations

import asyncio
import logging
import os
import numpy as np
import torch
from typing import Any, Iterator

logger = logging.getLogger(__name__)


# モデル重みとアーキテクチャ定義は backend/scripts/setup.sh のダウンロード設定と
# 一対一で対応している。変更する場合は setup.sh も同時に更新すること。
_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAM2_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
SAM2_CKPT = os.path.join(_BACKEND_DIR, "models", "sam2", "sam2.1_hiera_large.pt")


PropagateItem = tuple[int, list[int], list[np.ndarray]]
"""propagate() の yield 単位。

- frame_idx: フレーム番号
- obj_ids: そのフレームで結果が得られた object_id のリスト
- masks: 各 obj_id に対応する bool[H, W] マスク（base video 解像度）
"""


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

    # ---------- セッション ----------

    def _require_predictor(self):
        if self._predictor is None:
            raise RuntimeError("SAM2 predictor not loaded")
        return self._predictor

    def _require_state(self) -> Any:
        if self._state is None:
            raise RuntimeError("SAM2 session not opened")
        return self._state

    def open_session(self, video_path: str) -> None:
        predictor = self._require_predictor()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            self._state = predictor.init_state(video_path=video_path)

    def reset(self) -> None:
        predictor = self._require_predictor()
        state = self._require_state()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            predictor.reset_state(state)

    # ---------- マスク登録 ----------

    def add_mask(self, frame_idx: int, obj_id: int, mask: np.ndarray) -> None:
        predictor = self._require_predictor()
        state = self._require_state()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            predictor.add_new_mask(
                inference_state=state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                mask=mask,
            )

    def add_bbox_prompt(
        self,
        frame_idx: int,
        obj_id: int,
        bbox: list[float],
        height: int,
        width: int,
    ) -> np.ndarray:
        predictor = self._require_predictor()
        state = self._require_state()
        box = np.asarray(bbox, dtype=np.float32)  # xyxy pixel
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
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
        start_frame_idx: int,
        reverse: bool = False,
    ) -> Iterator[PropagateItem]:
        predictor = self._require_predictor()
        state = self._require_state()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(
                state, start_frame_idx=start_frame_idx, reverse=reverse,
            ):
                # mask_logits: (N, 1, H, W) tensor
                masks = [
                    (mask_logits[i, 0] > 0.0).cpu().numpy() for i in range(len(obj_ids))
                ]
                yield int(frame_idx), [int(x) for x in obj_ids], masks


sam2 = Sam2()
