"""SAM2 video predictor wrapper.

`sam2` パッケージ(pip install 経由)の `Sam2VideoPredictor` をラップする。
細粒度の reset / add_prompt / propagate ではなく、タスク単位の高位 API
(`segment_from_bbox` / `segment_from_bboxes`) を公開する。
prompt はすべて bbox 経由（SAM2 が bbox から内部で再セグメントする方が
境界が綺麗）に統一し、reset 忘れによる prompt・memory bank 混入の事故を
呼び出し側に起こさせない設計。

アプリ全体で同時に存在する SAM2 セッションは常に最大 1 件なので、`inference_state`
はこのクラス内部に持ち、呼び出し側には不透明ハンドルとして渡さない。
"""
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

    def segment_from_bbox(
        self,
        frame_idx: int,
        bbox: list[float],
    ) -> np.ndarray:
        """単一 bbox 版。`segment_from_bboxes` の薄いラッパー。

        戻り値: (T, H, W) bool。SAM2 が yield しなかったフレームは zero-fill。
        """
        return self.segment_from_bboxes([bbox], keyframe_idx=frame_idx)[0]

    def segment_from_bboxes(
        self,
        bboxes: list[list[float]],
        keyframe_idx: int,
    ) -> list[np.ndarray]:
        """各 bbox を登録 → 全フレームに順 + 逆方向 propagate。

        - 内部で reset → add_new_points_or_box × N → propagate を行い、最後にも reset する。
          後続呼び出し時に前回の prompt は残らない。
        - 戻り値: list of (T, H, W) bool。bbox 1 個につき 1 枚。
        """
        num_frames, height, width = self._dims()
        per_object: list[np.ndarray] = [
            np.zeros((num_frames, height, width), dtype=bool) for _ in bboxes
        ]
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            self._predictor.reset_state(self._state)
            for obj_id, bbox in enumerate(bboxes):
                box = np.asarray(bbox, dtype=np.float32)  # xyxy pixel
                self._predictor.add_new_points_or_box(
                    inference_state=self._state,
                    frame_idx=keyframe_idx,
                    obj_id=obj_id,
                    box=box,
                )
            self._collect_propagation(out_by_obj=per_object, start_frame_idx=keyframe_idx)
            self._predictor.reset_state(self._state)
        return per_object

    # ---------- 内部ヘルパー ----------

    def _collect_propagation(
        self,
        out_by_obj: list[np.ndarray],
        start_frame_idx: int,
    ) -> None:
        """順 + 逆方向に propagate し、各 obj_id ごとの (T, H, W) 出力に書き込む。

        呼び出し時点で autocast / inference_mode のコンテキストにいる前提。
        prompt は事前に登録済みであること。
        """
        for reverse in (False, True):
            for frame_idx, obj_ids, mask_logits in self._predictor.propagate_in_video(
                self._state, start_frame_idx=start_frame_idx, reverse=reverse,
            ):
                fi = int(frame_idx)
                for i, oid in enumerate(obj_ids):
                    out_by_obj[int(oid)][fi] = (mask_logits[i, 0] > 0.0).cpu().numpy()

    def _dims(self) -> tuple[int, int, int]:
        """現在の inference_state から (num_frames, height, width) を取り出す。

        SAM2 の inference_state dict のキー名に依存する。
        """
        s = self._state
        return int(s["num_frames"]), int(s["video_height"]), int(s["video_width"])


sam2 = Sam2()
