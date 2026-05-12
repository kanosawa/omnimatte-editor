"""動画全体に対する「全前景」マスク（R-CNN + SAM2 propagation 結果）を保持するストア。

`/session` 完了直後にバックグラウンドで:
  1. 中間フレームで Detectron2 (COCO Mask R-CNN) を実行
  2. 検出された各インスタンスを SAM2 video predictor に登録 → 全フレームに propagate
  3. 結果を per-object per-frame マスクとして保持

`/segment` で対象前景との IoU フィルタを行い、残りを「他の前景」として trimask に組み込む。
"""
import asyncio
import logging
import threading
import time
from dataclasses import dataclass
from typing import Literal

import numpy as np


logger = logging.getLogger(__name__)


FullFgState = Literal["empty", "loading", "ready", "failed"]


@dataclass
class FullForegroundRecord:
    per_object_masks: list[np.ndarray]  # 各要素 (T, H, W) bool。検出物体 1 つあたり 1 枚
    base_video_path: str                # マスク生成時点の base video パス（整合性チェック用）
    created_at: float


class FullForegroundStore:
    """単一スロット。`/session` 開始で `start_loading` → 完了で `set_ready` / `set_failed`。

    `/segment` は `wait_ready` で完了を待ってから読み出す。次の `/session` または
    `/remove` 完了で `clear` される。
    """

    def __init__(self) -> None:
        self._state: FullFgState = "empty"
        self._record: FullForegroundRecord | None = None
        self._error: str | None = None
        self._ready_event: asyncio.Event = asyncio.Event()
        self._lock = threading.Lock()

    @property
    def state(self) -> FullFgState:
        with self._lock:
            return self._state

    def start_loading(self) -> None:
        """`/session` 開始時に呼ぶ。スロットを loading 状態に初期化"""
        with self._lock:
            self._state = "loading"
            self._record = None
            self._error = None
        # asyncio.Event は loop に紐づくので新規作成
        self._ready_event = asyncio.Event()

    def set_ready(self, per_object_masks: list[np.ndarray], base_video_path: str) -> None:
        record = FullForegroundRecord(
            per_object_masks=per_object_masks,
            base_video_path=base_video_path,
            created_at=time.time(),
        )
        with self._lock:
            self._record = record
            self._state = "ready"
        self._ready_event.set()

    def set_failed(self, error: str) -> None:
        with self._lock:
            self._error = error
            self._state = "failed"
        self._ready_event.set()

    def clear(self) -> None:
        with self._lock:
            self._state = "empty"
            self._record = None
            self._error = None
        self._ready_event = asyncio.Event()

    def current(self) -> FullForegroundRecord | None:
        with self._lock:
            return self._record

    async def wait_ready(self, timeout: float | None = None) -> None:
        with self._lock:
            state = self._state
            error = self._error
        if state == "ready":
            return
        if state == "failed":
            raise RuntimeError(f"full foreground extraction failed: {error}")
        if state == "empty":
            raise RuntimeError("no full foreground extraction in progress")
        try:
            await asyncio.wait_for(self._ready_event.wait(), timeout=timeout)
        except asyncio.TimeoutError as exc:
            raise TimeoutError("full foreground extraction not ready (timeout)") from exc
        with self._lock:
            state = self._state
            error = self._error
        if state == "failed":
            raise RuntimeError(f"full foreground extraction failed: {error}")


full_foreground_store = FullForegroundStore()
