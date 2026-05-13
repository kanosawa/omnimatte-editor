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
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np


logger = logging.getLogger(__name__)


class _State(Enum):
    EMPTY = auto()    # 未開始（初期状態）
    LOADING = auto()  # start_loading 済み、完了待ち
    READY = auto()    # set_ready で完了
    FAILED = auto()   # set_failed で失敗


@dataclass
class FullForegroundRecord:
    object_masks: np.ndarray        # (N, T, H, W) bool。N = 検出物体数（0 のとき shape (0, T, H, W)）
    base_video_path: str            # マスク生成時点の base video パス（整合性チェック用）


class FullForegroundStore:
    """per-session スロット。`Session` が 1 個ずつ所有する。

    `/session` 開始で `start_loading` → 完了で `set_ready` / `set_failed`。
    `/segment` は `wait_ready` で完了を待ってから読み出す。新しい `/session` /
    `/remove` 時には旧 Session ごと GC されるので、明示的な clear API は提供しない。
    """

    def __init__(self) -> None:
        self._state: _State = _State.EMPTY
        self._record: FullForegroundRecord | None = None
        self._error: str | None = None
        self._ready_event: asyncio.Event = asyncio.Event()
        self._lock = threading.Lock()

    def start_loading(self) -> None:
        """`/session` 開始時に呼ぶ。スロットを loading 状態に初期化"""
        # asyncio.Event は loop に紐づくので新規作成
        new_event = asyncio.Event()
        with self._lock:
            if self._state is not _State.EMPTY:
                logger.warning("start_loading called in state %s; resetting", self._state.name)
            self._state = _State.LOADING
            self._record = None
            self._error = None
            self._ready_event = new_event

    def set_ready(self, object_masks: np.ndarray, base_video_path: str) -> None:
        with self._lock:
            if self._state is not _State.LOADING:
                logger.warning("set_ready called in state %s; ignoring", self._state.name)
                return
            self._record = FullForegroundRecord(
                object_masks=object_masks,
                base_video_path=base_video_path,
            )
            self._state = _State.READY
            event = self._ready_event
        event.set()

    def set_failed(self, error: str) -> None:
        with self._lock:
            if self._state is not _State.LOADING:
                logger.warning("set_failed called in state %s; ignoring", self._state.name)
                return
            self._error = error
            self._state = _State.FAILED
            event = self._ready_event
        event.set()

    def current(self) -> FullForegroundRecord | None:
        with self._lock:
            return self._record

    async def wait_ready(self, timeout: float | None = None) -> None:
        with self._lock:
            state = self._state
            error = self._error
            event = self._ready_event
        if state is _State.READY:
            return
        if state is _State.FAILED:
            raise RuntimeError(f"full foreground extraction failed: {error}")
        if state is _State.EMPTY:
            raise RuntimeError("no full foreground extraction in progress")
        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
        except asyncio.TimeoutError as exc:
            raise TimeoutError("full foreground extraction not ready (timeout)") from exc
        with self._lock:
            state = self._state
            error = self._error
        if state is _State.READY:
            return
        if state is _State.FAILED:
            raise RuntimeError(f"full foreground extraction failed: {error}")
        # 不変条件: ready_event が set された時点で READY/FAILED のいずれか
        raise RuntimeError(f"full foreground extraction in unexpected state: {state.name}")
