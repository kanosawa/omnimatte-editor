"""動画全体に対する「全前景」マスク（R-CNN + SAM2 propagation 結果）を保持するストア。

`/session` 完了直後にバックグラウンドで:
  1. 中間フレームで Detectron2 (COCO Mask R-CNN) を実行
  2. 検出された各インスタンスを SAM2 video predictor に登録 → 全フレームに propagate
  3. 結果を per-object per-frame マスクとして保持

`/segment` で対象前景との IoU フィルタを行い、残りを「他の前景」として trimask に組み込む。
"""
import asyncio
import logging
from collections.abc import Awaitable
from dataclasses import dataclass
from enum import Enum, auto
import numpy as np


logger = logging.getLogger(__name__)


class _State(Enum):
    EMPTY = auto()    # 未開始（初期状態）
    LOADING = auto()  # submit 済み、完了待ち
    READY = auto()    # 成功完了
    FAILED = auto()   # 失敗完了


@dataclass
class FullForegroundRecord:
    object_masks: np.ndarray        # (N, T, H, W) bool。N = 検出物体数（0 のとき shape (0, T, H, W)）
    base_video_path: str            # マスク生成時点の base video パス（整合性チェック用）


class FullForegroundStore:
    def __init__(self) -> None:
        self._state: _State = _State.EMPTY
        self._record: FullForegroundRecord | None = None
        self._error: str | None = None
        self._ready_event: asyncio.Event = asyncio.Event()

    def submit(self, work: Awaitable[FullForegroundRecord]) -> asyncio.Task:
        """LOADING に同期的に遷移し、work を実行する Task を返す。

        work が Record を返せば READY、例外を上げれば FAILED に遷移する。
        """
        if self._state is not _State.EMPTY:
            logger.warning("submit called in state %s; resetting", self._state.name)
        self._state = _State.LOADING
        self._record = None
        self._error = None
        self._ready_event = asyncio.Event()
        return asyncio.create_task(self._run(work))

    async def _run(self, work: Awaitable[FullForegroundRecord]) -> None:
        try:
            record = await work
        except Exception as exc:
            logger.exception("full foreground extraction failed")
            self._error = str(exc)
            self._state = _State.FAILED
            self._ready_event.set()
            return
        self._record = record
        self._state = _State.READY
        self._ready_event.set()

    def current(self) -> FullForegroundRecord | None:
        return self._record

    async def wait_ready(self, timeout: float | None = None) -> None:
        if self._state is _State.READY:
            return
        if self._state is _State.FAILED:
            raise RuntimeError(f"full foreground extraction failed: {self._error}")
        if self._state is _State.EMPTY:
            raise RuntimeError("no full foreground extraction in progress")
        try:
            await asyncio.wait_for(self._ready_event.wait(), timeout=timeout)
        except asyncio.TimeoutError as exc:
            raise TimeoutError("full foreground extraction not ready (timeout)") from exc
        if self._state is _State.FAILED:
            raise RuntimeError(f"full foreground extraction failed: {self._error}")
