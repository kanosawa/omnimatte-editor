import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any


logger = logging.getLogger(__name__)


@dataclass
class Session:
    inference_state: Any
    video_path: str
    width: int
    height: int
    fps: float
    num_frames: int
    created_at: float


class SessionSlot:
    """常に最大 1 件のセッションだけを保持するスロット。

    新しいセッションを `replace` で投入すると、直前のセッションは
    自動的に破棄され、関連する一時ファイルも削除される。
    """

    def __init__(self) -> None:
        self._current: Session | None = None
        self._lock = threading.Lock()

    def replace(
        self,
        inference_state: Any,
        video_path: str,
        width: int,
        height: int,
        fps: float,
        num_frames: int,
    ) -> Session:
        new_session = Session(
            inference_state=inference_state,
            video_path=video_path,
            width=width,
            height=height,
            fps=fps,
            num_frames=num_frames,
            created_at=time.time(),
        )
        with self._lock:
            old = self._current
            self._current = new_session
        if old is not None:
            self._cleanup(old)
        return new_session

    def current(self) -> Session | None:
        with self._lock:
            return self._current

    def is_active(self) -> bool:
        with self._lock:
            return self._current is not None

    @staticmethod
    def _cleanup(session: Session) -> None:
        try:
            if session.video_path and os.path.exists(session.video_path):
                os.unlink(session.video_path)
        except OSError:
            logger.warning("failed to remove video file: %s", session.video_path)


session_slot = SessionSlot()
