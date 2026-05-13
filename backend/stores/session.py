import logging
import os
import threading
import time
from dataclasses import dataclass


logger = logging.getLogger(__name__)


@dataclass
class Session:
    base_video_path: str
    width: int
    height: int
    fps: float
    num_frames: int
    created_at: float


class SessionSlot:
    """常に最大 1 件のセッションだけを保持するスロット。

    新しいセッションを `replace` で投入すると、直前のセッションは
    自動的に破棄され、関連する一時ファイルも削除される。

    `swap_base_video` は同一セッションの base video（とメタ情報）を差し替える。
    `/remove` 完了時にカスケード（前景削除済み動画 → 次の base video）を
    実現するために使う。SAM2 の `inference_state` は `Sam2` クラスが内部で
    保持するので、ここでは扱わない（呼び出し側で `sam2.open_session()` を
    別途呼ぶこと）。
    """

    def __init__(self) -> None:
        self._current: Session | None = None
        self._lock = threading.Lock()

    def replace(
        self,
        base_video_path: str,
        width: int,
        height: int,
        fps: float,
        num_frames: int,
    ) -> Session:
        new_session = Session(
            base_video_path=base_video_path,
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

    def swap_base_video(
        self,
        new_base_video_path: str,
        width: int,
        height: int,
        fps: float,
        num_frames: int,
    ) -> Session:
        """ベース動画を差し替える（`/remove` 完了時に使う）。

        旧 base video のファイルを削除する。セッションそのものは維持される
        （`session_id` などは元から無いが、フロントから見て「同じセッションの
        続き」として扱われる）。呼び出し側で `sam2.open_session()` と
        `MaskStore.clear()` を別途呼ぶこと。
        """
        old_path: str | None = None
        with self._lock:
            if self._current is None:
                raise RuntimeError("no active session to swap base video")
            old_path = self._current.base_video_path
            self._current = Session(
                base_video_path=new_base_video_path,
                width=width,
                height=height,
                fps=fps,
                num_frames=num_frames,
                created_at=time.time(),
            )
            new_session = self._current
        if old_path and old_path != new_base_video_path and os.path.exists(old_path):
            try:
                os.unlink(old_path)
            except OSError:
                logger.warning("failed to remove old base video file: %s", old_path)
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
            if session.base_video_path and os.path.exists(session.base_video_path):
                os.unlink(session.base_video_path)
        except OSError:
            logger.warning("failed to remove video file: %s", session.base_video_path)


session_slot = SessionSlot()
