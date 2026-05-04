import logging
import os
import shutil
import threading
import time
from dataclasses import dataclass
from typing import Any


logger = logging.getLogger(__name__)


@dataclass
class Session:
    inference_state: Any
    base_video_path: str
    sam_frames_dir: str | None  # SAM2 init_state 用に展開した JPEG ディレクトリ
    width: int
    height: int
    fps: float
    num_frames: int
    created_at: float


class SessionSlot:
    """常に最大 1 件のセッションだけを保持するスロット。

    新しいセッションを `replace` で投入すると、直前のセッションは
    自動的に破棄され、関連する一時ファイルも削除される。

    `swap_base_video` は同一セッションの base video（と inference_state /
    メタ情報）を差し替える。`/remove` 完了時にカスケード（前景削除済み動画
    → 次の base video）を実現するために使う。
    """

    def __init__(self) -> None:
        self._current: Session | None = None
        self._lock = threading.Lock()

    def replace(
        self,
        inference_state: Any,
        base_video_path: str,
        sam_frames_dir: str | None,
        width: int,
        height: int,
        fps: float,
        num_frames: int,
    ) -> Session:
        new_session = Session(
            inference_state=inference_state,
            base_video_path=base_video_path,
            sam_frames_dir=sam_frames_dir,
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
        new_inference_state: Any,
        new_sam_frames_dir: str | None,
        width: int,
        height: int,
        fps: float,
        num_frames: int,
    ) -> Session:
        """ベース動画を差し替える（`/remove` 完了時に使う）。

        旧 base video のファイルと SAM フレームディレクトリを削除し、
        `inference_state` を新値に置き換える。
        セッションそのものは維持される（`session_id` などは元から無いが、
        フロントから見て「同じセッションの続き」として扱われる）。
        呼び出し側で `MaskStore.clear()` を別途呼ぶこと。
        """
        old_path: str | None = None
        old_frames_dir: str | None = None
        with self._lock:
            if self._current is None:
                raise RuntimeError("no active session to swap base video")
            old_path = self._current.base_video_path
            old_frames_dir = self._current.sam_frames_dir
            self._current = Session(
                inference_state=new_inference_state,
                base_video_path=new_base_video_path,
                sam_frames_dir=new_sam_frames_dir,
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
        if (
            old_frames_dir
            and old_frames_dir != new_sam_frames_dir
            and os.path.isdir(old_frames_dir)
        ):
            shutil.rmtree(old_frames_dir, ignore_errors=True)
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
        if session.sam_frames_dir and os.path.isdir(session.sam_frames_dir):
            shutil.rmtree(session.sam_frames_dir, ignore_errors=True)


session_slot = SessionSlot()
