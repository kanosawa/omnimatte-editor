import asyncio
import logging
import os
import threading
from dataclasses import dataclass, field

from backend.state.full_foreground import FullForegroundStore
from backend.state.mask import MaskStore


logger = logging.getLogger(__name__)


@dataclass
class Session:
    """1 本の base video に紐づく編集セッション。

    Session は自分の派生状態 (`mask_store`, `full_foreground_store`) と
    バックグラウンドタスクを所有する。`session_slot.install` で新しい Session に
    差し替えられると、旧 Session は dispose されて GC 対象になる。
    """
    base_video_path: str
    width: int
    height: int
    fps: float
    num_frames: int
    created_at: float
    mask_store: MaskStore = field(default_factory=MaskStore)
    full_foreground_store: FullForegroundStore = field(default_factory=FullForegroundStore)
    background_tasks: list[asyncio.Task] = field(default_factory=list)

    def add_task(self, task: asyncio.Task) -> None:
        """このセッションが起動したバックグラウンドタスクを記録する。"""
        self.background_tasks.append(task)

    async def wait_for_tasks(self) -> None:
        """記録済みのバックグラウンドタスクの完了を待つ。

        新セッションへの差し替え前に呼んで、旧 Session の GPU 処理が
        新 Session と並走しないようにする。タスクの失敗は握り潰す
        (失敗は各 store に既に記録されているか、誰も読まずに捨てられる)。
        """
        for task in self.background_tasks:
            if not task.done():
                try:
                    await task
                except Exception:
                    pass

    def dispose(self) -> None:
        """セッション終了時に呼ぶ。base video の tmp ファイルを削除する。

        `mask_store` / `full_foreground_store` / `background_tasks` は
        Session が GC されると同時に消えるので明示的な破棄は不要。
        ただし `background_tasks` は事前に `wait_for_tasks()` で
        完了させておくこと(install 側の責務)。
        """
        try:
            if self.base_video_path and os.path.exists(self.base_video_path):
                os.unlink(self.base_video_path)
        except OSError:
            logger.warning("failed to remove video file: %s", self.base_video_path)


class SessionSlot:
    """常に最大 1 件のセッションだけを保持するスロット。

    `install` で新しい session を登録すると、旧 session は `dispose` で破棄される。
    """

    def __init__(self) -> None:
        self._current: Session | None = None
        self._lock = threading.Lock()

    def install(self, session: Session) -> None:
        """新しい session を登録。旧 session は dispose される。

        旧 session の background_tasks の完了は呼び出し側で `wait_for_tasks()`
        を await してから install すること。
        """
        with self._lock:
            old = self._current
            self._current = session
        if old is not None and old.base_video_path != session.base_video_path:
            old.dispose()

    def current(self) -> Session | None:
        with self._lock:
            return self._current


session_slot = SessionSlot()
