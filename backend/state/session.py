import asyncio
import logging
import os
from dataclasses import dataclass, field

import numpy as np

from backend.config import MODEL_STARTUP_TIMEOUT_SEC
from backend.media.video_io import VideoMetadata, probe_video, read_frame_at
from backend.predictors.detectron2 import detectron2
from backend.predictors.sam2 import sam2
from backend.state.stores.full_foreground import FullForegroundRecord, FullForegroundStore
from backend.state.stores.mask import MaskStore


logger = logging.getLogger(__name__)


@dataclass
class Session:
    """1 本の base video に紐づく編集セッション。

    Session は自分の派生状態 (`mask_store`, `full_foreground_store`) と
    全前景抽出タスク (`extraction_task`) を所有する。`session_slot.open` で
    新しい Session に差し替えられると、旧 Session は dispose されて GC 対象になる。
    """
    base_video_path: str
    meta: VideoMetadata
    mask_store: MaskStore = field(default_factory=MaskStore)
    full_foreground_store: FullForegroundStore = field(default_factory=FullForegroundStore)
    extraction_task: asyncio.Task | None = field(default=None, init=False)

    def start_extraction(self) -> None:
        """全前景抽出をバックグラウンドで起動する。
        """
        self.extraction_task = self.full_foreground_store.submit(self._produce_full_foreground())

    async def wait_for_extraction(self) -> None:
        """抽出タスクの完了を待つ。

        新セッションへの差し替え前に呼んで、旧 Session の GPU 処理が
        新 Session と並走しないようにする。タスクの失敗は握り潰す。
        """
        task = self.extraction_task
        if task is not None and not task.done():
            try:
                await task
            except Exception:
                pass

    def dispose(self) -> None:
        try:
            if self.base_video_path and os.path.exists(self.base_video_path):
                os.unlink(self.base_video_path)
        except OSError:
            logger.warning("failed to remove video file: %s", self.base_video_path)

    async def _produce_full_foreground(self) -> FullForegroundRecord:
        """中間フレームに COCO Mask R-CNN で物体検出 → 各 bbox を SAM で全フレームに propagate。
        """
        await detectron2.wait_ready(timeout=MODEL_STARTUP_TIMEOUT_SEC)

        # 中間フレームを BGR で取得
        middle_frame_idx = self.meta.num_frames // 2
        frame_bgr = await asyncio.to_thread(read_frame_at, self.base_video_path, middle_frame_idx)

        # Detectron2 で物体検出（class-agnostic、bbox のみ）
        detected_bboxes = await asyncio.to_thread(detectron2.detect, frame_bgr)
        logger.info("R-CNN detected %d objects on frame %d", len(detected_bboxes), middle_frame_idx)

        # 検出が 0 のときは SAM propagate を skip
        if detected_bboxes:
            object_masks = await asyncio.to_thread(
                sam2.segment_from_bboxes,
                detected_bboxes,
                keyframe_idx=middle_frame_idx,
            )
        else:
            object_masks = np.zeros(
                (0, self.meta.num_frames, self.meta.height, self.meta.width),
                dtype=bool,
            )
        logger.info("full foreground extraction complete: %d objects", object_masks.shape[0])
        return FullForegroundRecord(
            object_masks=object_masks,
            base_video_path=self.base_video_path,
        )


class SessionSlot:
    """常に最大 1 件のセッションだけを保持するスロット。

    `open` で新しい video のセッションを active にすると、旧セッションは破棄される。
    """

    def __init__(self) -> None:
        self._current: Session | None = None

    def current(self) -> Session | None:
        return self._current

    async def open(self, video_path: str) -> Session:
        """新しい video のセッションを作って active にする。旧セッションは破棄。
        """
        meta = probe_video(video_path)
        old = self.current()
        if old is not None:
            await old.wait_for_extraction()
        sam2.open_session(video_path=video_path)
        session = Session(
            base_video_path=video_path,
            meta=meta,
        )
        self._install(session)
        session.start_extraction()
        return session

    def _install(self, session: Session) -> None:
        """slot に session を登録し、旧 session を dispose する。`open` の内部実装。"""
        old = self._current
        if old is not None:
            task = old.extraction_task
            if task is not None and not task.done():
                raise RuntimeError(
                    "_install called before old session's extraction completed; "
                    "await old.wait_for_extraction() first"
                )
        self._current = session
        if old is not None and old.base_video_path != session.base_video_path:
            old.dispose()


session_slot = SessionSlot()
