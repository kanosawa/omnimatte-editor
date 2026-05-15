"""Casperモジュール(Preload対応)
"""
import asyncio
import json
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass

import numpy as np

from backend.config import MODEL_STARTUP_TIMEOUT_SEC
from backend.media.video_io import VideoMetadata, write_trimask_mp4
from backend.predictors.casper_adapter import CASPER_TRANSFORMER_PATH, CasperAdapter


logger = logging.getLogger(__name__)


# Casper への prompt 既定値。
CASPER_DEFAULT_PROMPT = "a clean background video."


@dataclass
class PendingPreload:
    """`Casper.preload` が登録する直近 1 件の結果（または進行中の Task）。
    """
    trimask: np.ndarray
    task: asyncio.Task  # Task[bytes]


def _round_to_multiple_of_16(value: int) -> int:
    if value <= 0:
        raise ValueError(f"invalid dimension: {value}")
    snapped = int(round(value / 16.0)) * 16
    return max(16, snapped)


def _log_preload_result(task: asyncio.Task) -> None:
    """preload Task の完了時ログ。await されない例外をここで引き取り、
    "Task exception was never retrieved" の警告を抑える。"""
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.exception("[preload] failed", exc_info=exc)
    else:
        logger.info("[preload] complete: %d bytes", len(task.result()))


class Casper:
    def __init__(self) -> None:
        self.adapter: CasperAdapter | None = None
        self._error: str | None = None
        self._ready_event = asyncio.Event()
        self._preload: PendingPreload | None = None # preload結果、または進行中のTask。
        self._run_lock = asyncio.Lock() # 複数同時処理の防止。runとpreloadで共有。

    def is_ready(self) -> bool:
        """ロードが完了し、かつ失敗していないか。`preload` の早期 return 用。"""
        return self._ready_event.is_set() and self._error is None

    async def load(self) -> None:
        try:
            self.adapter = await asyncio.to_thread(self._load_sync)
            logger.info("casper adapter loaded")
        except Exception as exc:
            logger.exception("casper adapter load failed")
            self._error = str(exc)
        self._ready_event.set()

    async def wait_ready(self, timeout: float | None = None) -> None:
        if self._ready_event.is_set():
            if self._error is not None:
                raise RuntimeError(f"casper failed to load: {self._error}")
            return
        try:
            await asyncio.wait_for(self._ready_event.wait(), timeout=timeout)
        except asyncio.TimeoutError as exc:
            raise TimeoutError("casper not ready (timeout)") from exc
        if self._error is not None:
            raise RuntimeError(f"casper failed to load: {self._error}")

    def _load_sync(self) -> CasperAdapter:
        if not os.path.exists(CASPER_TRANSFORMER_PATH):
            raise FileNotFoundError(
                f"casper model not found: {CASPER_TRANSFORMER_PATH}"
            )
        return CasperAdapter()

    def preload(
        self,
        base_video_path: str,
        trimask: np.ndarray,
        meta: VideoMetadata,
    ) -> None:
        """先回りで Casper を回して `_preload` に登録する fire-and-forget。
        """
        if not self.is_ready():
            logger.info("[preload] skipped: casper not ready")
            return

        logger.info("[preload] starting adapter")
        task = asyncio.create_task(
            self._execute(base_video_path, trimask, meta)
        )
        self._preload = PendingPreload(trimask=trimask, task=task)
        task.add_done_callback(_log_preload_result)

    async def run(
        self,
        base_video_path: str,
        trimask: np.ndarray,
        meta: VideoMetadata,
    ) -> bytes:
        await self.wait_ready(timeout=MODEL_STARTUP_TIMEOUT_SEC)

        pending = self._preload
        if pending is not None and pending.trimask is trimask:
            try:
                mp4_bytes = await pending.task
                logger.info("[run] preload HIT: returning %d bytes", len(mp4_bytes))
                return mp4_bytes
            except Exception:
                logger.warning("[run] preload failed; falling back to fresh run")

        logger.info("[run] preload MISS: running adapter")
        return await self._execute(base_video_path, trimask, meta)

    async def _execute(
        self,
        base_video_path: str,
        trimask: np.ndarray,
        meta: VideoMetadata,
    ) -> bytes:
        """`_do_adapter_run` を回す `preload` と `run` の共通ヘルパ。
        """
        fd, trimask_path = tempfile.mkstemp(prefix="casper_trimask_", suffix=".mp4")
        os.close(fd)
        try:
            write_trimask_mp4(trimask=trimask, fps=meta.fps, out_path=trimask_path)
            async with self._run_lock:
                return await asyncio.to_thread(
                    self._do_adapter_run,
                    base_video_path,
                    trimask_path,
                    meta,
                )
        finally:
            try:
                os.unlink(trimask_path)
            except OSError:
                pass

    def _do_adapter_run(
        self,
        input_video_path: str,
        trimask_path: str,
        meta: VideoMetadata,
    ) -> bytes:
        """CasperAdapterのrun関数呼び出し
        """
        snap_h = _round_to_multiple_of_16(meta.height)
        snap_w = _round_to_multiple_of_16(meta.width)
        sample_size = (snap_h, snap_w)
        logger.info(
            "casper adapter run: input=%dx%d -> sample_size=%dx%d",
            meta.width, meta.height, snap_w, snap_h,
        )

        work_root = tempfile.mkdtemp(prefix="casper_pipe_")
        seq_name = "seq"
        seq_dir = os.path.join(work_root, "data", seq_name)
        save_dir = os.path.join(work_root, "out")
        os.makedirs(seq_dir, exist_ok=True)
        os.makedirs(save_dir, exist_ok=True)

        try:
            shutil.copyfile(input_video_path, os.path.join(seq_dir, "input_video.mp4"))
            shutil.copyfile(trimask_path, os.path.join(seq_dir, "trimask_00.mp4"))

            with open(os.path.join(seq_dir, "prompt.json"), "w", encoding="utf-8") as f:
                json.dump({"bg": CASPER_DEFAULT_PROMPT}, f)

            out_path = self.adapter.run(
                seq_dir,
                save_dir,
                sample_size,
                meta,
            )

            with open(out_path, "rb") as f:
                return f.read()
        finally:
            shutil.rmtree(work_root, ignore_errors=True)


casper = Casper()
