"""Casper（前景削除）を本サーバプロセス内で実行するためのモジュール。

ロード状態管理（`Casper` クラス）、同時実行制御（`run_lock`）、直近の
preload 結果（`_preload`）、推論本体（`_do_pipeline_run`）、および
ルート（`/segment`、`/remove`）から呼ばれる高レベル API
（`run_casper` / `preload_casper`）を集約する。

`/segment` 完了直後に `preload_casper` が走り、推論 Task を `_preload` に
登録する。続く `/remove` の `run_casper` は `_preload` を覗いて trimask が
一致していれば Task を await して結果を受け取る（完了済みなら即返し、
進行中なら寝て待つ）。
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
from backend.ml.casper_adapter import CASPER_TRANSFORMER_PATH, CasperPipeline


logger = logging.getLogger(__name__)


# Casper への prompt 既定値。
CASPER_DEFAULT_PROMPT = "a clean background video."


class Casper:
    """Casper パイプラインのロード状態管理。"""

    def __init__(self) -> None:
        self.pipeline: CasperPipeline | None = None
        self._error: str | None = None
        self._ready_event = asyncio.Event()

    def is_ready(self) -> bool:
        """ロードが完了し、かつ失敗していないか。`preload_casper` の早期 return 用。"""
        return self._ready_event.is_set() and self._error is None

    def _load_sync(self) -> CasperPipeline:
        if not os.path.exists(CASPER_TRANSFORMER_PATH):
            raise FileNotFoundError(
                f"casper model not found: {CASPER_TRANSFORMER_PATH}"
            )
        return CasperPipeline()

    async def load(self) -> None:
        try:
            self.pipeline = await asyncio.to_thread(self._load_sync)
            logger.info("casper pipeline loaded")
        except Exception as exc:
            logger.exception("casper pipeline load failed")
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


casper = Casper()


# ============================================================================
# 直近の preload 結果スロット
# ============================================================================

@dataclass
class PendingPreload:
    """`preload_casper` が登録する直近 1 件の結果（または進行中の Task）。

    - `trimask`: preload を起動した時点の trimask ndarray 参照。`run_casper`
      は同じ ndarray を持って来た場合に hit させる（identity 比較）。
    - `task`: Casper 出力 mp4 バイト列で resolve される Task。preload がまだ
      走っている間に `/remove` が来た場合は、ここを await して待ち合わせる。
    """
    trimask: np.ndarray
    task: asyncio.Task  # Task[bytes]


# 直近 1 件の preload 結果（または進行中の Task）。後発の preload で
# 自然に上書きされ、明示的な clear は不要（古い Task は誰にも await
# されなくなるだけで、完了自体は副作用なく終わる）。
# 単一イベントループ上の async コードだけが触る前提なので、ロックは不要。
_preload: PendingPreload | None = None


# ============================================================================
# 推論実行
# ============================================================================

# 同時に 1 件しか pipeline を回さない（GPU メモリ・状態整合性）。
# run_casper と preload_casper の双方が共有する。
run_lock = asyncio.Lock()


def _round_to_multiple_of_16(value: int) -> int:
    if value <= 0:
        raise ValueError(f"invalid dimension: {value}")
    snapped = int(round(value / 16.0)) * 16
    return max(16, snapped)


def _do_pipeline_run(
    input_video_path: str,
    trimask_path: str,
    meta: VideoMetadata,
) -> bytes:
    """既に一時ファイルとして配置された動画・トリマスクで Casper を回し、mp4 バイトを返す。

    呼び出し側で `run_lock` を取得しておくこと。GPU を 1 件ずつ使う前提。

    `trimask_path` は 3 値 trimask（0=remove / 128=neutral / 255=keep）の mp4。
    seq_dir には `trimask_00.mp4` として配置し、`mask_*.mp4` は置かないことで
    gen-omnimatte の trimask 読み込み経路に流す。
    """
    snap_h = _round_to_multiple_of_16(meta.height)
    snap_w = _round_to_multiple_of_16(meta.width)
    sample_size = (snap_h, snap_w)
    logger.info(
        "casper pipeline run: input=%dx%d -> sample_size=%dx%d",
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

        out_path = casper.pipeline.run(
            seq_dir,
            save_dir,
            sample_size,
            meta,
        )

        with open(out_path, "rb") as f:
            return f.read()
    finally:
        shutil.rmtree(work_root, ignore_errors=True)


async def _execute(
    base_video_path: str,
    trimask: np.ndarray,
    meta: VideoMetadata,
) -> bytes:
    """trimask を tempfile に書き出して `_do_pipeline_run` を回す共通ヘルパ。

    `run_lock` 取得と一時ファイル管理を担当する。例外はそのまま伝播するので、
    呼び出し側で受け止めて HTTP / Task 経由でハンドリングすること。
    """
    fd, trimask_path = tempfile.mkstemp(prefix="casper_trimask_", suffix=".mp4")
    os.close(fd)
    try:
        write_trimask_mp4(trimask=trimask, fps=meta.fps, out_path=trimask_path)
        async with run_lock:
            return await asyncio.to_thread(
                _do_pipeline_run,
                base_video_path,
                trimask_path,
                meta,
            )
    finally:
        try:
            os.unlink(trimask_path)
        except OSError:
            pass


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


# ============================================================================
# 高レベル API（routes から呼ばれる）
# ============================================================================

async def run_casper(
    base_video_path: str,
    trimask: np.ndarray,
    meta: VideoMetadata,
) -> bytes:
    """Casper を回して前景削除済み mp4 バイナリを返す。

    `trimask`: (T, H, W) uint8。値は {0, 128, 255} の 3 値で、それぞれ
      remove（対象前景） / neutral（背景） / keep（他の前景）を表す。

    `preload_casper` が同じ trimask ndarray で先回り中・完了済みなら
    そちらの結果（Task）を再利用する。一致しない・preload が失敗していた
    場合はその場で pipeline を回す。
    """
    await casper.wait_ready(timeout=MODEL_STARTUP_TIMEOUT_SEC)

    pending = _preload
    if pending is not None and pending.trimask is trimask:
        try:
            mp4_bytes = await pending.task
            logger.info("[run] preload HIT: returning %d bytes", len(mp4_bytes))
            return mp4_bytes
        except Exception:
            logger.warning("[run] preload failed; falling back to fresh run")

    logger.info("[run] preload MISS: running pipeline")
    return await _execute(base_video_path, trimask, meta)


def preload_casper(
    base_video_path: str,
    trimask: np.ndarray,
    meta: VideoMetadata,
) -> None:
    """先回りで Casper を回して `_preload` に登録する fire-and-forget。

    `/segment` 完了直後に呼ぶ。内部で `asyncio.create_task` を発火し、
    Task を `_preload` に登録した上で即座に return する。`/remove` 側は
    同じ Task を await して結果を待ち合わせる。失敗時の例外は 2 経路で
    扱われる: (1) `_log_preload_result` がログ出力＋引き取りで "Task
    exception was never retrieved" 警告を抑制、(2) `run_casper` 側は
    `await pending.task` で同じ例外を捕捉してフレッシュ実行に
    フォールバックする。
    """
    if not casper.is_ready():
        logger.info("[preload] skipped: casper not ready")
        return

    global _preload
    logger.info("[preload] starting pipeline")
    task = asyncio.create_task(
        _execute(base_video_path, trimask, meta)
    )
    _preload = PendingPreload(trimask=trimask, task=task)
    task.add_done_callback(_log_preload_result)
