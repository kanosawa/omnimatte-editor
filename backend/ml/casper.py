"""Casper（前景削除）を本サーバプロセス内で実行するためのモジュール。

ロード状態管理（`Casper` クラス）、同時実行制御（`run_lock`）、出力 mp4 の
キャッシュ（`OutputCache`）、推論本体（`do_pipeline_run`）、および
ルート（`/segment`、`/remove`）から呼ばれる高レベル API
（`run_casper` / `preload_casper` / `get_casper_state`）を集約する。

以前は別プロセスの sidecar (`casper_server/`) に分離していたが、
absl.flags の汚染が fork 版 `vendor/gen-omnimatte-public` で解消されたため、
本サーバプロセスに同居させて HTTP 経由のオーバヘッドと sidecar 管理を
排除している。
"""
import asyncio
import hashlib
import json
import logging
import os
import shutil
import tempfile
import threading
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from backend.config import (
    CASPER_DEFAULT_PROMPT,
    CASPER_STARTUP_TIMEOUT_SEC,
    CASPER_TRANSFORMER_PATH,
)
from backend.media.video_io import write_trimask_mp4
from backend.ml.casper_pipeline import build_default_config, load_pipeline, run_one_seq


logger = logging.getLogger(__name__)


# ============================================================================
# 例外
# ============================================================================

class CasperNotReadyError(Exception):
    """Casper モデルがロード中 / 失敗で呼べない状態。`/remove` で 503 にマップ。"""


class CasperBusyError(Exception):
    """別の Casper 推論が進行中で `run_lock` が取れない（409）。"""


class CasperRunError(Exception):
    """Casper 推論パイプライン内で例外が発生（500）。"""


# ============================================================================
# ロード状態管理
# ============================================================================

CasperState = Literal["loading", "ready", "failed"]


class Casper:
    """Casper パイプラインのロード状態管理。"""

    def __init__(self) -> None:
        self._state: CasperState = "loading"
        self._pipeline: Any = None
        self._vae: Any = None
        self._generator: Any = None
        self._cfg: Any = None
        self._error: str | None = None
        self._ready_event = asyncio.Event()

    @property
    def state(self) -> CasperState:
        return self._state

    @property
    def pipeline(self) -> Any:
        return self._pipeline

    @property
    def vae(self) -> Any:
        return self._vae

    @property
    def generator(self) -> Any:
        return self._generator

    @property
    def cfg(self) -> Any:
        return self._cfg

    @property
    def error(self) -> str | None:
        return self._error

    def _load_sync(self):
        if not os.path.exists(CASPER_TRANSFORMER_PATH):
            raise FileNotFoundError(
                f"casper model not found: {CASPER_TRANSFORMER_PATH}"
            )
        cfg = build_default_config()
        pipeline, vae, generator = load_pipeline(cfg)
        return cfg, pipeline, vae, generator

    async def load(self) -> None:
        try:
            cfg, pipeline, vae, generator = await asyncio.to_thread(self._load_sync)
            self._cfg = cfg
            self._pipeline = pipeline
            self._vae = vae
            self._generator = generator
            self._state = "ready"
            logger.info("casper pipeline loaded")
        except Exception as exc:
            logger.exception("casper pipeline load failed")
            self._error = str(exc)
            self._state = "failed"
        self._ready_event.set()

    async def wait_ready(self, timeout: float | None = None) -> None:
        if self._state == "ready":
            return
        if self._state == "failed":
            raise CasperNotReadyError(f"casper failed to load: {self._error}")
        try:
            await asyncio.wait_for(self._ready_event.wait(), timeout=timeout)
        except asyncio.TimeoutError as exc:
            raise CasperNotReadyError("casper not ready (timeout)") from exc
        if self._state == "failed":
            raise CasperNotReadyError(f"casper failed to load: {self._error}")


casper = Casper()


# ============================================================================
# 出力キャッシュ
# ============================================================================

def _hash_file(path: str) -> bytes:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.digest()


@dataclass
class _CacheEntry:
    video_hash: bytes
    mask_hash: bytes
    mp4_bytes: bytes


class OutputCache:
    """単一スロット。`preload_casper` と `run_casper` の双方からアクセスされる。"""

    def __init__(self) -> None:
        self._entry: _CacheEntry | None = None
        self._lock = threading.Lock()

    def set(self, video_hash: bytes, mask_hash: bytes, mp4_bytes: bytes) -> None:
        with self._lock:
            self._entry = _CacheEntry(video_hash, mask_hash, mp4_bytes)

    def get(self, video_hash: bytes, mask_hash: bytes) -> bytes | None:
        with self._lock:
            if self._entry is None:
                return None
            if self._entry.video_hash == video_hash and self._entry.mask_hash == mask_hash:
                return self._entry.mp4_bytes
            return None

    def clear(self) -> None:
        with self._lock:
            self._entry = None


output_cache = OutputCache()


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
    width: int,
    height: int,
    prompt: str,
) -> bytes:
    """既に一時ファイルとして配置された動画・トリマスクで Casper を回し、mp4 バイトを返す。

    呼び出し側で `run_lock` を取得しておくこと。GPU を 1 件ずつ使う前提。

    `trimask_path` は 3 値 trimask（0=remove / 128=neutral / 255=keep）の mp4。
    seq_dir には `trimask_00.mp4` として配置し、`mask_*.mp4` は置かないことで
    gen-omnimatte の trimask 読み込み経路に流す。
    """
    snap_h = _round_to_multiple_of_16(height)
    snap_w = _round_to_multiple_of_16(width)
    sample_size = (snap_h, snap_w)
    logger.info(
        "casper pipeline run: input=%dx%d -> sample_size=%dx%d",
        width, height, snap_w, snap_h,
    )

    work_root = tempfile.mkdtemp(prefix="casper_pipe_")
    seq_name = f"seq_{os.path.basename(work_root).split('_', 2)[-1]}"
    seq_dir = os.path.join(work_root, "data", seq_name)
    save_dir = os.path.join(work_root, "out")
    os.makedirs(seq_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    try:
        shutil.copyfile(input_video_path, os.path.join(seq_dir, "input_video.mp4"))
        shutil.copyfile(trimask_path, os.path.join(seq_dir, "trimask_00.mp4"))

        with open(os.path.join(seq_dir, "prompt.json"), "w", encoding="utf-8") as f:
            json.dump({"bg": prompt or CASPER_DEFAULT_PROMPT}, f)

        out_path = run_one_seq(
            casper.cfg,
            casper.pipeline,
            casper.vae,
            casper.generator,
            seq_dir,
            save_dir,
            sample_size,
        )

        with open(out_path, "rb") as f:
            return f.read()
    finally:
        shutil.rmtree(work_root, ignore_errors=True)


# ============================================================================
# 高レベル API（routes から呼ばれる）
# ============================================================================

def get_casper_state() -> str:
    """`/health` から参照される現在の Casper ロード状態。"""
    return casper.state


async def run_casper(
    base_video_path: str,
    trimask: np.ndarray,
    fps: float,
    width: int,
    height: int,
) -> bytes:
    """Casper を回して前景削除済み mp4 バイナリを返す。

    `trimask`: (T, H, W) uint8。値は {0, 128, 255} の 3 値で、それぞれ
      remove（対象前景） / neutral（背景） / keep（他の前景）を表す。

    呼び出し側はロード未完・busy・推論失敗を例外で受け取り、適切な HTTP ステータスに変換する。
    一時ファイルは関数内で作成・削除する。`preload_casper` で先回り計算済みなら
    `output_cache` から即返す。
    """
    await casper.wait_ready(timeout=CASPER_STARTUP_TIMEOUT_SEC)

    fd, trimask_path = tempfile.mkstemp(prefix="casper_trimask_", suffix=".mp4")
    os.close(fd)
    try:
        write_trimask_mp4(trimask=trimask, fps=fps, out_path=trimask_path)

        video_hash = _hash_file(base_video_path)
        trimask_hash = _hash_file(trimask_path)

        # キャッシュ check（lock 取得前のファストパス）
        cached = output_cache.get(video_hash, trimask_hash)
        if cached is not None:
            logger.info("output_cache HIT (fast path), returning %d bytes", len(cached))
            return cached

        # キャッシュミス。lock 取得して再 check（preload が並走していた可能性）
        if run_lock.locked():
            # 後続用：locked() は厳密ではないが、診断ログに使う
            logger.info("run_lock contended; will wait")
        try:
            # busy 判定は明示的に非ブロッキング acquire を使う必要があるが、
            # 仕様上は preload と /remove は逐次化されればよく、待ち合わせて構わない。
            # 過去 sidecar 実装でも preload は asyncio.Lock 内で待っていた。
            async with run_lock:
                cached = output_cache.get(video_hash, trimask_hash)
                if cached is not None:
                    logger.info("output_cache HIT (after lock), returning %d bytes", len(cached))
                    return cached

                logger.info("output_cache MISS, running pipeline")
                try:
                    mp4_bytes = await asyncio.to_thread(
                        _do_pipeline_run,
                        base_video_path,
                        trimask_path,
                        width,
                        height,
                        CASPER_DEFAULT_PROMPT,
                    )
                except ValueError as exc:
                    raise CasperRunError(str(exc)) from exc
                except Exception as exc:
                    logger.exception("casper run failed")
                    raise CasperRunError(f"casper run failed: {exc}") from exc

                output_cache.set(video_hash, trimask_hash, mp4_bytes)
                return mp4_bytes
        except CasperRunError:
            raise
    finally:
        try:
            os.unlink(trimask_path)
        except OSError:
            pass


async def preload_casper(
    base_video_path: str,
    trimask: np.ndarray,
    fps: float,
    width: int,
    height: int,
) -> None:
    """先回りで Casper を回して `output_cache` に保存する投げ捨て関数。

    `/segment` 完了直後に `asyncio.create_task` で発射する。ユーザーが
    「前景削除」ボタンを押すまでの裏で計算を済ませ、`run_casper` のキャッシュ
    ヒットを狙う。失敗しても例外を投げず、ログだけ残す。
    """
    # casper が ready でないなら何もしない（後で run_casper が来たときに動く）
    if casper.state != "ready":
        logger.info("preload skipped: casper_state=%s", casper.state)
        return

    fd, trimask_path = tempfile.mkstemp(prefix="casper_preload_trimask_", suffix=".mp4")
    os.close(fd)
    try:
        try:
            write_trimask_mp4(trimask=trimask, fps=fps, out_path=trimask_path)
        except Exception:
            logger.exception("preload: trimask mp4 write failed (silently swallowed)")
            return

        video_hash = _hash_file(base_video_path)
        trimask_hash = _hash_file(trimask_path)
        if output_cache.get(video_hash, trimask_hash) is not None:
            logger.info("preload skipped: already cached")
            return

        async with run_lock:
            if output_cache.get(video_hash, trimask_hash) is not None:
                logger.info("preload skipped (after lock): already cached")
                return

            logger.info("preload starting pipeline")
            try:
                mp4_bytes = await asyncio.to_thread(
                    _do_pipeline_run,
                    base_video_path,
                    trimask_path,
                    width,
                    height,
                    CASPER_DEFAULT_PROMPT,
                )
            except Exception:
                logger.exception("preload pipeline failed (silently swallowed)")
                return

            output_cache.set(video_hash, trimask_hash, mp4_bytes)
            logger.info("preload complete: cached %d bytes", len(mp4_bytes))
    finally:
        try:
            os.unlink(trimask_path)
        except OSError:
            pass
