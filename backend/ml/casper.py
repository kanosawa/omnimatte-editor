"""Casper（前景削除）を本サーバプロセス内で実行するためのモジュール。

ロード状態管理（`Casper` クラス）、同時実行制御（`run_lock`）、直近の
preload 結果（`_preload`）、推論本体（`_do_pipeline_run`）、および
ルート（`/segment`、`/remove`）から呼ばれる高レベル API
（`run_casper` / `preload_casper`）を集約する。

`/segment` 完了直後に `preload_casper` が走り、結果を `_preload` に
登録（または進行中なら Future を登録）する。続く `/remove` の
`run_casper` は `_preload` を覗いて trimask が一致していれば preload 結果を
そのまま返す。一致は ndarray の identity 比較（`is`）で行うので、
mask_store は trimask を copy せず参照のまま保持する前提に依存している。

以前は別プロセスの sidecar (`casper_server/`) に分離していたが、
absl.flags の汚染が fork 版 `vendor/gen-omnimatte-public` で解消されたため、
本サーバプロセスに同居させて HTTP 経由のオーバヘッドと sidecar 管理を
排除している。
"""
import asyncio
import json
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass
from typing import Any

import numpy as np

from backend.config import MODEL_STARTUP_TIMEOUT_SEC
from backend.media.video_io import write_trimask_mp4
from backend.ml.casper_pipeline import (
    CASPER_TRANSFORMER_PATH,
    build_default_config,
    load_pipeline,
    run_one_seq,
)


logger = logging.getLogger(__name__)


# Casper への prompt 既定値。リクエスト側で省略された場合に使う。
CASPER_DEFAULT_PROMPT = "a clean background video."


# ============================================================================
# ロード状態管理
# ============================================================================

class Casper:
    """Casper パイプラインのロード状態管理。"""

    def __init__(self) -> None:
        self.pipeline: Any = None
        self.vae: Any = None
        self.generator: Any = None
        self.cfg: Any = None
        self._error: str | None = None
        self._ready_event = asyncio.Event()

    def is_ready(self) -> bool:
        """ロードが完了し、かつ失敗していないか。`preload_casper` の早期 return 用。"""
        return self._ready_event.is_set() and self._error is None

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
            self.cfg = cfg
            self.pipeline = pipeline
            self.vae = vae
            self.generator = generator
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
    """`preload_casper` が登録する直近 1 件の結果（または進行中の Future）。

    - `trimask`: preload を起動した時点の trimask ndarray 参照。`run_casper`
      は同じ ndarray を持って来た場合に hit させる（identity 比較）。
    - `future`: Casper 出力 mp4 バイト列で resolve される。preload がまだ
      走っている間に `/remove` が来た場合は、ここを await して待ち合わせる。
    """
    trimask: np.ndarray
    future: asyncio.Future  # Future[bytes]


# 直近 1 件の preload 結果（または進行中の Future）。後発の preload で
# 自然に上書きされ、明示的な clear は不要（古い Future は誰にも await
# されなくなるだけで、resolve 自体は副作用なく終わる）。
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
    width: int,
    height: int,
    fps: float,
    prompt: str,
) -> bytes:
    """既に一時ファイルとして配置された動画・トリマスクで Casper を回し、mp4 バイトを返す。

    呼び出し側で `run_lock` を取得しておくこと。GPU を 1 件ずつ使う前提。

    `trimask_path` は 3 値 trimask（0=remove / 128=neutral / 255=keep）の mp4。
    seq_dir には `trimask_00.mp4` として配置し、`mask_*.mp4` は置かないことで
    gen-omnimatte の trimask 読み込み経路に流す。

    `fps` は出力 mp4 のフレームレート。入力動画の fps をそのまま渡す。
    """
    snap_h = _round_to_multiple_of_16(height)
    snap_w = _round_to_multiple_of_16(width)
    sample_size = (snap_h, snap_w)
    logger.info(
        "casper pipeline run: input=%dx%d -> sample_size=%dx%d",
        width, height, snap_w, snap_h,
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
            json.dump({"bg": prompt or CASPER_DEFAULT_PROMPT}, f)

        out_path = run_one_seq(
            casper.cfg,
            casper.pipeline,
            casper.vae,
            casper.generator,
            seq_dir,
            save_dir,
            sample_size,
            fps,
        )

        with open(out_path, "rb") as f:
            return f.read()
    finally:
        shutil.rmtree(work_root, ignore_errors=True)


async def _execute(
    base_video_path: str,
    trimask: np.ndarray,
    fps: float,
    width: int,
    height: int,
) -> bytes:
    """trimask を tempfile に書き出して `_do_pipeline_run` を回す共通ヘルパ。

    `run_lock` 取得と一時ファイル管理を担当する。例外はそのまま伝播するので、
    呼び出し側で受け止めて HTTP / Future にマップすること。
    """
    fd, trimask_path = tempfile.mkstemp(prefix="casper_trimask_", suffix=".mp4")
    os.close(fd)
    try:
        write_trimask_mp4(trimask=trimask, fps=fps, out_path=trimask_path)
        async with run_lock:
            return await asyncio.to_thread(
                _do_pipeline_run,
                base_video_path,
                trimask_path,
                width,
                height,
                fps,
                CASPER_DEFAULT_PROMPT,
            )
    finally:
        try:
            os.unlink(trimask_path)
        except OSError:
            pass


# ============================================================================
# 高レベル API（routes から呼ばれる）
# ============================================================================

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

    `preload_casper` が同じ trimask ndarray で先回り中・完了済みなら
    そちらの結果（Future）を再利用する。一致しない・preload が失敗していた
    場合はその場で pipeline を回す。
    """
    await casper.wait_ready(timeout=MODEL_STARTUP_TIMEOUT_SEC)

    pending = _preload
    if pending is not None and pending.trimask is trimask:
        try:
            mp4_bytes = await pending.future
            logger.info("[run] preload HIT: returning %d bytes", len(mp4_bytes))
            return mp4_bytes
        except Exception:
            logger.warning("[run] preload failed; falling back to fresh run")

    logger.info("[run] preload MISS: running pipeline")
    return await _execute(base_video_path, trimask, fps, width, height)


async def preload_casper(
    base_video_path: str,
    trimask: np.ndarray,
    fps: float,
    width: int,
    height: int,
) -> None:
    """先回りで Casper を回して `_preload` に登録する投げ捨て関数。

    `/segment` 完了直後に `asyncio.create_task` で発射する。`_preload` への
    登録は pipeline 開始前に行うので、preload と `/remove` が並走しても
    `/remove` 側は同じ Future を await して結果を待ち合わせられる。
    失敗時は Future に例外をセットし、`run_casper` 側でフォールバックさせる。
    """
    if not casper.is_ready():
        logger.info("[preload] skipped: casper not ready")
        return

    global _preload
    future: asyncio.Future = asyncio.get_running_loop().create_future()
    _preload = PendingPreload(trimask=trimask, future=future)

    logger.info("[preload] starting pipeline")
    try:
        mp4_bytes = await _execute(base_video_path, trimask, fps, width, height)
        future.set_result(mp4_bytes)
        logger.info("[preload] complete: %d bytes", len(mp4_bytes))
    except Exception as exc:
        logger.exception("[preload] failed")
        future.set_exception(exc)
