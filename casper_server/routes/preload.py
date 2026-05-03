"""`/preload` エンドポイント。先回りで Casper を回して結果を `output_cache` に
保存する。本サーバが `/segment` 完了時に投げ捨てで叩き、ユーザーが「前景削除」
ボタンを押す前に計算を済ませておく。

レスポンスは即時 `202 Accepted` を返し、実推論はバックグラウンドタスクで実行する。
"""
import asyncio
import logging
import os
import shutil
import tempfile

from fastapi import APIRouter, File, Form, UploadFile

from casper_server.holder import casper_holder
from casper_server.output_cache import hash_file, output_cache
from casper_server.runner import do_pipeline_run, run_lock
from server.model import CASPER_DEFAULT_PROMPT


router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/preload", status_code=202)
async def preload(
    input_video: UploadFile = File(...),
    mask: UploadFile = File(...),
    width: int = Form(...),
    height: int = Form(...),
    prompt: str = Form(CASPER_DEFAULT_PROMPT),
    fps: str = Form(""),
) -> dict:
    """multipart を受けたら一時ファイルに保存し、即 202 を返す。
    バックグラウンドで `_do_preload_background` が走る。
    """
    work_root = tempfile.mkdtemp(prefix="casper_preload_")
    input_video_path = os.path.join(work_root, "input_video.mp4")
    mask_path = os.path.join(work_root, "mask_00.mp4")

    try:
        with open(input_video_path, "wb") as f:
            shutil.copyfileobj(input_video.file, f)
        with open(mask_path, "wb") as f:
            shutil.copyfileobj(mask.file, f)
    except Exception:
        shutil.rmtree(work_root, ignore_errors=True)
        raise

    asyncio.create_task(
        _do_preload_background(work_root, input_video_path, mask_path, width, height, prompt)
    )
    return {"status": "accepted"}


async def _do_preload_background(
    work_root: str,
    input_video_path: str,
    mask_path: str,
    width: int,
    height: int,
    prompt: str,
) -> None:
    try:
        # casper_holder が ready でないなら何もせず終了（後で /run が来たときに動く）
        if casper_holder.state != "ready":
            logger.info("preload skipped: casper_state=%s", casper_holder.state)
            return

        video_hash = hash_file(input_video_path)
        mask_hash = hash_file(mask_path)
        if output_cache.get(video_hash, mask_hash) is not None:
            logger.info("preload skipped: already cached")
            return

        async with run_lock:
            # lock 取得後に再 check
            if output_cache.get(video_hash, mask_hash) is not None:
                logger.info("preload skipped (after lock): already cached")
                return

            logger.info("preload starting pipeline")
            try:
                mp4_bytes = await asyncio.to_thread(
                    do_pipeline_run,
                    input_video_path,
                    mask_path,
                    width,
                    height,
                    prompt,
                )
            except Exception:
                logger.exception("preload pipeline failed (silently swallowed)")
                return

            output_cache.set(video_hash, mask_hash, mp4_bytes)
            logger.info("preload complete: cached %d bytes", len(mp4_bytes))
    finally:
        shutil.rmtree(work_root, ignore_errors=True)
