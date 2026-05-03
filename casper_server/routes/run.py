import asyncio
import logging
import os
import shutil
import tempfile

from fastapi import APIRouter, File, Form, HTTPException, Response, UploadFile

from casper_server.holder import casper_holder
from casper_server.output_cache import hash_file, output_cache
from casper_server.runner import do_pipeline_run, run_lock
from server.model import CASPER_DEFAULT_PROMPT, CASPER_STARTUP_TIMEOUT_SEC


router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/run")
async def run(
    input_video: UploadFile = File(...),
    trimask: UploadFile = File(...),
    width: int = Form(...),
    height: int = Form(...),
    prompt: str = Form(CASPER_DEFAULT_PROMPT),
    fps: str = Form(""),  # 参考用（cfg.data.fps を使うため未使用）
) -> Response:
    try:
        await casper_holder.wait_ready(timeout=CASPER_STARTUP_TIMEOUT_SEC)
    except TimeoutError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    # 一時ファイルに保存（ハッシュ計算と pipeline 実行に使う）
    work_root = tempfile.mkdtemp(prefix="casper_run_")
    input_video_path = os.path.join(work_root, "input_video.mp4")
    trimask_path = os.path.join(work_root, "trimask_00.mp4")

    try:
        with open(input_video_path, "wb") as f:
            shutil.copyfileobj(input_video.file, f)
        with open(trimask_path, "wb") as f:
            shutil.copyfileobj(trimask.file, f)

        # キャッシュ check（lock 取得前のファストパス）
        video_hash = hash_file(input_video_path)
        trimask_hash = hash_file(trimask_path)
        cached = output_cache.get(video_hash, trimask_hash)
        if cached is not None:
            logger.info("output_cache HIT (fast path), returning %d bytes", len(cached))
            return Response(content=cached, media_type="video/mp4")

        # キャッシュミス。lock 取得して再 check（preload が走っていた可能性）
        async with run_lock:
            cached = output_cache.get(video_hash, trimask_hash)
            if cached is not None:
                logger.info("output_cache HIT (after lock), returning %d bytes", len(cached))
                return Response(content=cached, media_type="video/mp4")

            logger.info("output_cache MISS, running pipeline")
            try:
                mp4_bytes = await asyncio.to_thread(
                    do_pipeline_run,
                    input_video_path,
                    trimask_path,
                    width,
                    height,
                    prompt,
                )
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc))
            except Exception as exc:
                logger.exception("casper run failed")
                raise HTTPException(status_code=500, detail=f"casper run failed: {exc}")

            output_cache.set(video_hash, trimask_hash, mp4_bytes)
            return Response(content=mp4_bytes, media_type="video/mp4")
    finally:
        shutil.rmtree(work_root, ignore_errors=True)
