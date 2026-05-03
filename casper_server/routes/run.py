import asyncio
import json
import logging
import os
import shutil
import tempfile

from fastapi import APIRouter, File, Form, HTTPException, Response, UploadFile

from casper_server.holder import casper_holder
from casper_server.pipeline import run_one_seq
from server.model import CASPER_DEFAULT_PROMPT, CASPER_STARTUP_TIMEOUT_SEC


router = APIRouter()
logger = logging.getLogger(__name__)


# 同時に 1 件しか処理させない（GPU メモリ・状態整合性）
_run_lock = asyncio.Lock()


def _round_to_multiple_of_16(value: int) -> int:
    """Wan2.1 の VAE (8x) + transformer patch (2x) 制約に合わせ、16 の倍数に丸める。

    最近傍に丸める。最小値は 16。
    """
    if value <= 0:
        raise ValueError(f"invalid dimension: {value}")
    snapped = int(round(value / 16.0)) * 16
    return max(16, snapped)


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

    # 既に処理中なら 409。待たせない（仕様 §3.10.5）
    if _run_lock.locked():
        raise HTTPException(status_code=409, detail="another run is in progress")

    # 元動画の解像度を 16 の倍数に丸めて推論サイズとする
    try:
        snap_h = _round_to_multiple_of_16(height)
        snap_w = _round_to_multiple_of_16(width)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    sample_size = (snap_h, snap_w)
    logger.info(
        "casper run: input=%dx%d -> sample_size=%dx%d",
        width, height, snap_w, snap_h,
    )

    async with _run_lock:
        work_root = tempfile.mkdtemp(prefix="casper_run_")
        seq_name = f"seq_{os.path.basename(work_root).split('_', 2)[-1]}"
        seq_dir = os.path.join(work_root, "data", seq_name)
        save_dir = os.path.join(work_root, "out")
        os.makedirs(seq_dir, exist_ok=True)
        os.makedirs(save_dir, exist_ok=True)

        try:
            # 1. 入力動画を input_video.mp4 として配置
            with open(os.path.join(seq_dir, "input_video.mp4"), "wb") as f:
                shutil.copyfileobj(input_video.file, f)

            # 2. トリマスクを trimask_00.mp4 として配置
            #    （mask_*.mp4 を置かないことで gen-omnimatte の trimask 経路に流れる）
            with open(os.path.join(seq_dir, "trimask_00.mp4"), "wb") as f:
                shutil.copyfileobj(trimask.file, f)

            # 3. prompt.json を生成
            with open(os.path.join(seq_dir, "prompt.json"), "w", encoding="utf-8") as f:
                json.dump({"bg": prompt or CASPER_DEFAULT_PROMPT}, f)

            # 4. Casper 推論（CPU/GPU の重い処理は worker thread に逃がす）
            try:
                out_path = await asyncio.to_thread(
                    run_one_seq,
                    casper_holder.cfg,
                    casper_holder.pipeline,
                    casper_holder.vae,
                    casper_holder.generator,
                    seq_dir,
                    save_dir,
                    sample_size,
                )
            except Exception as exc:
                logger.exception("casper run failed")
                raise HTTPException(status_code=500, detail=f"casper run failed: {exc}")

            with open(out_path, "rb") as f:
                mp4_bytes = f.read()

            return Response(content=mp4_bytes, media_type="video/mp4")
        finally:
            shutil.rmtree(work_root, ignore_errors=True)
