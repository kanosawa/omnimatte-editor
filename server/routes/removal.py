import asyncio
import logging
import os
import shutil
import tempfile

from fastapi import APIRouter, HTTPException, Response

from server.casper_client import (
    CasperBusyError,
    CasperRunError,
    CasperUnreachableError,
    run_casper,
)
from server.full_foreground_store import full_foreground_store
from server.mask_store import mask_store
from server.sam_backend import sam_backend
from server.session import session_slot
from server.video_io import extract_frames_to_jpeg_dir, probe_video


router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/remove")
async def remove_foreground() -> Response:
    try:
        await sam_backend.wait_ready(timeout=5.0)
    except TimeoutError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    session = session_slot.current()
    if session is None:
        raise HTTPException(status_code=409, detail="no active session")

    record = mask_store.current()
    if record is None:
        raise HTTPException(status_code=409, detail="no segmentation result available")

    if record.base_video_path != session.base_video_path:
        # base video が前景削除や新規セッションで差し替わったあとに残っていたマスク
        raise HTTPException(status_code=409, detail="mask is stale")

    base_video_path = session.base_video_path
    fps = session.fps

    try:
        mp4_bytes = await run_casper(
            base_video_path=base_video_path,
            trimask=record.trimask,
            fps=fps,
            width=session.width,
            height=session.height,
        )
    except CasperUnreachableError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except CasperBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except CasperRunError as exc:
        logger.exception("casper run failed")
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception:
        logger.exception("foreground removal failed")
        raise HTTPException(status_code=500, detail="foreground removal failed")

    # 受け取った mp4 を一時ファイルに書き出し（init_state がパスを要求するため）
    fd, new_video_path = tempfile.mkstemp(prefix="casper_out_", suffix=".mp4")
    os.close(fd)
    with open(new_video_path, "wb") as f:
        f.write(mp4_bytes)

    # 新 base video のメタ取得 + SAM inference_state 再構築
    try:
        new_meta = probe_video(new_video_path)
    except ValueError as exc:
        if os.path.exists(new_video_path):
            os.unlink(new_video_path)
        raise HTTPException(status_code=500, detail=str(exc))

    # SAM2 init_state には JPEG ディレクトリを渡す（routes/session.py と同じ理由）
    try:
        new_sam_frames_dir = await asyncio.to_thread(
            extract_frames_to_jpeg_dir, new_video_path,
        )
    except ValueError as exc:
        if os.path.exists(new_video_path):
            os.unlink(new_video_path)
        raise HTTPException(status_code=500, detail=str(exc))

    try:
        new_inference_state = sam_backend.init_state(video_path=new_sam_frames_dir)
    except Exception:
        logger.exception("init_state failed for new base video")
        if os.path.exists(new_video_path):
            os.unlink(new_video_path)
        if os.path.isdir(new_sam_frames_dir):
            shutil.rmtree(new_sam_frames_dir, ignore_errors=True)
        raise HTTPException(
            status_code=500,
            detail="failed to reinitialize SAM state on new base video",
        )

    new_session = session_slot.swap_base_video(
        new_base_video_path=new_video_path,
        new_inference_state=new_inference_state,
        new_sam_frames_dir=new_sam_frames_dir,
        width=new_meta.width,
        height=new_meta.height,
        fps=new_meta.fps,
        num_frames=new_meta.num_frames,
    )
    mask_store.clear()

    # base video が差し替わったので、全前景データも再生成する。
    # ここで再 propagate を待たない: lazy に /segment が wait_ready する。
    full_foreground_store.start_loading()
    from server.routes.session import _extract_full_foreground  # 循環 import 回避のため遅延 import
    asyncio.create_task(_extract_full_foreground(new_session))

    return Response(content=mp4_bytes, media_type="video/mp4")
