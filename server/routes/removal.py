import logging
import os

import torch
from fastapi import APIRouter, HTTPException, Response

from server.mask_store import mask_store
from server.model import SAM2_DEVICE, model_holder
from server.removal import (
    CasperRunError,
    CasperWeightMissingError,
    run_foreground_removal,
)
from server.session import session_slot
from server.video_io import probe_video


router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/remove")
async def remove_foreground() -> Response:
    try:
        await model_holder.wait_ready(timeout=5.0)
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

    predictor = model_holder.predictor
    if predictor is None:
        raise HTTPException(status_code=503, detail="predictor unavailable")

    base_video_path = session.base_video_path
    fps = session.fps

    try:
        new_video_path = await run_foreground_removal(
            base_video_path=base_video_path,
            masks=record.masks,
            fps=fps,
        )
    except CasperWeightMissingError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except CasperRunError as exc:
        logger.exception("casper run failed")
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception:
        logger.exception("foreground removal failed")
        raise HTTPException(status_code=500, detail="foreground removal failed")

    # 新 base video のメタを取得し、SAM2 inference_state を作り直す
    try:
        new_meta = probe_video(new_video_path)
    except ValueError as exc:
        if os.path.exists(new_video_path):
            os.unlink(new_video_path)
        raise HTTPException(status_code=500, detail=str(exc))

    try:
        with torch.inference_mode(), torch.autocast(SAM2_DEVICE, dtype=torch.bfloat16):
            new_inference_state = predictor.init_state(video_path=new_video_path)
    except Exception:
        logger.exception("init_state failed for new base video")
        if os.path.exists(new_video_path):
            os.unlink(new_video_path)
        raise HTTPException(
            status_code=500,
            detail="failed to reinitialize SAM2 state on new base video",
        )

    session_slot.swap_base_video(
        new_base_video_path=new_video_path,
        new_inference_state=new_inference_state,
        width=new_meta.width,
        height=new_meta.height,
        fps=new_meta.fps,
        num_frames=new_meta.num_frames,
    )
    mask_store.clear()

    with open(new_video_path, "rb") as f:
        mp4_bytes = f.read()

    return Response(content=mp4_bytes, media_type="video/mp4")
