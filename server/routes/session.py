import logging
import os
import shutil
import tempfile

import torch
from fastapi import APIRouter, File, HTTPException, UploadFile

from server.mask_store import mask_store
from server.model import SAM2_DEVICE, model_holder
from server.schemas import StartSessionResponse, VideoMeta
from server.session import session_slot
from server.video_io import probe_video


router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/session", response_model=StartSessionResponse)
async def start_session(video: UploadFile = File(...)) -> StartSessionResponse:
    try:
        await model_holder.wait_ready(timeout=5.0)
    except TimeoutError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    suffix = os.path.splitext(video.filename or "")[1] or ".mp4"
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)

    try:
        with open(tmp_path, "wb") as out:
            shutil.copyfileobj(video.file, out)

        try:
            meta = probe_video(tmp_path)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        predictor = model_holder.predictor
        if predictor is None:
            raise HTTPException(status_code=503, detail="predictor unavailable")

        with torch.inference_mode(), torch.autocast(SAM2_DEVICE, dtype=torch.bfloat16):
            inference_state = predictor.init_state(video_path=tmp_path)

        session = session_slot.replace(
            inference_state=inference_state,
            base_video_path=tmp_path,
            width=meta.width,
            height=meta.height,
            fps=meta.fps,
            num_frames=meta.num_frames,
        )
        # 新規セッション開始時に直前のマスクは無効
        mask_store.clear()
    except HTTPException:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        logger.exception("session creation failed")
        raise HTTPException(status_code=500, detail="failed to create session")

    return StartSessionResponse(
        video_meta=VideoMeta(
            width=session.width,
            height=session.height,
            fps=session.fps,
            num_frames=session.num_frames,
            duration_sec=session.num_frames / session.fps if session.fps > 0 else 0.0,
        ),
    )
