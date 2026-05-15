import logging
import os
import shutil
import tempfile

from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.config import MODEL_STARTUP_TIMEOUT_SEC
from backend.predictors.sam2 import sam2
from backend.schemas import VideoMeta
from backend.state.session import session_slot


router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/session", response_model=VideoMeta)
async def start_session(video: UploadFile = File(...)) -> VideoMeta:
    try:
        await sam2.wait_ready(timeout=MODEL_STARTUP_TIMEOUT_SEC)
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
            session = await session_slot.open(tmp_path)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
    except HTTPException:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        logger.exception("session creation failed")
        raise HTTPException(status_code=500, detail="failed to create session")

    return VideoMeta(
        width=session.meta.width,
        height=session.meta.height,
        fps=session.meta.fps,
        num_frames=session.meta.num_frames,
        duration_sec=session.meta.duration_sec,
    )
