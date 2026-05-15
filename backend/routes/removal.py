import logging
import os
import tempfile

from fastapi import APIRouter, HTTPException, Response

from backend.config import MODEL_STARTUP_TIMEOUT_SEC
from backend.ml.casper import casper
from backend.ml.sam import sam2
from backend.state.session import session_slot


router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/remove")
async def remove_foreground() -> Response:
    try:
        await sam2.wait_ready(timeout=MODEL_STARTUP_TIMEOUT_SEC)
    except TimeoutError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    session = session_slot.current()
    if session is None:
        raise HTTPException(status_code=409, detail="no active session")

    record = session.mask_store.current()
    if record is None:
        raise HTTPException(status_code=409, detail="no segmentation result available")

    if record.base_video_path != session.base_video_path:
        # base video が前景削除や新規セッションで差し替わったあとに残っていたマスク
        raise HTTPException(status_code=409, detail="mask is stale")

    base_video_path = session.base_video_path

    try:
        mp4_bytes = await casper.run(
            base_video_path=base_video_path,
            trimask=record.trimask,
            meta=session.meta,
        )
    except TimeoutError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception:
        logger.exception("foreground removal failed")
        raise HTTPException(status_code=500, detail="foreground removal failed")

    # 受け取った mp4 を一時ファイルに書き出し（init_state がパスを要求するため）
    fd, new_video_path = tempfile.mkstemp(prefix="casper_out_", suffix=".mp4")
    os.close(fd)
    with open(new_video_path, "wb") as f:
        f.write(mp4_bytes)

    # base video 差し替え: probe + 旧 session の GPU 待ち + SAM2 再 init + 新 Session 構築 + 抽出再開
    # (再 propagate を待たない: lazy に /segment が wait_ready する)
    try:
        await session_slot.open(new_video_path)
    except Exception:
        logger.exception("failed to swap to new base video")
        if os.path.exists(new_video_path):
            os.unlink(new_video_path)
        raise HTTPException(
            status_code=500,
            detail="failed to reinitialize SAM state on new base video",
        )

    return Response(content=mp4_bytes, media_type="video/mp4")
