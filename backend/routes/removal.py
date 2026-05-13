import logging
import os
import tempfile
import time

from fastapi import APIRouter, HTTPException, Response

from backend.config import MODEL_STARTUP_TIMEOUT_SEC
from backend.media.video_io import probe_video
from backend.ml.casper import (
    CasperNotReadyError,
    CasperRunError,
    run_casper,
)
from backend.ml.sam import sam2
from backend.state.session import Session, session_slot


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
    fps = session.fps

    try:
        mp4_bytes = await run_casper(
            base_video_path=base_video_path,
            trimask=record.trimask,
            fps=fps,
            width=session.width,
            height=session.height,
        )
    except CasperNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
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

    try:
        sam2.open_session(video_path=new_video_path)
    except Exception:
        logger.exception("open_session failed for new base video")
        if os.path.exists(new_video_path):
            os.unlink(new_video_path)
        raise HTTPException(
            status_code=500,
            detail="failed to reinitialize SAM state on new base video",
        )

    # 旧 session のバックグラウンド抽出が走っていれば完了を待つ
    # (実際には /segment が wait_ready で待っているので no-op の場合が多い)
    await session.wait_for_tasks()

    new_session = Session(
        base_video_path=new_video_path,
        width=new_meta.width,
        height=new_meta.height,
        fps=new_meta.fps,
        num_frames=new_meta.num_frames,
        created_at=time.time(),
    )
    # base video が差し替わったので、全前景データも再生成する。
    # ここで再 propagate を待たない: lazy に /segment が wait_ready する。
    new_session.full_foreground_store.start_loading()
    session_slot.install(new_session)

    from backend.routes.session import schedule_extraction  # 循環 import 回避のため遅延 import
    schedule_extraction(new_session)

    return Response(content=mp4_bytes, media_type="video/mp4")
