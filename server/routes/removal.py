import asyncio
import logging
import os
import tempfile

import cv2
import numpy as np
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
from server.video_io import probe_video


router = APIRouter()
logger = logging.getLogger(__name__)


def _debug_probe_casper_output(path: str, mp4_size: int) -> None:
    """Casper 出力動画の metadata と各サンプルフレームの平均輝度をログに出す.

    [DEBUG black-screen] 真っ黒問題の切り分け用。原因確定後に削除する。
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        logger.warning("casper-debug: cannot open %s (size=%d bytes)", path, mp4_size)
        return
    try:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        codec_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join(chr((codec_int >> (8 * i)) & 0xFF) for i in range(4)) if codec_int else "?"
        logger.info(
            "casper-debug: probe path=%s size=%d bytes  cv2=%dx%d fps=%.2f frames=%d codec=%s",
            path, mp4_size, w, h, fps, n, codec,
        )
        # サンプルフレーム: 先頭・中央・末尾の平均輝度を取得（真っ黒なら ~0）
        samples = sorted(set([0, max(0, n // 2), max(0, n - 1)])) if n > 0 else []
        for idx in samples:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                logger.warning("casper-debug: failed to read frame %d", idx)
                continue
            mean = float(np.mean(frame))
            std  = float(np.std(frame))
            logger.info(
                "casper-debug: frame %3d/%d  shape=%s  mean=%.2f  std=%.2f%s",
                idx, n, tuple(frame.shape), mean, std,
                "  (NEAR-BLACK!)" if mean < 5.0 else "",
            )
    finally:
        cap.release()


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

    # [DEBUG black-screen] Casper 出力を cv2 で probe して metadata + 各フレームの
    # 平均輝度をログに出す。クラウドで動画ファイルを直接確認できない環境でも、
    # 「Casper 出力が真っ黒か / 寸法・fps・frame 数が想定通りか」をログだけで判定できる。
    _debug_probe_casper_output(new_video_path, mp4_size=len(mp4_bytes))

    # 新 base video のメタ取得 + SAM inference_state 再構築
    try:
        new_meta = probe_video(new_video_path)
    except ValueError as exc:
        if os.path.exists(new_video_path):
            os.unlink(new_video_path)
        raise HTTPException(status_code=500, detail=str(exc))

    try:
        new_inference_state = sam_backend.init_state(video_path=new_video_path)
    except Exception:
        logger.exception("init_state failed for new base video")
        if os.path.exists(new_video_path):
            os.unlink(new_video_path)
        raise HTTPException(
            status_code=500,
            detail="failed to reinitialize SAM state on new base video",
        )

    new_session = session_slot.swap_base_video(
        new_base_video_path=new_video_path,
        new_inference_state=new_inference_state,
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
