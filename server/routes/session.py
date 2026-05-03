import asyncio
import logging
import os
import shutil
import tempfile

import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile

from server.detector import detector_holder
from server.full_foreground_store import full_foreground_store
from server.mask_store import mask_store
from server.sam_backend import sam_backend
from server.schemas import StartSessionResponse, VideoMeta
from server.session import Session, session_slot
from server.video_io import probe_video, read_frame_at


router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/session", response_model=StartSessionResponse)
async def start_session(video: UploadFile = File(...)) -> StartSessionResponse:
    try:
        await sam_backend.wait_ready(timeout=5.0)
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

        inference_state = sam_backend.init_state(video_path=tmp_path)

        session = session_slot.replace(
            inference_state=inference_state,
            base_video_path=tmp_path,
            width=meta.width,
            height=meta.height,
            fps=meta.fps,
            num_frames=meta.num_frames,
        )
        # 新規セッション開始時に直前のマスク・全前景データは無効
        mask_store.clear()
        full_foreground_store.start_loading()

        # 全前景抽出（R-CNN + SAM propagate）はバックグラウンドで実行。
        # `/session` は即時 videoMeta を返し、`/segment` 側で wait_ready する。
        asyncio.create_task(_extract_full_foreground(session))
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


async def _extract_full_foreground(session: Session) -> None:
    """中間フレームに COCO Mask R-CNN を実行 → 各検出を SAM で全フレームに propagate。

    結果を `full_foreground_store` に保存する。`/session` 完了後にバックグラウンドで実行される。
    SAM inference_state を共有して使うが、最後に reset_state してから返るので、
    後続の `/segment` は通常通り state を使える。
    """
    base_video_path = session.base_video_path
    try:
        await detector_holder.wait_ready(timeout=30.0)

        # 中間フレームを BGR で取得
        middle_frame_idx = session.num_frames // 2
        frame_bgr = await asyncio.to_thread(read_frame_at, base_video_path, middle_frame_idx)

        # Detectron2 で物体検出（class-agnostic）
        detected_masks = await asyncio.to_thread(detector_holder.detect, frame_bgr)
        logger.info("R-CNN detected %d objects on frame %d", len(detected_masks), middle_frame_idx)

        # 検出 0 のときも空リストを保持して ready 状態に遷移
        if not detected_masks:
            full_foreground_store.set_ready(per_object_masks=[], base_video_path=base_video_path)
            return

        # SAM video propagate を別 thread で実行
        per_object_masks = await asyncio.to_thread(
            _propagate_detected_masks,
            session,
            detected_masks,
            middle_frame_idx,
        )
        full_foreground_store.set_ready(
            per_object_masks=per_object_masks,
            base_video_path=base_video_path,
        )
        logger.info("full foreground extraction complete: %d objects", len(per_object_masks))
    except Exception as exc:
        logger.exception("full foreground extraction failed")
        full_foreground_store.set_failed(str(exc))


def _propagate_detected_masks(
    session: Session,
    detected_masks: list[np.ndarray],
    keyframe_idx: int,
) -> list[np.ndarray]:
    """各 detected mask を SAM video predictor に登録し、順方向＋逆方向に propagate。

    返り値: list of (T, H, W) bool。検出物体 1 つあたり 1 枚。
    """
    state = session.inference_state
    n_objects = len(detected_masks)
    n_frames = session.num_frames
    h, w = session.height, session.width

    per_object: list[np.ndarray] = [
        np.zeros((n_frames, h, w), dtype=bool) for _ in range(n_objects)
    ]

    sam_backend.reset_state(state)
    for obj_id, mask in enumerate(detected_masks):
        sam_backend.add_mask(
            state=state, frame_idx=keyframe_idx, obj_id=obj_id, mask=mask,
        )

    for frame_idx, obj_ids, masks in sam_backend.propagate(
        state, start_frame_idx=keyframe_idx, num_frames=n_frames,
    ):
        for i, obj_id in enumerate(obj_ids):
            per_object[obj_id][frame_idx] = masks[i]
    for frame_idx, obj_ids, masks in sam_backend.propagate(
        state, start_frame_idx=keyframe_idx, num_frames=n_frames, reverse=True,
    ):
        for i, obj_id in enumerate(obj_ids):
            per_object[obj_id][frame_idx] = masks[i]

    # /segment が後でクリーンな state を使えるように reset
    sam_backend.reset_state(state)

    return per_object
