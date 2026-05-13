import asyncio
import logging
import os
import shutil
import tempfile

import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.config import MODEL_STARTUP_TIMEOUT_SEC
from backend.media.video_io import probe_video, read_frame_at
from backend.ml.detector import detectron2
from backend.ml.sam import sam2
from backend.schemas import VideoMeta
from backend.stores.full_foreground_store import full_foreground_store
from backend.stores.mask_store import mask_store
from backend.stores.session import Session, session_slot


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
            meta = probe_video(tmp_path)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        sam2.open_session(video_path=tmp_path)

        session = session_slot.replace(
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
        # `/session` は即時 VideoMeta を返し、`/segment` 側で wait_ready する。
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

    return VideoMeta(
        width=session.width,
        height=session.height,
        fps=session.fps,
        num_frames=session.num_frames,
        duration_sec=session.num_frames / session.fps if session.fps > 0 else 0.0,
    )


async def _extract_full_foreground(session: Session) -> None:
    """中間フレームに COCO Mask R-CNN を実行 → 各検出を SAM で全フレームに propagate。

    結果を `full_foreground_store` に保存する。`/session` 完了後にバックグラウンドで実行される。
    SAM2 のセッション状態（`sam2._state`）を共有して使うが、最後に `sam2.reset()`
    してから返るので、後続の `/segment` は通常通り使える。
    """
    base_video_path = session.base_video_path
    try:
        await detectron2.wait_ready(timeout=MODEL_STARTUP_TIMEOUT_SEC)

        # 中間フレームを BGR で取得
        middle_frame_idx = session.num_frames // 2
        frame_bgr = await asyncio.to_thread(read_frame_at, base_video_path, middle_frame_idx)

        # Detectron2 で物体検出（class-agnostic）
        detected_masks = await asyncio.to_thread(detectron2.detect, frame_bgr)
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
    n_objects = len(detected_masks)
    n_frames = session.num_frames
    h, w = session.height, session.width

    per_object: list[np.ndarray] = [
        np.zeros((n_frames, h, w), dtype=bool) for _ in range(n_objects)
    ]

    sam2.reset()
    for obj_id, mask in enumerate(detected_masks):
        sam2.add_mask(frame_idx=keyframe_idx, obj_id=obj_id, mask=mask)

    for frame_idx, obj_ids, masks in sam2.propagate(start_frame_idx=keyframe_idx):
        for i, obj_id in enumerate(obj_ids):
            per_object[obj_id][frame_idx] = masks[i]
    for frame_idx, obj_ids, masks in sam2.propagate(
        start_frame_idx=keyframe_idx, reverse=True,
    ):
        for i, obj_id in enumerate(obj_ids):
            per_object[obj_id][frame_idx] = masks[i]

    # /segment が後でクリーンな state を使えるように reset
    sam2.reset()

    return per_object
