import logging

import numpy as np
from fastapi import APIRouter, HTTPException, Response

from backend.config import (
    DETECTRON2_IOU_WITH_TARGET,
    FULL_FOREGROUND_WAIT_TIMEOUT_SEC,
    MODEL_STARTUP_TIMEOUT_SEC,
)
from backend.media.video_io import composite_overlay_to_mp4
from backend.predictors.casper import casper
from backend.predictors.sam2 import sam2
from backend.schemas import SegmentRequest
from backend.state.session import session_slot


router = APIRouter()
logger = logging.getLogger(__name__)


TRIMASK_REMOVE = 0      # 対象前景（消す）
TRIMASK_NEUTRAL = 128   # 背景（neutral）
TRIMASK_KEEP = 255      # 他の前景（残す）


@router.post("/segment")
async def segment(req: SegmentRequest) -> Response:
    try:
        await sam2.wait_ready(timeout=MODEL_STARTUP_TIMEOUT_SEC)
    except TimeoutError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    session = session_slot.current()
    if session is None:
        raise HTTPException(status_code=409, detail="no active session")

    if req.frame_idx >= session.meta.num_frames:
        raise HTTPException(
            status_code=422,
            detail=f"frame_idx out of range: {req.frame_idx} >= {session.meta.num_frames}",
        )

    logger.info(
        "segment request: frame_idx=%d bbox=%s session=(%dx%d) num_frames=%d base_video=%s",
        req.frame_idx, req.bbox, session.meta.width, session.meta.height, session.meta.num_frames,
        session.base_video_path,
    )

    try:
        await session.full_foreground_store.wait_ready(timeout=FULL_FOREGROUND_WAIT_TIMEOUT_SEC)
    except TimeoutError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    full_fg = session.full_foreground_store.current()
    if full_fg is None:
        raise HTTPException(status_code=503, detail="full foreground data unavailable")
    if full_fg.base_video_path != session.base_video_path:
        raise HTTPException(status_code=409, detail="full foreground data is stale")

    try:
        masks_target = sam2.segment_from_bboxes(
            [req.bbox], keyframe_idx=req.frame_idx,
        )[0]  # (T, H, W) bool
    except Exception:
        logger.exception("segmentation failed")
        raise HTTPException(status_code=500, detail="segmentation failed")

    masks_in_order = list(masks_target)  # composite 用に (H, W) のリストへ展開

    # R-CNN 検出物体から、対象前景と高 IoU のものを除外（領域単位）。
    # IoU は req.frame_idx 時点で計算（BBox を打ったフレーム）
    target_at_frame = masks_target[req.frame_idx]
    om = full_fg.object_masks  # (N, T, H, W)
    if om.shape[0] == 0:
        other_fg_combined = np.zeros_like(masks_target, dtype=bool)
        kept = 0
        excluded = 0
    else:
        om_at_frame = om[:, req.frame_idx]  # (N, H, W)
        inter = np.logical_and(om_at_frame, target_at_frame).reshape(om.shape[0], -1).sum(axis=1)
        union = np.logical_or(om_at_frame, target_at_frame).reshape(om.shape[0], -1).sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            iou = np.where(union > 0, inter / union, 0.0)
        keep_flags = iou <= DETECTRON2_IOU_WITH_TARGET  # (N,)
        kept = int(keep_flags.sum())
        excluded = int(om.shape[0] - kept)
        if keep_flags.any():
            other_fg_combined = om[keep_flags].any(axis=0)  # (T, H, W)
        else:
            other_fg_combined = np.zeros_like(masks_target, dtype=bool)
    logger.info(
        "trimask: target_at_frame_pixels=%d, other_fg_objects kept=%d excluded=%d",
        int(target_at_frame.sum()), kept, excluded,
    )

    trimask = np.full(masks_target.shape, TRIMASK_NEUTRAL, dtype=np.uint8)
    trimask[other_fg_combined & ~masks_target] = TRIMASK_KEEP
    trimask[masks_target] = TRIMASK_REMOVE

    try:
        mp4_bytes = composite_overlay_to_mp4(
            original_video_path=session.base_video_path,
            masks_in_order=masks_in_order,
            fps=session.meta.fps,
        )
    except Exception:
        logger.exception("composite encoding failed")
        raise HTTPException(status_code=500, detail="composite encoding failed")

    # /remove で再利用するため、生成した trimask をサーバ側に保持する
    session.mask_store.set(
        trimask=trimask,
        base_video_path=session.base_video_path,
        fps=session.meta.fps,
    )

    # 先回り Casper 推論をバックグラウンドで起動（fire-and-forget）。
    casper.preload(
        base_video_path=session.base_video_path,
        trimask=trimask,
        meta=session.meta,
    )

    return Response(content=mp4_bytes, media_type="video/mp4")
