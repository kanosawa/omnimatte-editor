import logging

import numpy as np
import torch
from fastapi import APIRouter, HTTPException, Response

from server.model import SAM2_DEVICE, model_holder
from server.schemas import SegmentRequest
from server.session import session_slot
from server.video_io import composite_overlay_to_mp4


router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/segment")
async def segment(req: SegmentRequest) -> Response:
    try:
        await model_holder.wait_ready(timeout=5.0)
    except TimeoutError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    session = session_slot.current()
    if session is None:
        raise HTTPException(status_code=409, detail="no active session")

    if req.frame_idx >= session.num_frames:
        raise HTTPException(
            status_code=422,
            detail=f"frame_idx out of range: {req.frame_idx} >= {session.num_frames}",
        )

    predictor = model_holder.predictor
    if predictor is None:
        raise HTTPException(status_code=503, detail="predictor unavailable")

    state = session.inference_state
    results: dict[int, np.ndarray] = {}

    try:
        with torch.inference_mode(), torch.autocast(SAM2_DEVICE, dtype=torch.bfloat16):
            predictor.reset_state(state)
            predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=req.frame_idx,
                obj_id=0,
                box=req.bbox,
            )

            def collect(propagation):
                for frame_idx, _, masks in propagation:
                    mask = (masks[0, 0] > 0.0).cpu().numpy()
                    results[frame_idx] = mask

            collect(predictor.propagate_in_video(state, start_frame_idx=req.frame_idx))
            collect(
                predictor.propagate_in_video(state, start_frame_idx=req.frame_idx, reverse=True)
            )
    except Exception:
        logger.exception("segmentation failed")
        raise HTTPException(status_code=500, detail="segmentation failed")

    if not results:
        raise HTTPException(status_code=500, detail="no masks produced")

    masks_in_order = []
    for i in range(session.num_frames):
        if i in results:
            masks_in_order.append(results[i])
        else:
            masks_in_order.append(np.zeros((session.height, session.width), dtype=bool))

    fps = session.fps
    try:
        mp4_bytes = composite_overlay_to_mp4(
            original_video_path=session.video_path,
            masks_in_order=masks_in_order,
            fps=fps,
        )
    except Exception:
        logger.exception("composite encoding failed")
        raise HTTPException(status_code=500, detail="composite encoding failed")

    return Response(content=mp4_bytes, media_type="video/mp4")
