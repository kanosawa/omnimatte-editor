import asyncio
import logging

import numpy as np
from fastapi import APIRouter, HTTPException, Response

from server.casper import preload_casper
from server.full_foreground_store import full_foreground_store
from server.mask_store import mask_store
from server.model import DETECTRON2_IOU_WITH_TARGET
from server.sam_backend import sam_backend
from server.schemas import SegmentRequest
from server.session import session_slot
from server.video_io import composite_overlay_to_mp4


router = APIRouter()
logger = logging.getLogger(__name__)


# Casper の trimask 規約（mp4 ピクセル値）。Casper 内部でバケット化される:
#   < 64    → remove (1.0)
#   64-192  → neutral (0.5)
#   > 192   → keep (0.0)
TRIMASK_REMOVE = 0      # 対象前景（消す）
TRIMASK_NEUTRAL = 128   # 背景（neutral）
TRIMASK_KEEP = 255      # 他の前景（残す）


@router.post("/segment")
async def segment(req: SegmentRequest) -> Response:
    try:
        await sam_backend.wait_ready(timeout=5.0)
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

    # 診断用: 受け取ったパラメータと base video のパスをログに出す。
    # これを scripts/sam2_diagnose.py に同じ値で渡せば、実験コードパスでの結果と直接比較できる。
    logger.info(
        "segment request: frame_idx=%d bbox=%s session=(%dx%d) num_frames=%d base_video=%s",
        req.frame_idx, req.bbox, session.width, session.height, session.num_frames,
        session.base_video_path,
    )

    # 全前景抽出（バックグラウンド）の完了を待つ。タイムアウトは長めに
    try:
        await full_foreground_store.wait_ready(timeout=600.0)
    except TimeoutError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    full_fg = full_foreground_store.current()
    if full_fg is None:
        raise HTTPException(status_code=503, detail="full foreground data unavailable")
    if full_fg.base_video_path != session.base_video_path:
        raise HTTPException(status_code=409, detail="full foreground data is stale")

    state = session.inference_state
    results: dict[int, np.ndarray] = {}

    try:
        # 1. backend に bbox プロンプトを登録（SAM2/SAM3 とも video predictor の
        #    add_new_points_or_box に bbox を直接渡す）。
        #    呼び出し側はリセットしてから登録 → 順 / 逆 propagate するだけ。
        sam_backend.reset_state(state)
        sam_backend.add_bbox_prompt(
            state=state,
            frame_idx=req.frame_idx,
            obj_id=0,
            bbox=req.bbox,
            base_video_path=session.base_video_path,
            height=session.height,
            width=session.width,
        )

        for frame_idx, _obj_ids, masks in sam_backend.propagate(
            state, start_frame_idx=req.frame_idx, num_frames=session.num_frames,
        ):
            results[frame_idx] = masks[0]
        for frame_idx, _obj_ids, masks in sam_backend.propagate(
            state, start_frame_idx=req.frame_idx, num_frames=session.num_frames, reverse=True,
        ):
            results[frame_idx] = masks[0]
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
    masks_target = np.stack(masks_in_order, axis=0)  # (T, H, W) bool

    # R-CNN 検出物体から、対象前景と高 IoU のものを除外（領域単位）。
    # IoU は req.frame_idx 時点で計算（BBox を打ったフレーム）
    target_at_frame = masks_target[req.frame_idx]
    other_fg_combined = np.zeros_like(masks_target, dtype=bool)
    excluded = 0
    kept = 0
    for obj_masks in full_fg.per_object_masks:
        if obj_masks.shape != masks_target.shape:
            logger.warning(
                "skip object: shape mismatch %s vs %s",
                obj_masks.shape, masks_target.shape,
            )
            continue
        iou = _compute_iou(target_at_frame, obj_masks[req.frame_idx])
        if iou > DETECTRON2_IOU_WITH_TARGET:
            excluded += 1
            continue
        other_fg_combined |= obj_masks
        kept += 1
    logger.info(
        "trimask: target_at_frame_pixels=%d, other_fg_objects kept=%d excluded=%d",
        int(target_at_frame.sum()), kept, excluded,
    )

    # trimask 構築（mp4 ピクセル値で 3 値）。
    # 既定 = neutral、他の前景 = keep、対象前景 = remove（target が他より優先）
    trimask = np.full(masks_target.shape, TRIMASK_NEUTRAL, dtype=np.uint8)
    trimask[other_fg_combined & ~masks_target] = TRIMASK_KEEP
    trimask[masks_target] = TRIMASK_REMOVE

    fps = session.fps
    try:
        mp4_bytes = composite_overlay_to_mp4(
            original_video_path=session.base_video_path,
            masks_in_order=masks_in_order,
            fps=fps,
        )
    except Exception:
        logger.exception("composite encoding failed")
        raise HTTPException(status_code=500, detail="composite encoding failed")

    # /remove で再利用するため、生成した trimask をサーバ側に保持する
    mask_store.set(
        trimask=trimask,
        base_video_path=session.base_video_path,
        fps=fps,
    )

    # 先回り Casper 推論をバックグラウンドで起動（fire-and-forget）。
    # ユーザーが結果を眺めて「前景削除」ボタンを押すまでに本サーバ内で
    # 計算を済ませてキャッシュするので、/remove は cache hit で即返ることが期待される。
    asyncio.create_task(
        preload_casper(
            base_video_path=session.base_video_path,
            trimask=trimask,
            fps=fps,
            width=session.width,
            height=session.height,
        )
    )

    return Response(content=mp4_bytes, media_type="video/mp4")


def _compute_iou(a: np.ndarray, b: np.ndarray) -> float:
    """2 つのバイナリマスクの IoU。形状が違うときは 0 を返す。"""
    if a.shape != b.shape:
        return 0.0
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(inter) / float(union)
