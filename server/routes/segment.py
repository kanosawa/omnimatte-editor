import asyncio
import logging

import cv2
import numpy as np
import torch
from fastapi import APIRouter, HTTPException, Response

from server.casper_client import preload_casper
from server.full_foreground_store import full_foreground_store
from server.mask_store import mask_store
from server.model import (
    DETECTRON2_IOU_WITH_TARGET,
    SAM2_DEVICE,
    model_holder,
)
from server.schemas import SegmentRequest
from server.session import session_slot
from server.video_io import composite_overlay_to_mp4, read_frame_at


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

    predictor = model_holder.predictor
    if predictor is None:
        raise HTTPException(status_code=503, detail="predictor unavailable")

    state = session.inference_state
    results: dict[int, np.ndarray] = {}

    try:
        # 1. 画像プレディクタで multimask 候補を取得し、ユーザー bbox を最も埋める候補を採用。
        #    SAM2 既定の「IoU スコア最高」選択ではサブパーツが拾われがちなため。
        best_initial_mask = _select_best_mask_candidate_for_bbox(
            video_predictor=predictor,
            base_video_path=session.base_video_path,
            frame_idx=req.frame_idx,
            bbox=req.bbox,
        )

        # 2. 採用したマスクを video predictor に登録 → 順方向 + 逆方向に propagate
        with torch.inference_mode(), torch.autocast(SAM2_DEVICE, dtype=torch.bfloat16):
            predictor.reset_state(state)
            predictor.add_new_mask(
                inference_state=state,
                frame_idx=req.frame_idx,
                obj_id=0,
                mask=best_initial_mask,
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

    # 先回り Casper 推論を sidecar にバックグラウンドで依頼。
    # ユーザーが結果を眺めて「前景削除」ボタンを押すまでに sidecar が
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


def _select_best_mask_candidate_for_bbox(
    video_predictor,
    base_video_path: str,
    frame_idx: int,
    bbox: list[float],
) -> np.ndarray:
    """SAM2 画像プレディクタで multimask 候補 3 つを取得し、ユーザー bbox を
    最も埋める候補を選んで返す。

    SAM2 既定の「IoU スコア最高」選択は、bbox 内のサブパーツ（顔だけ、車輪だけ等）が
    高スコアになり物体全体が拾えないケースが少なくない。bbox 内に占める面積比で
    選び直すことで、ユーザーの意図に沿ったマスクを採用する。

    返り値: `(H, W) bool` の選ばれたマスク。`video_predictor.add_new_mask` に渡す。
    """
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    # video predictor は SAM2Base のサブクラスなので、そのまま image predictor の
    # ベースモデルとして共有できる（GPU メモリの二重ロード回避）
    image_predictor = SAM2ImagePredictor(video_predictor)

    frame_bgr = read_frame_at(base_video_path, frame_idx)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    bbox_np = np.array(bbox, dtype=np.float32)
    with torch.inference_mode(), torch.autocast(SAM2_DEVICE, dtype=torch.bfloat16):
        image_predictor.set_image(frame_rgb)
        masks, scores, _ = image_predictor.predict(
            box=bbox_np,
            multimask_output=True,
        )
    # masks: (N, H, W) uint8 or bool; scores: (N,) float

    if len(masks) == 0:
        raise RuntimeError("SAM2 image predictor returned no candidates")

    # bbox を画像範囲にクリップ
    h, w = masks.shape[1], masks.shape[2]
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w, x2); y2 = min(h, y2)
    bbox_area = max(1, (x2 - x1) * (y2 - y1))

    ratios = []
    for m in masks:
        m_bool = m.astype(bool) if m.dtype != bool else m
        in_bbox = m_bool[y1:y2, x1:x2].sum()
        ratios.append(float(in_bbox) / float(bbox_area))

    best_idx = int(np.argmax(ratios))
    best_mask = masks[best_idx].astype(bool) if masks[best_idx].dtype != bool else masks[best_idx]
    logger.info(
        "multimask candidates: scores=%s fill_ratios=%s -> picked idx=%d",
        [f"{s:.3f}" for s in scores.tolist()],
        [f"{r:.3f}" for r in ratios],
        best_idx,
    )
    return best_mask
