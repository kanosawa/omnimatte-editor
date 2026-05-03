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
        # 1. 反復補正で初期マスクを得る（SAM2 image predictor 1〜2 回）。
        #    初回は bbox のみで予測し、サブコンポーネント疑い（fill_ratio < 閾値）なら
        #    「bbox 内かつマスク外」の最大連結成分の重心に positive point を追加して再予測。
        best_initial_mask = _refine_mask_iteratively_for_bbox(
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


# 反復補正の閾値: bbox 内に対するマスク面積の比がこれ未満なら
# 「サブコンポーネント疑い」とみなして positive point を追加して再予測する
_REFINE_FILL_RATIO_THRESHOLD = 0.5
# 追加する positive point の最大数
_REFINE_MAX_POINTS = 3
# 追加候補とする「bbox 内かつマスク外」の連結成分の最小面積（bbox 面積比）
_REFINE_MIN_COMPONENT_AREA_RATIO = 0.02


def _refine_mask_iteratively_for_bbox(
    video_predictor,
    base_video_path: str,
    frame_idx: int,
    bbox: list[float],
) -> np.ndarray:
    """SAM2 画像プレディクタで反復補正してマスクを得る（案 Y）。

    アルゴリズム:
      1. bbox のみで予測 → 初回マスク M0
      2. M0 が bbox を埋める比率（fill_ratio）を計算
      3. fill_ratio >= threshold なら M0 を採用してリターン
      4. 「bbox 内かつ M0 外」の連結成分のうち面積上位を抽出、それぞれの重心を
         positive point として追加し再予測 → M1
      5. M1 が M0 より bbox を埋めれば M1、そうでなければ M0 を採用

    案 X（multimask 候補から fill ratio 最大を選ぶ）の弊害（背景まで含む候補が
    選ばれる）を避け、「最初に SAM2 が選んだ自然なマスク」を出発点にして必要時
    だけ拡張する保守的な戦略。
    """
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    # video predictor は SAM2Base のサブクラスなので、そのまま image predictor の
    # ベースモデルとして共有できる（GPU メモリの二重ロード回避）
    image_predictor = SAM2ImagePredictor(video_predictor)

    frame_bgr = read_frame_at(base_video_path, frame_idx)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    bbox_np = np.array(bbox, dtype=np.float32)
    h, w = frame_rgb.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w, x2); y2 = min(h, y2)
    bbox_area = max(1, (x2 - x1) * (y2 - y1))

    with torch.inference_mode(), torch.autocast(SAM2_DEVICE, dtype=torch.bfloat16):
        image_predictor.set_image(frame_rgb)

        # 1. 初回予測（bbox のみ、SAM2 既定の選択に任せる）
        masks_0, _scores_0, _ = image_predictor.predict(
            box=bbox_np,
            multimask_output=False,
        )
        if len(masks_0) == 0:
            raise RuntimeError("SAM2 image predictor returned no mask")
        mask_0 = masks_0[0].astype(bool)

        in_bbox_0 = int(mask_0[y1:y2, x1:x2].sum())
        ratio_0 = in_bbox_0 / bbox_area
        logger.info("refine: initial fill_ratio=%.3f (threshold=%.3f)", ratio_0, _REFINE_FILL_RATIO_THRESHOLD)

        if ratio_0 >= _REFINE_FILL_RATIO_THRESHOLD:
            return mask_0

        # 2. サブコンポーネント疑い → 追加 positive point を探す
        refine_points = _find_refinement_points(mask_0, (x1, y1, x2, y2), bbox_area)
        if not refine_points:
            logger.info("refine: no refinement points found, using initial mask")
            return mask_0

        point_coords = np.array(refine_points, dtype=np.float32)
        point_labels = np.ones(len(refine_points), dtype=np.int32)  # 全て positive
        logger.info("refine: re-predicting with %d positive points: %s",
                    len(refine_points),
                    [(f"{p[0]:.0f}", f"{p[1]:.0f}") for p in refine_points])

        # 3. 再予測（bbox + 追加 positive points）
        masks_1, _scores_1, _ = image_predictor.predict(
            box=bbox_np,
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=False,
        )
        mask_1 = masks_1[0].astype(bool)

        in_bbox_1 = int(mask_1[y1:y2, x1:x2].sum())
        ratio_1 = in_bbox_1 / bbox_area
        logger.info("refine: refined fill_ratio=%.3f", ratio_1)

        # 4. サニティチェック: 補正で fill_ratio が下がったら初回マスクに戻す
        if ratio_1 < ratio_0:
            logger.info("refine: refinement reduced fill_ratio (%.3f -> %.3f), falling back",
                        ratio_0, ratio_1)
            return mask_0

        return mask_1


def _find_refinement_points(
    mask: np.ndarray,
    bbox_clipped: tuple[int, int, int, int],
    bbox_area: int,
) -> list[tuple[float, float]]:
    """「bbox 内かつ mask 外」の領域から positive point の候補を抽出する。

    アルゴリズム:
      1. (in_bbox AND not mask) のバイナリを作る
      2. 連結成分に分割（cv2.connectedComponentsWithStats）
      3. 面積上位、かつ bbox 面積比 >= 閾値の成分を最大 N 個選ぶ
      4. 各成分の重心を point として返す。重心が成分外（凹形状）なら成分内の
         代表ピクセルにフォールバック

    返り値: `[(x, y), ...]` のピクセル座標リスト
    """
    h, w = mask.shape
    x1, y1, x2, y2 = bbox_clipped

    bbox_region = np.zeros_like(mask, dtype=np.uint8)
    bbox_region[y1:y2, x1:x2] = 1
    candidate = bbox_region & (~mask).astype(np.uint8)
    if candidate.sum() == 0:
        return []

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(candidate, connectivity=8)
    if n_labels <= 1:  # label 0 は背景
        return []

    min_area = max(1, int(_REFINE_MIN_COMPONENT_AREA_RATIO * bbox_area))

    # ラベル 1.. の (area, idx) を面積降順
    components = sorted(
        ((int(stats[i, cv2.CC_STAT_AREA]), i) for i in range(1, n_labels)),
        reverse=True,
    )

    points: list[tuple[float, float]] = []
    for area, idx in components:
        if area < min_area:
            break
        cx, cy = centroids[idx]
        cx_int = int(round(cx))
        cy_int = int(round(cy))
        # 重心が成分内にあればそのまま使う。凹形状で外側に落ちる場合は成分内
        # の代表ピクセル（中央値）にフォールバック
        if 0 <= cy_int < h and 0 <= cx_int < w and labels[cy_int, cx_int] == idx:
            points.append((float(cx), float(cy)))
        else:
            ys, xs = np.where(labels == idx)
            if len(xs) > 0:
                mid = len(xs) // 2
                points.append((float(xs[mid]), float(ys[mid])))
        if len(points) >= _REFINE_MAX_POINTS:
            break

    return points
