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

# bbox 周辺をクロップする時のマージン（bbox サイズに対する比率）
_CROP_MARGIN_RATIO = 0.5
# クロップ後の画像を upscale する目標サイズ（長辺）。SAM2 内部は 1024 で動くため、
# bbox 周辺の effective 解像度を上げて detail を引き出す
_UPSCALE_TARGET_LONG_SIDE = 1024


def _refine_mask_iteratively_for_bbox(
    video_predictor,
    base_video_path: str,
    frame_idx: int,
    bbox: list[float],
) -> np.ndarray:
    """SAM2 画像プレディクタで反復補正してマスクを得る（案 Y + 案 A: bbox crop + upscale）。

    アルゴリズム:
      1. bbox 周辺 + マージンでフレームをクロップ
      2. クロップを長辺 1024 に upscale（cv2.INTER_CUBIC）— SAM2 の内部解像度に
         合わせて effective 解像度を上げ、低解像度動画でも精度を確保
      3. 拡大後のクロップ + 変換した bbox で SAM2 image predictor を実行 → M0
      4. M0 が bbox を埋める比率（fill_ratio）を計算
      5. fill_ratio >= threshold なら M0 を採用
      6. それ以外は「bbox 内かつ M0 外」の連結成分の重心を positive point として
         追加し再予測 → M1。M1 が M0 より良ければ M1、そうでなければ M0
      7. 最終マスクを元解像度にダウンスケールし、原座標に貼り戻す
    """
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    # video predictor は SAM2Base のサブクラスなので、そのまま image predictor の
    # ベースモデルとして共有できる（GPU メモリの二重ロード回避）
    image_predictor = SAM2ImagePredictor(video_predictor)

    frame_bgr = read_frame_at(base_video_path, frame_idx)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h_orig, w_orig = frame_rgb.shape[:2]

    # 1. bbox 周辺 + マージンでクロップ
    x1, y1, x2, y2 = [float(v) for v in bbox]
    bw = x2 - x1
    bh = y2 - y1
    cx1 = max(0, int(round(x1 - bw * _CROP_MARGIN_RATIO)))
    cy1 = max(0, int(round(y1 - bh * _CROP_MARGIN_RATIO)))
    cx2 = min(w_orig, int(round(x2 + bw * _CROP_MARGIN_RATIO)))
    cy2 = min(h_orig, int(round(y2 + bh * _CROP_MARGIN_RATIO)))
    if cx2 <= cx1 or cy2 <= cy1:
        raise RuntimeError(f"invalid crop region: ({cx1},{cy1})-({cx2},{cy2})")
    crop = frame_rgb[cy1:cy2, cx1:cx2]
    crop_h, crop_w = crop.shape[:2]

    # 2. 長辺 1024 に upscale（小さい場合のみ）
    long_side = max(crop_h, crop_w)
    if long_side < _UPSCALE_TARGET_LONG_SIDE:
        scale = _UPSCALE_TARGET_LONG_SIDE / long_side
        new_w = int(round(crop_w * scale))
        new_h = int(round(crop_h * scale))
        crop_proc = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    else:
        scale = 1.0
        crop_proc = crop
    proc_h, proc_w = crop_proc.shape[:2]

    # 3. bbox を crop+scale 座標系に変換
    bbox_np = np.array(
        [(x1 - cx1) * scale, (y1 - cy1) * scale, (x2 - cx1) * scale, (y2 - cy1) * scale],
        dtype=np.float32,
    )
    bx1 = max(0, int(round(bbox_np[0])))
    by1 = max(0, int(round(bbox_np[1])))
    bx2 = min(proc_w, int(round(bbox_np[2])))
    by2 = min(proc_h, int(round(bbox_np[3])))
    bbox_clipped_proc = (bx1, by1, bx2, by2)
    bbox_area = max(1, (bx2 - bx1) * (by2 - by1))

    logger.info(
        "refine: frame=%dx%d crop=%dx%d (margin=%.1f) → proc=%dx%d (scale=%.2f)",
        w_orig, h_orig, crop_w, crop_h, _CROP_MARGIN_RATIO, proc_w, proc_h, scale,
    )

    with torch.inference_mode(), torch.autocast(SAM2_DEVICE, dtype=torch.bfloat16):
        image_predictor.set_image(crop_proc)

        # 4. 初回予測（bbox のみ、SAM2 既定の選択に任せる）
        masks_0, _scores_0, _ = image_predictor.predict(
            box=bbox_np,
            multimask_output=False,
        )
        if len(masks_0) == 0:
            raise RuntimeError("SAM2 image predictor returned no mask")
        mask_0_proc = masks_0[0].astype(bool)

        in_bbox_0 = int(mask_0_proc[by1:by2, bx1:bx2].sum())
        ratio_0 = in_bbox_0 / bbox_area
        logger.info("refine: initial fill_ratio=%.3f (threshold=%.3f)", ratio_0, _REFINE_FILL_RATIO_THRESHOLD)

        if ratio_0 >= _REFINE_FILL_RATIO_THRESHOLD:
            mask_proc = mask_0_proc
        else:
            # 5. サブコンポーネント疑い → 追加 positive point を探す
            refine_points = _find_refinement_points(mask_0_proc, bbox_clipped_proc, bbox_area)
            if not refine_points:
                logger.info("refine: no refinement points found, using initial mask")
                mask_proc = mask_0_proc
            else:
                point_coords = np.array(refine_points, dtype=np.float32)
                point_labels = np.ones(len(refine_points), dtype=np.int32)  # 全て positive
                logger.info("refine: re-predicting with %d positive points: %s",
                            len(refine_points),
                            [(f"{p[0]:.0f}", f"{p[1]:.0f}") for p in refine_points])

                # 6. 再予測（bbox + 追加 positive points）
                masks_1, _scores_1, _ = image_predictor.predict(
                    box=bbox_np,
                    point_coords=point_coords,
                    point_labels=point_labels,
                    multimask_output=False,
                )
                mask_1_proc = masks_1[0].astype(bool)

                in_bbox_1 = int(mask_1_proc[by1:by2, bx1:bx2].sum())
                ratio_1 = in_bbox_1 / bbox_area
                logger.info("refine: refined fill_ratio=%.3f", ratio_1)

                # サニティチェック: 補正で fill_ratio が下がったら初回マスクに戻す
                if ratio_1 < ratio_0:
                    logger.info("refine: refinement reduced fill_ratio (%.3f -> %.3f), falling back",
                                ratio_0, ratio_1)
                    mask_proc = mask_0_proc
                else:
                    mask_proc = mask_1_proc

    # 7. 元解像度にダウンスケールして原座標に貼り戻す
    if scale != 1.0:
        mask_crop = cv2.resize(
            mask_proc.astype(np.uint8), (crop_w, crop_h), interpolation=cv2.INTER_NEAREST
        ).astype(bool)
    else:
        mask_crop = mask_proc

    full_mask = np.zeros((h_orig, w_orig), dtype=bool)
    full_mask[cy1:cy2, cx1:cx2] = mask_crop
    return full_mask


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
