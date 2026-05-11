import logging
import os


logger = logging.getLogger(__name__)

_here = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_here)

# ---------- SAM2 ----------
SAM2_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
SAM2_CKPT = os.path.join(_project_root, "models", "sam2", "sam2.1_hiera_large.pt")
SAM2_DEVICE = "cuda"

# ---------- Casper（gen-omnimatte-public, Wan2.1-Fun-1.3B-InP）----------
CASPER_REPO_DIR = os.path.join(_project_root, "vendor", "gen-omnimatte-public")
CASPER_TRANSFORMER_PATH = os.path.join(
    CASPER_REPO_DIR, "models", "Casper", "wan2.1_fun_1.3b_casper.safetensors"
)
CASPER_CONFIG_PATH = "config/default_wan.py"  # CASPER_REPO_DIR からの相対
# 推論解像度は固定値ではなく、リクエストごとに base video の解像度から動的に決定する
# （server.casper 側で 16 の倍数に丸める）
CASPER_FPS = 8
CASPER_NUM_INFERENCE_STEPS = 1
CASPER_TEMPORAL_WINDOW_SIZE = 21
CASPER_MATTING_MODE = "all_fg"
CASPER_DEFAULT_PROMPT = "a clean background video."

# ---------- Detectron2 (COCO Mask R-CNN) ----------
DETECTRON2_CONFIG = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
DETECTRON2_DEVICE = "cuda"
DETECTRON2_SCORE_THRESH = 0.5
DETECTRON2_MAX_DETECTIONS = 5      # 検出数の上限（area 降順で上位）
DETECTRON2_MIN_AREA_RATIO = 0.001  # 画像面積の 0.1% 未満は誤検出として弾く
DETECTRON2_IOU_WITH_TARGET = 0.3   # 対象前景との IoU がこれを超えた検出は対象本人として除外

# 前景部分のみ Casper 出力で書き換える後処理。
# OFF にすると Casper の出力 mp4 をそのまま返す（デバッグ・品質比較用）
FOREGROUND_ONLY_REPLACE = os.environ.get("OMNIMATTE_FOREGROUND_ONLY_REPLACE", "0") != "0"
# 判定マスクの diff 閾値（per-channel 0-255）。SAM2 マスク（dilate 後）と AND を取る
FG_REPLACE_DIFF_THRESHOLD = int(os.environ.get("OMNIMATTE_FG_REPLACE_DIFF_THRESHOLD", "10"))
# SAM2 マスクの dilate 量（ピクセル）。Casper の cfg.data.dilate_width=11 と一致
FG_REPLACE_MASK_DILATE = int(os.environ.get("OMNIMATTE_FG_REPLACE_MASK_DILATE", "11"))

# /remove で casper.wait_ready を待つ最大秒数（SAM と統一）
CASPER_STARTUP_TIMEOUT_SEC = float(os.environ.get("CASPER_STARTUP_TIMEOUT_SEC", "5.0"))
