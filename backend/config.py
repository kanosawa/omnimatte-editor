DETECTRON2_SCORE_THRESH = 0.5
DETECTRON2_MAX_DETECTIONS = 5      # 検出数の上限（area 降順で上位）
DETECTRON2_MIN_AREA_RATIO = 0.001  # 画像面積の 0.1% 未満は誤検出として弾く
DETECTRON2_IOU_WITH_TARGET = 0.3   # 対象前景との IoU がこれを超えた検出は対象本人として除外

# /remove で casper.wait_ready を待つ最大秒数（SAM と統一）
CASPER_STARTUP_TIMEOUT_SEC = 5.0
