#!/usr/bin/env bash
# backend の依存パッケージとモデル重みを一括セットアップする (Ubuntu 想定)。
# 事前に torch / torchvision / torchaudio を CUDA 環境に合わせて入れておくこと。

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
BACKEND_DIR="$(dirname "$HERE")"
VENDOR_DIR="$BACKEND_DIR/vendor/gen-omnimatte-public"

# --- 依存パッケージ ---
# detectron2 は `iopath<0.1.10` を、sam-2 は `iopath>=0.1.10` を要求して衝突するため、
# detectron2 だけ --no-deps で入れる (必要な deps は requirements.txt に明示)。
pip install --no-build-isolation -r "$HERE/requirements.txt"
pip install --no-build-isolation --no-deps \
    'git+https://github.com/facebookresearch/detectron2.git@b599f139756bd3646a26a909caf86a1a159e53a7'

# --- モデル重み (既存ファイルは各ツールが自動でスキップ) ---
mkdir -p "$BACKEND_DIR/models/sam2" "$VENDOR_DIR/models/Casper"

wget -nc -P "$BACKEND_DIR/models/sam2" \
    https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

HF_HUB_ENABLE_HF_TRANSFER=1 hf download alibaba-pai/Wan2.1-Fun-1.3B-InP \
    --local-dir "$VENDOR_DIR/models/Diffusion_Transformer/Wan2.1-Fun-1.3B-InP"

CASPER_SAFE="$VENDOR_DIR/models/Casper/wan2.1_fun_1.3b_casper.safetensors"
[ -f "$CASPER_SAFE" ] || gdown "1n3Sv4d0pbTjfa5UhypEhTaylrSy2X4C1" -O "$CASPER_SAFE"
