#!/usr/bin/env bash
# backend の依存パッケージとモデル重みを一括セットアップする (Ubuntu 想定)。
# 事前に torch / torchvision / torchaudio をユーザの CUDA 環境に合わせて入れておくこと。
# 詳細は ../../README.md 参照。
#
# pip install を 2 段に分けている理由:
#   - sam-2 は `iopath>=0.1.10` を要求
#   - detectron2 は `iopath>=0.1.7,<0.1.10` を要求
#   両者を 1 つの `pip install` で解決すると ResolutionImpossible になるため、
#   detectron2 だけ別ファイル + --no-deps で install し、iopath 制約を回避する。
#   detectron2 の他の deps は requirements.txt 側に明示的に並べてある。

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
BACKEND_DIR="$(dirname "$HERE")"

echo "==> [1/3] requirements.txt を install (sam-2 含む)"
pip install --no-build-isolation -r "$HERE/requirements.txt"

echo "==> [2/3] detectron2 を --no-deps で install"
pip install --no-build-isolation --no-deps -r "$HERE/requirements-detectron2.txt"

echo "==> [3/3] AI モデル重みをダウンロード"

# --- SAM2 (sam2.1_hiera_large) ---
SAM2_DIR="$BACKEND_DIR/models/sam2"
SAM2_PT="$SAM2_DIR/sam2.1_hiera_large.pt"
if [ -f "$SAM2_PT" ]; then
    echo "    [sam2] already exists: $SAM2_PT"
else
    echo "    [sam2] downloading..."
    mkdir -p "$SAM2_DIR"
    wget -O "$SAM2_PT" \
        https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
fi

# --- Casper Diffusion Transformer (Wan2.1-Fun-1.3B-InP) ---
# hf cache を経由するので二度目以降は実体ダウンロードがスキップされる。
CASPER_HF_DIR="$BACKEND_DIR/vendor/gen-omnimatte-public/models/Diffusion_Transformer/Wan2.1-Fun-1.3B-InP"
echo "    [casper-hf] hf download alibaba-pai/Wan2.1-Fun-1.3B-InP -> $CASPER_HF_DIR"
mkdir -p "$CASPER_HF_DIR"
HF_HUB_ENABLE_HF_TRANSFER=1 hf download alibaba-pai/Wan2.1-Fun-1.3B-InP \
    --local-dir "$CASPER_HF_DIR"

# --- Casper safetensors (Google Drive) ---
CASPER_SAFE="$BACKEND_DIR/vendor/gen-omnimatte-public/models/Casper/wan2.1_fun_1.3b_casper.safetensors"
if [ -f "$CASPER_SAFE" ]; then
    echo "    [casper-safetensors] already exists: $CASPER_SAFE"
else
    echo "    [casper-safetensors] downloading via gdown..."
    mkdir -p "$(dirname "$CASPER_SAFE")"
    gdown "1n3Sv4d0pbTjfa5UhypEhTaylrSy2X4C1" -O "$CASPER_SAFE"
fi

echo "==> done"
