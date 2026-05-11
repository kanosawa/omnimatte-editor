# omnimatte-editor

## 環境構築（初回のみ）

### システム要件

- Python 3.10+
- CUDA 対応 GPU + NVIDIA ドライバ
- **`ffmpeg` バイナリ**がシステム PATH に通っていること（`mediapy` が subprocess で呼ぶ）
  - Linux: `sudo apt install -y ffmpeg`
  - macOS: `brew install ffmpeg`
  - Windows: `winget install ffmpeg`

### 依存パッケージのインストール

**順序が重要**: 先に `requirements.txt`（冒頭で torch を CUDA 12.4 wheel に固定済み）→ 続いて Detectron2 をビルド、という順で入れる。これを逆順にすると、Detectron2 が古い torch を持ち込んだ後に他依存が「より新しい torch」を要求してアップグレードチェーンが走り、**CUDA 12 系と 13 系の nvidia-* ライブラリが混在してパフォーマンスが激しく劣化**する（cuDNN/cuBLAS の不整合で SDPA が math fallback に落ち、Casper の 1 diffusion step が秒単位で遅くなる）。

```bash
# 0. (推奨) 既存の torch / nvidia-* 系を一度クリーンにする
pip uninstall -y torch torchvision torchaudio \
    $(pip list 2>/dev/null | awk '/^nvidia-/ {print $1}')

# 1. 共通依存をインストール（torch 2.5.1+cu124 が requirements.txt 冒頭で
#    --extra-index-url 経由に固定されているので、ここで全部解決される）
pip install --upgrade pip setuptools wheel ninja
pip install -r requirements.txt

# 2. Detectron2 を git+ URL から手動でビルド。
#    setup.py が import 時に torch を要求するため build isolation を切る。
pip install --no-build-isolation \
    'git+https://github.com/facebookresearch/detectron2.git'

# 3. SAM2 の C++ 拡張ビルドを確認（editable インストール時にビルドされていない場合は再実行）
python -c "from sam2 import _C" 2>/dev/null || \
    pip install -e ./vendor/sam2 --no-build-isolation --force-reinstall
```

### 検証

```bash
python -c "
import torch
print('torch    :', torch.__version__)
print('cuda     :', torch.version.cuda)
print('cudnn    :', torch.backends.cudnn.version())
print('device   :', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')
from sam2 import _C  # noqa
print('sam2._C  : OK')
"
pip list | grep -Ei '^(torch|torchvision|torchaudio|nvidia-)'
```

期待する状態:
- `torch 2.5.1+cu124` / `torchvision 0.20.1+cu124` / `torchaudio 2.5.1+cu124`
- `nvidia-*-cu12` のみ（`-cu13` 系が混在していないこと）
- `from sam2 import _C` が例外なしで通る

### モデル重みのダウンロード

```bash
hf download alibaba-pai/Wan2.1-Fun-1.3B-InP \
    --local-dir vendor/gen-omnimatte-public/models/Diffusion_Transformer/Wan2.1-Fun-1.3B-InP
gdown "1n3Sv4d0pbTjfa5UhypEhTaylrSy2X4C1" \
    -O vendor/gen-omnimatte-public/models/Casper/wan2.1_fun_1.3b_casper.safetensors
```

## サーバ起動

```bash
python run.py
```

`127.0.0.1:8000` で待ち受ける（外部からは直接到達できない）。クラウドの GPU サーバで動かす場合も同じコマンドで、クライアントからは SSH ポート転送経由で接続する。アクセス制御はサーバ側のネットワーク設定（SSH のみ開放）で担保する。

`OMNIMATTE_PORT` 環境変数でポートを変更可（既定 `8000`）。

### GPU サーバ運用（SSH トンネル経由）

サーバ側:

```bash
python run.py
```

クライアント側で別ターミナルにトンネルを張る:

```bash
ssh -p <SSH_PORT> -N -L 8000:127.0.0.1:8000 user@gpu-server
```

| 引数 | 意味 |
|---|---|
| `-p <SSH_PORT>` | gpu-server の SSH サーバが listen しているポート（クラウド GPU では非標準ポートが使われることが多い。22 番なら省略可） |
| `-N` | リモートでコマンドを実行せず、トンネルだけ張る |
| `-L 8000:127.0.0.1:8000` | ローカル `8000` → gpu-server から見た `127.0.0.1:8000`（= omnimatte-editor）へ転送 |

サーバ側で `OMNIMATTE_PORT=9000 python run.py` のようにポートを変えている場合は、右側 2 つの数字を合わせる:

```bash
ssh -p <SSH_PORT> -N -L 8000:127.0.0.1:9000 user@gpu-server
```

左側（ローカル）を `8000` のままにしておけば、フロントの `VITE_API_BASE` は触らずに済む。

> Casper sidecar（GPU マシン内部の `127.0.0.1:8765`）は本サーバが内部で呼ぶため、SSH トンネルは `8000` だけで OK。

フロント側は `frontend/.env` の `VITE_API_BASE` をサーバ URL に設定する。詳細は [frontend/README.md](frontend/README.md)。

モデルDL
hf download alibaba-pai/Wan2.1-Fun-1.3B-InP --local-dir models/Diffusion_Transformer/Wan2.1-Fun-1.3B-InP
gdown "1n3Sv4d0pbTjfa5UhypEhTaylrSy2X4C1" -O models/Casper/wan2.1_fun_1.3b_casper.safetensors