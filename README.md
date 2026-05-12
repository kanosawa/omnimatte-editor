# omnimatte-editor

## 環境構築（初回のみ）

### システム要件

- Python 3.10+
- CUDA 対応 GPU + NVIDIA ドライバ
- **`ffmpeg` バイナリ**がシステム PATH に通っていること（`mediapy` が subprocess で呼ぶ）
  - Linux: `sudo apt install -y ffmpeg`
  - macOS: `brew install ffmpeg`
  - Windows: `winget install ffmpeg`

### 推奨環境（torch / CUDA）

torch / torchvision / torchaudio / CUDA は `requirements.txt` で固定していない。ユーザが自分の GPU と CUDA に合わせて先に install する。

**推奨**:
- `torch 2.5.x` + CUDA 12.4 系（`torch==2.5.1+cu124` で動作確認済み）
- 推奨から大きく離れなければ、数年は本リポジトリの他の依存と問題なく組み合わせ可能

**避けるべき構成**:
- CUDA 12 系と 13 系の `nvidia-*` ライブラリ混在（cuDNN / cuBLAS の不整合で SDPA が math fallback に落ち、Casper の 1 diffusion step が秒単位で遅くなる）
- 大きく離れた torch バージョン（後述 `transformers<5` 制約により torch 2.7+ は組み合わせ困難）

### 依存パッケージのインストール

```bash
# 0. (推奨) 既存の torch / nvidia-* 系を一度クリーンにする
pip uninstall -y torch torchvision torchaudio \
    $(pip list 2>/dev/null | awk '/^nvidia-/ {print $1}')

# 1. ビルドツールを最新化
pip install --upgrade pip setuptools wheel ninja

# 2. torch を先に install する（推奨: 2.5.x + CUDA 12.4 系。
#    GPU / CUDA バージョンに合わせて --index-url を選ぶ）
pip install torch~=2.5.0 torchvision~=0.20.0 torchaudio~=2.5.0 \
    --index-url https://download.pytorch.org/whl/cu124

# 3. 残りの依存（sam-2 + detectron2）+ AI モデル重みを一括セットアップする。
#    setup.sh 内で 3 段に分かれている:
#      - requirements.txt の install
#      - detectron2 の --no-deps install (iopath 制約衝突を回避するため)
#      - SAM2 / Casper の重みダウンロード (wget / hf / gdown を直接呼ぶ)
#    Ubuntu 想定。
bash backend/scripts/setup.sh
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
- `torch` / `torchvision` / `torchaudio` が全て同じ CUDA タグ（例 `+cu124`）で揃っている
- `nvidia-*` ライブラリの CUDA メジャーが混在していない（`-cu12` 系で揃う、など）
- `from sam2 import _C` が例外なしで通る

### モデル重みのダウンロード先

`setup.sh` がまとめてダウンロードするので通常は気にしなくてよいが、参考として:

- SAM2: `backend/models/sam2/sam2.1_hiera_large.pt`
- Casper Diffusion Transformer: `backend/vendor/gen-omnimatte-public/models/Diffusion_Transformer/Wan2.1-Fun-1.3B-InP/`
- Casper safetensors: `backend/vendor/gen-omnimatte-public/models/Casper/wan2.1_fun_1.3b_casper.safetensors`

既に存在するファイルはスキップするので、途中で失敗しても `bash backend/scripts/setup.sh` を再実行すれば続きから取得できる。

## サーバ起動

```bash
cd backend
python run.py
```

`127.0.0.1:8000` で待ち受ける（外部からは直接到達できない）。クラウドの GPU サーバで動かす場合も同じコマンドで、クライアントからは SSH ポート転送経由で接続する。アクセス制御はサーバ側のネットワーク設定（SSH のみ開放）で担保する。

`OMNIMATTE_PORT` 環境変数でポートを変更可（既定 `8000`）。

### GPU サーバ運用（SSH トンネル経由）

サーバ側:

```bash
cd backend
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