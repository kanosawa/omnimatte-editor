# omnimatte-editor

## SAM backend の選択

`OMNIMATTE_SAM_VERSION` 環境変数で SAM2 / SAM3 を切替できる（既定: `2` = SAM2）。

| 値 | backend | 必要環境 |
|---|---|---|
| `2` | SAM2 (`vendor/sam2`) | Python 3.10+, PyTorch 2.x |
| `3` | SAM3 (`vendor/sam3`) | **Python 3.12+, PyTorch 2.7+, CUDA 12.6+** |

### SAM3 セットアップ（初回のみ）

```bash
# 1. Python 3.12+ の venv を別途用意（既存 venv は SAM3 非互換）
python3.12 -m venv .venv-sam3
source .venv-sam3/bin/activate   # Windows: .venv-sam3\Scripts\activate

# 2. 共通依存をインストール
pip install -r requirements.txt

# 3. vendor/sam3 のクローン + pip install + 重み（sam3.safetensors）DL
python scripts/setup_sam3.py
```

`scripts/setup_sam3.py` は以下を行う:
1. `git submodule update --init vendor/sam3`
2. `pip install -e vendor/sam3`
3. [AEmotionStudio/sam3](https://huggingface.co/AEmotionStudio/sam3)（HF, non-gated mirror）から `sam3.safetensors` を `vendor/sam3/checkpoints/` に取得

オプション: `--no-install`（pip install スキップ）、`--skip-weights`（重み DL スキップ）。

### 起動例

```bash
# SAM2（既定）
python run.py

# SAM3（環境構築済み前提）
OMNIMATTE_SAM_VERSION=3 python run.py
```

`/health` レスポンスの `samVersion` フィールドで現在の選択を確認できる。

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
