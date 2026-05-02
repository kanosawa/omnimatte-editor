# omnimatte-editor

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
ssh -N -L 8000:127.0.0.1:8000 user@gpu-server
```

フロント側は `frontend/.env` の `VITE_API_BASE` をサーバ URL に設定する。詳細は [frontend/README.md](frontend/README.md)。
