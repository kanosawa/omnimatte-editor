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
