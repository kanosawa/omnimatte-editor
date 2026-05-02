# 04. API 仕様

## 4.1 概要

フロントエンド⇔バックエンド間のすべての通信は HTTP。すべて FastAPI が提供する。

- ベース URL: フロントエンドの環境変数 `VITE_API_BASE`（既定値 `http://localhost:8000`）
- 認証: 本仕様では非対応
- 文字コード: JSON は UTF-8

## 4.2 エンドポイント一覧

| パス | メソッド | 概要 |
|---|---|---|
| `/health` | GET | 稼働確認とモデルロード状態 |
| `/session` | POST | mp4 アップロード（multipart） |
| `/segment` | POST | BBox を指定して原動画＋マスク半透明合成済み mp4 を取得 |

## 4.3 `/health`

### Request

`GET /health`

### Response (200)

```json
{
  "status": "ok",
  "model_state": "loading",
  "session_active": false
}
```

| フィールド | 型 | 説明 |
|---|---|---|
| `status` | string | `"ok"` 固定 |
| `model_state` | string | `"loading"` / `"ready"` / `"failed"` |
| `session_active` | boolean | 現在セッションが存在するか（バックエンドは常に最大1件） |

### 用途

- バックエンド側でも参照可能な外形監視用エンドポイント
- フロントエンド本体はポーリングしない（`/session` 呼び出し時にバックエンドが `wait_ready(5.0)` でロード完了を待ち合わせるため）

## 4.4 `/session`

### Request

`POST /session`

- Content-Type: `multipart/form-data`
- Field: `video` （mp4 ファイルバイナリ）

### Response (200)

```json
{
  "videoMeta": {
    "width": 1920,
    "height": 1080,
    "fps": 29.97,
    "numFrames": 600,
    "durationSec": 20.02
  }
}
```

| フィールド | 型 | 説明 |
|---|---|---|
| `videoMeta.width` / `videoMeta.height` | number | ピクセル単位 |
| `videoMeta.fps` | number | 浮動小数点 |
| `videoMeta.numFrames` | number | 総フレーム数 |
| `videoMeta.durationSec` | number | `numFrames / fps` |

セッション ID は返さない。バックエンドは常に最大 1 件のセッションだけを保持し、新規 `/session` 呼び出し時は直前のセッションが自動的に破棄される（[03-backend.md §3.5.2](03-backend.md#352-ライフタイム)）。

### Error

| HTTP | 内容 |
|---|---|
| 400 | mp4 が読み込めない |
| 500 | セッション作成失敗（内部例外） |
| 503 | モデルロード未完了（5秒タイムアウト）またはロード失敗 |

### フロントエンドの取り扱い

- レスポンスの `video_meta` は zustand ストアに保存し、UI 表示（時間・総フレーム数）に利用
- `videoMeta !== null` を「セッション保持中」のフラグとして使う（`session_id` を持たないため）

## 4.5 `/segment`

### Request

`POST /segment`

- Content-Type: `application/json`
- Body:

```json
{
  "frame_idx": 42,
  "bbox": [120.5, 80.0, 540.0, 380.5]
}
```

| フィールド | 型 | 説明 |
|---|---|---|
| `frame_idx` | number (int) | BBox を指定したフレーム番号（0始まり） |
| `bbox` | number[4] | `[x1, y1, x2, y2]` 動画ピクセル座標。`x2 > x1`、`y2 > y1` |

セッションは常にバックエンド側の現在スロットを暗黙的に参照する（`session_id` フィールドは存在しない）。

### Response (200)

- Content-Type: `video/mp4`
- Body: 原動画にマスクを半透明＋着色合成済みの mp4 バイナリ（H.264 / yuv420p）

### Error

| HTTP | 内容 |
|---|---|
| 409 | バックエンドにセッションが存在しない（`/session` 未呼び出し、または初期状態） |
| 422 | `frame_idx` または `bbox` のバリデーションエラー（範囲外、サイズ不正、座標不正） |
| 500 | 推論またはエンコード失敗 |
| 503 | モデルロード未完了（5秒タイムアウト）またはロード失敗 |

### フロントエンドの取り扱い

- レスポンスを `Blob`（`type: 'video/mp4'`）として受け取り、`URL.createObjectURL(blob)` で `videoElement.src` に差し替え
- 直前の `videoElement.src`（原動画 ObjectURL or 前回の合成 mp4）は `URL.revokeObjectURL` でメモリ解放
- 差し替え時に `currentTime` と `paused` を保存し、`canplay` 後に復元する（再生位置を維持）
- 1物体のみなので、新しい mp4 を受信したら古いものは破棄（[01-overview.md §1.2](01-overview.md#12-機能要件) F6）

## 4.6 BBox 座標系の規約

- 原点: 動画の左上 `(0, 0)`
- 単位: 動画のピクセル座標（解像度ベース）。キャンバス上の表示座標ではない
- フロントエンドは Pixi 上のマウス座標を、動画の実解像度座標に逆変換してから送信する（[07-pixi-canvas.md](07-pixi-canvas.md) 参照）
- BBox の幅／高さが極端に小さい（例: 5px 未満）場合のバリデーションはバックエンド側ではなくフロントエンドで早期に弾く（UX 配慮）

## 4.7 タイムアウト方針

| 操作 | バックエンド側タイムアウト | フロント側のリトライ |
|---|---|---|
| `/health` | なし（即応答） | フロントからは現状ポーリングしない |
| `/session` | モデル待機 5 秒 → 503 | 自動リトライしない。失敗時はユーザーにエラー表示 |
| `/segment` | モデル待機 5 秒 → 503。推論自体には上限なし | 自動リトライしない |

5秒は**起動直後にモデルロード未完了で来たリクエストが諦めるまでの時間**。SAM2 のロードは通常もっと時間がかかるが、起動して時間が経っているのにロードが終わっていないケースは異常とみなして早めに 503 を返す方針。

## 4.8 サンプル: フロントエンド呼び出し

参考実装。詳細は `frontend/src/renderer/api/client.ts` に集約する。

```ts
// /session
async function uploadVideo(file: File): Promise<SessionResponse> {
  const fd = new FormData();
  fd.append("video", file);
  const res = await fetch(`${API_BASE}/session`, { method: "POST", body: fd });
  if (!res.ok) throw new Error(`session failed: ${res.status}`);
  return res.json();
}

// /segment
async function segment(req: SegmentRequest): Promise<Blob> {
  const res = await fetch(`${API_BASE}/segment`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) throw new Error(`segment failed: ${res.status}`);
  return res.blob();
}
```

## 4.9 実装チェックリスト

- [ ] `/health` がモデルロード状態を返す
- [ ] `/session` が multipart で mp4 を受け、`videoMeta` を返す（`session_id` は返さない）
- [ ] `/segment` が原動画＋マスク半透明合成済み mp4 をバイナリで返す
- [ ] `/segment` を `/session` 未呼び出しで叩くと 409 を返す
- [ ] フロントの `client.ts` が `/session` と `/segment` を型付きで呼ぶ
- [ ] BBox はフロント側で動画ピクセル座標に変換してから送信される
- [ ] エラーレスポンスがフロントエンドで分類処理されている（503は再試行、409はセッション再作成 等）
