# 02. 全体アーキテクチャ

## 2.1 システム構成

```mermaid
flowchart LR
    subgraph FE["フロントエンド (Electron)"]
        UI[React UI]
        Store[zustand Store]
        VE[VideoElement]
        Canvas[Pixi Canvas Class]
        UI <--> Store
        Store <--> VE
        UI -.uses.-> Canvas
        Canvas -.draws.-> VE
    end

    subgraph BE["バックエンド (FastAPI)"]
        API[HTTP Endpoints]
        Loader["Model Loader<br/>(起動時プリロード)"]
        Pred[SAM2 Predictor]
        Sess["Session Slot<br/>(base_video_path / inference_state)"]
        MStore["MaskStore<br/>(直近の per-frame masks)"]
        Enc["Composite Encoder<br/>(base video＋マスク半透明合成)"]
        Casper["Casper Runner<br/>(gen-omnimatte-public, subprocess)"]
        API --> Sess
        API --> Pred
        Loader --> Pred
        Pred --> Enc
        Pred --> MStore
        API --> Casper
        Casper --> Sess
    end

    UI -- "POST /session (mp4 binary)" --> API
    API -- "video_meta" --> UI
    UI -- "POST /segment (frame_idx, bbox)" --> API
    API -- "composite mp4 binary" --> UI
    UI -- "POST /remove (empty body)" --> API
    API -- "foreground-removed mp4 binary" --> UI
```

主な特徴:
- フロントエンドは Electron アプリ。React がUI、Pixi がキャンバス描画、zustand が状態を担当
- バックエンドは FastAPI。起動時に SAM2 モデルをプリロード
- 通信はすべて HTTP。mp4 はリクエスト/レスポンスともにバイナリで送受
- `/segment` のレスポンスはマスク単体ではなく、**サーバ側で base video にマスクを半透明合成済みの mp4**。フロントは `<video>` 1 本で再生するだけになり、原動画とマスク動画の同期問題が発生しない
- `/remove` は SAM2 マスクで指定した前景を base video から削除した mp4 を返し、**サーバ側のセッションのベース動画も新動画に差し替える**（カスケード）
- バックエンドは常に最大 1 件の動画セッションだけを保持する（新規 `/session` で旧セッションは自動破棄）。クライアント・サーバー間でセッション ID をやり取りしない
- 直近の SAM2 マスクは `MaskStore` がサーバ側で保持し、`/remove` の入力に使う（フロント⇔サーバ間でマスクを往復させない）

## 2.2 ディレクトリ構成

リポジトリ全体の構成は以下のとおり。

```
omnimatte-editor/
├── README.md
├── docs/
│   └── spec/                       # 本仕様書
├── server/                         # 本プロジェクトのFastAPIサーバ
│   ├── __init__.py                 # 空
│   ├── main.py                     # FastAPIエントリポイント
│   ├── model.py                    # SAM2モデルのロード状態管理 + 設定値（ハードコード）
│   ├── session.py                  # セッションスロット（常に最大1件）。base video の差し替えも担当
│   ├── mask_store.py               # 直近 SAM2 結果（per-frame バイナリマスク）の単一スロット
│   ├── removal.py                  # gen-omnimatte-public を subprocess で呼ぶ前景削除ランナ
│   ├── video_io.py                 # mp4 読み書き、マスク → mp4 エンコード
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── health.py
│   │   ├── session.py
│   │   ├── segment.py
│   │   └── removal.py              # POST /remove
│   └── schemas.py                  # Pydanticスキーマ
├── vendor/
│   ├── sam2/                       # SAM2 公式リポジトリ（依存ライブラリとして配置）
│   │   └── examples/
│   │       └── segment_video_server.py
│   └── gen-omnimatte-public/       # Casper（Wan2.1-1.3B）リポジトリ（git submodule）
├── run.py                          # 起動スクリプト（uvicorn 起動のみ）
├── requirements.txt                # SAM2 (-e ./vendor/sam2) を含む
└── frontend/
    ├── package.json
    ├── electron.vite.config.ts
    ├── tsconfig.json
    ├── index.html
    ├── src/
    │   ├── main/                   # Electron main プロセス
    │   │   └── index.ts
    │   ├── preload/                # preload スクリプト
    │   │   └── index.ts
    │   └── renderer/               # React アプリ本体
    │       ├── main.tsx            # React エントリ
    │       ├── App.tsx
    │       ├── api/                # バックエンド呼び出し
    │       │   └── client.ts
    │       ├── components/
    │       │   ├── TopBar/
    │       │   │   ├── TopBar.tsx
    │       │   │   ├── LoadVideoButton.tsx
    │       │   │   └── Sam2Button.tsx
    │       │   ├── Canvas/
    │       │   │   ├── CanvasView.tsx           # React ラッパ
    │       │   │   └── VideoCanvas.ts           # Pixi クラス（07-pixi-canvas.md）
    │       │   └── BottomBar/
    │       │       ├── BottomBar.tsx
    │       │       ├── PlaybackControls.tsx     # 再生/停止/コマ送り/コマ戻し
    │       │       ├── Seekbar.tsx              # シークバー
    │       │       └── TimeDisplay.tsx          # 時間 + フレーム番号表示
    │       ├── store/
    │       │   └── videoStore.ts                # VideoElementとzustandの同期（08-state-management.md）
    │       └── types/
    │           └── index.ts
    └── README.md                   # フロントエンドの起動手順
```

`vendor/sam2/` は SAM2 公式リポジトリそのもの（依存ライブラリとして配置）。本プロジェクトのサーバ実装は `server/` に配置し、`vendor/sam2/examples/segment_video_server.py` をコピペするのではなく、参考にしながら本仕様に合わせて再実装する。

`vendor/` 層を挟むのは、`sam2/` リポジトリが内部に同名 `sam2/sam2/` パッケージを持つため、`server/` などと同じ階層に置くと `import sam2` 時に namespace package として shadowing されるのを防ぐため。`vendor/sam2/` は sys.path 上のどのエントリ直下にも来ないので、editable install の finder 経由で正規パッケージが解決される。

## 2.3 ビルド／起動の概要

### バックエンド

```bash
# 環境構築（初回のみ。project root で実行）
pip install -r requirements.txt   # SAM2 (-e ./vendor/sam2) と依存ライブラリを一括インストール

# 起動（推奨・OS 非依存）
python run.py
```

`requirements.txt` の冒頭で `-e ./vendor/sam2` を指定しているため、SAM2 本体は editable インストールされる。`run.py` は単に uvicorn を起動するだけのラッパスクリプト（sys.path 操作は不要）。

uvicorn を直接使う場合は project root から `uvicorn server.main:app` でも可。

### フロントエンド

```bash
# 環境構築（初回のみ）
cd frontend
npm install

# 開発起動
npm run dev
```

`VITE_API_BASE` 環境変数でバックエンドURLを切り替える。

## 2.4 通信プロトコル概要

```mermaid
sequenceDiagram
    participant U as User
    participant FE as Frontend
    participant BE as Backend
    participant M as SAM2 Model

    Note over BE,M: 起動時にモデルプリロード
    BE->>M: load(weights)
    M-->>BE: ready

    U->>FE: mp4 ファイル選択
    FE->>BE: POST /session (mp4 binary)
    BE->>BE: 旧セッション破棄 + 動画一時保存 + init_state
    BE-->>FE: video_meta

    U->>FE: 動画停止 → BBox指定 → SAM2ボタン
    FE->>BE: POST /segment (frame_idx, bbox)
    BE->>M: propagate_in_video(...)
    M-->>BE: mask frames
    BE->>BE: マスクを MaskStore に保存 + base video＋マスクを合成し mp4 エンコード
    BE-->>FE: composite mp4 binary
    FE->>FE: video.src を合成 mp4 に差し替え

    U->>FE: 前景削除ボタン
    FE->>BE: POST /remove (空ボディ)
    BE->>BE: MaskStore から masks 取り出し + Casper を subprocess で起動
    BE->>BE: 出力 mp4 で base video を差し替え + inference_state を再構築 + MaskStore.clear()
    BE-->>FE: foreground-removed mp4 binary
    FE->>FE: video.src を新 base video に差し替え + hasSegmentation=false
```

詳細なリクエスト/レスポンスは [04-api.md](04-api.md) 参照。

## 2.5 主要モジュール責務サマリ

| モジュール | 責務 | 詳細仕様 |
|---|---|---|
| `server/main.py` | FastAPI 起動、ルータ登録、CORS、lifespan で非同期ロード起動 | [03-backend.md](03-backend.md) |
| `server/model.py` | SAM2 モデルのロード状態管理 + 設定値ハードコード | [03-backend.md](03-backend.md) |
| `server/session.py` | 現在の `inference_state` と base video を保持する単一スロット。`swap_base_video` でベース動画差し替え | [03-backend.md](03-backend.md) |
| `server/mask_store.py` | 直近 SAM2 マスクの単一スロット | [03-backend.md](03-backend.md) |
| `server/removal.py` | Casper（gen-omnimatte-public）の subprocess 呼び出し | [03-backend.md](03-backend.md) |
| `server/video_io.py` | mp4 デコード、base video＋マスクの半透明合成 mp4 エンコード、マスク → mp4 書き出し | [03-backend.md](03-backend.md) |
| `frontend/src/renderer/store/videoStore.ts` | zustandストア + VideoElement同期 | [08-state-management.md](08-state-management.md) |
| `frontend/src/renderer/components/Canvas/VideoCanvas.ts` | Pixi 描画ロジック | [07-pixi-canvas.md](07-pixi-canvas.md) |
| `frontend/src/renderer/components/Canvas/CanvasView.tsx` | Pixi クラスを React コンポーネントとしてラップ | [07-pixi-canvas.md](07-pixi-canvas.md) |
| `frontend/src/renderer/components/TopBar/Sam2Button.tsx` | SAM2 実行ボタン。BBoxの有無で活性制御 | [09-state-transitions.md](09-state-transitions.md) |
| `frontend/src/renderer/components/TopBar/RemoveForegroundButton.tsx` | 前景削除ボタン。`hasSegmentation` で活性制御 | [09-state-transitions.md](09-state-transitions.md) |

## 2.6 実装チェックリスト

- [ ] `server/` と `frontend/src/renderer/` のディレクトリが本仕様どおりに配置されている
- [ ] バックエンドは project root から `python run.py` または `uvicorn server.main:app` で起動できる
- [ ] フロントエンドは `npm run dev` で Electron ウィンドウが開く
- [ ] `VITE_API_BASE` でローカル/クラウドが切り替えられる
- [ ] `OMNIMATTE_PORT` 環境変数でバックエンドのリッスンポートが切り替えられる
