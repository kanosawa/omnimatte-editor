# Frontend (Electron + React + Pixi)

Omnimatte Editor のフロントエンド。仕様詳細は [../docs/spec/05-frontend-structure.md](../docs/spec/05-frontend-structure.md) を参照。

## セットアップ

```bash
cd frontend
npm install
```

## 開発起動

```bash
npm run dev
```

Electron ウィンドウが開き、Vite の HMR が有効。

## 環境変数

`frontend/.env` を作成してバックエンドの URL を指定する:

```
VITE_API_BASE=http://localhost:8000
VITE_HEALTH_POLL_INTERVAL_MS=1000
```

## ビルド

```bash
npm run build
npm run preview
```

## ディレクトリ構成

```
src/
├── main/index.ts            # Electron main プロセス
├── preload/index.ts         # preload（MVP では空）
└── renderer/
    ├── main.tsx             # React エントリ
    ├── App.tsx              # 3段レイアウト
    ├── styles.css
    ├── api/client.ts        # バックエンド呼び出し
    ├── components/
    │   ├── TopBar/
    │   ├── Canvas/          # CanvasView.tsx + VideoCanvas.ts
    │   └── BottomBar/
    ├── store/videoStore.ts  # zustand + VideoElement 同期
    └── types/index.ts
```

## 操作

1. 「動画を読み込む」で mp4 を選択
2. 動画を停止し、キャンバス上でドラッグして BBox を指定
3. 「SAM2 で検出」ボタンを押す（推論完了後にマスクが重畳表示）
4. 推論完了後、新しい BBox を指定して再実行可能
