import { create } from "zustand";
import type { Bbox, RemoveState, SegmentState, VideoMeta } from "../types";
import { removeForeground, segment, uploadVideo } from "../api/client";

type VideoStoreState = {
  videoMeta: VideoMeta | null;
  videoElement: HTMLVideoElement | null;
  isPlaying: boolean;
  currentFrame: number;
  bbox: Bbox | null;
  segmentState: SegmentState;
  segmentError: string | null;
  // 直近の SAM2 抽出が完了済みか（合成 mp4 を表示中）。
  // /segment 完了で true、/remove 完了 / loadVideo で false。
  hasSegmentation: boolean;
  removeState: RemoveState;
  removeError: string | null;
};

type VideoStoreActions = {
  loadVideo: (file: File) => Promise<void>;
  togglePlay: () => void;
  play: () => void;
  pause: () => void;
  stepFrame: (delta: number) => void;
  seekTo: (frameIdx: number) => void;
  setBbox: (bbox: Bbox | null) => void;
  clearBbox: () => void;
  runSegment: () => Promise<void>;
  runRemoveForeground: () => Promise<void>;
  reset: () => void;
};

type VideoStore = VideoStoreState & VideoStoreActions;

function createVideoElement(): HTMLVideoElement {
  const v = document.createElement("video");
  v.crossOrigin = "anonymous";
  v.muted = true;
  v.playsInline = true;
  v.preload = "auto";
  v.loop = true;
  return v;
}

const videoElement = createVideoElement();

// videoElement.src は loadVideo 直後は原動画、SAM2 実行後は backend がオーバーレイ合成した動画。
// 「原動画 vs マスク」の二本立てではなく、常に「画面に出すべき動画」を1本だけ持つ。
let videoObjectUrl: string | null = null;
let videoFrameCallbackId: number | null = null;

const initialState: VideoStoreState = {
  videoMeta: null,
  videoElement,
  isPlaying: false,
  currentFrame: 0,
  bbox: null,
  segmentState: "idle",
  segmentError: null,
  hasSegmentation: false,
  removeState: "idle",
  removeError: null,
};

export const useVideoStore = create<VideoStore>((set, get) => {
  const onPlay = () => set({ isPlaying: true, bbox: null });
  const onPause = () => set({ isPlaying: false });
  const onEnded = () => set({ isPlaying: false });
  const onLoadedMetadata = () => {
    const meta = get().videoMeta;
    if (!meta) return;
    set({
      videoMeta: {
        ...meta,
        width: videoElement.videoWidth || meta.width,
        height: videoElement.videoHeight || meta.height,
      },
    });
  };

  const updateCurrentFrame = () => {
    const meta = get().videoMeta;
    if (!meta || meta.fps <= 0) return;
    const frame = Math.min(meta.numFrames - 1, Math.max(0, Math.round(videoElement.currentTime * meta.fps)));
    if (frame !== get().currentFrame) set({ currentFrame: frame });
  };

  const onTimeUpdate = () => updateCurrentFrame();

  const startVideoFrameCallbackLoop = () => {
    type VFCEnabledVideoElement = HTMLVideoElement & {
      requestVideoFrameCallback?: (cb: () => void) => number;
    };
    const v = videoElement as VFCEnabledVideoElement;
    if (typeof v.requestVideoFrameCallback !== "function") return;
    const tick = () => {
      updateCurrentFrame();
      videoFrameCallbackId = v.requestVideoFrameCallback!(tick);
    };
    videoFrameCallbackId = v.requestVideoFrameCallback(tick);
  };

  videoElement.addEventListener("play", onPlay);
  videoElement.addEventListener("pause", onPause);
  videoElement.addEventListener("ended", onEnded);
  videoElement.addEventListener("loadedmetadata", onLoadedMetadata);
  videoElement.addEventListener("timeupdate", onTimeUpdate);
  startVideoFrameCallbackLoop();

  return {
    ...initialState,

    loadVideo: async (file: File) => {
      // 旧 URL を破棄
      if (videoObjectUrl) {
        URL.revokeObjectURL(videoObjectUrl);
        videoObjectUrl = null;
      }

      set({
        videoMeta: null,
        bbox: null,
        currentFrame: 0,
        isPlaying: false,
        segmentState: "idle",
        segmentError: null,
        hasSegmentation: false,
        removeState: "idle",
        removeError: null,
      });

      // ローカルでまず原動画を再生
      videoObjectUrl = URL.createObjectURL(file);
      videoElement.src = videoObjectUrl;
      videoElement.load();

      // バックエンドへアップロード
      const res = await uploadVideo(file);
      set({ videoMeta: res.videoMeta });
    },

    togglePlay: () => {
      if (videoElement.paused) get().play();
      else get().pause();
    },

    play: () => {
      const meta = get().videoMeta;
      if (!meta) return;
      set({ bbox: null });
      void videoElement.play();
    },

    pause: () => {
      videoElement.pause();
    },

    stepFrame: (delta: number) => {
      const meta = get().videoMeta;
      if (!meta || meta.fps <= 0) return;
      videoElement.pause();
      const next = Math.min(meta.numFrames - 1, Math.max(0, get().currentFrame + delta));
      videoElement.currentTime = next / meta.fps;
      set({ currentFrame: next, bbox: null, isPlaying: false });
    },

    seekTo: (frameIdx: number) => {
      const meta = get().videoMeta;
      if (!meta || meta.fps <= 0) return;
      videoElement.pause();
      const next = Math.min(meta.numFrames - 1, Math.max(0, frameIdx));
      videoElement.currentTime = next / meta.fps;
      set({ currentFrame: next, bbox: null, isPlaying: false });
    },

    setBbox: (bbox) => set({ bbox }),
    clearBbox: () => set({ bbox: null }),

    runSegment: async () => {
      const { videoMeta, bbox, currentFrame, segmentState } = get();
      if (!videoMeta || !bbox || segmentState === "running") return;

      const sentBbox: [number, number, number, number] = [bbox.x1, bbox.y1, bbox.x2, bbox.y2];
      // BBox は表示したまま推論を走らせる（合成 mp4 を表示し始める段階でクリアする）
      set({ segmentState: "running", segmentError: null });

      try {
        const blob = await segment({
          frame_idx: currentFrame,
          bbox: sentBbox,
        });

        // 再生位置を保存して、合成動画に差し替え後に復元する
        const restoreTime = videoElement.currentTime;

        if (videoObjectUrl) URL.revokeObjectURL(videoObjectUrl);
        videoObjectUrl = URL.createObjectURL(blob);
        videoElement.src = videoObjectUrl;
        videoElement.load();

        // canplay を待ってから再生位置を復元し、必ず再生状態にする
        await new Promise<void>((resolve) => {
          const onCanPlay = () => {
            videoElement.removeEventListener("canplay", onCanPlay);
            try { videoElement.currentTime = restoreTime; } catch { /* ignore */ }
            void videoElement.play();
            resolve();
          };
          videoElement.addEventListener("canplay", onCanPlay, { once: true });
          // フォールバック: 5 秒経っても canplay が来なければ resolve
          setTimeout(() => {
            videoElement.removeEventListener("canplay", onCanPlay);
            resolve();
          }, 5000);
        });

        // 合成 mp4 の表示が始まる段階で BBox をクリアする
        set({ segmentState: "idle", hasSegmentation: true, bbox: null });
      } catch (err) {
        // 失敗時は BBox を残してユーザーが再試行できるようにする
        const message = err instanceof Error ? err.message : String(err);
        set({ segmentState: "error", segmentError: message });
      }
    },

    runRemoveForeground: async () => {
      const { videoMeta, hasSegmentation, segmentState, removeState } = get();
      if (!videoMeta || !hasSegmentation) return;
      if (segmentState === "running" || removeState === "running") return;

      set({ removeState: "running", removeError: null, bbox: null });

      try {
        const blob = await removeForeground();

        const restoreTime = videoElement.currentTime;
        const wasPaused = videoElement.paused;

        if (videoObjectUrl) URL.revokeObjectURL(videoObjectUrl);
        videoObjectUrl = URL.createObjectURL(blob);
        videoElement.src = videoObjectUrl;
        videoElement.load();

        await new Promise<void>((resolve) => {
          const onCanPlay = () => {
            videoElement.removeEventListener("canplay", onCanPlay);
            // Casper sidecar は推論解像度を 16 の倍数に丸めるため、
            // 元動画の幅が 16 の倍数でないと出力動画の解像度が変わる
            // （例: 536x288 → 544x288）。新解像度を videoMeta に反映しないと、
            // CanvasView の useEffect が発火せず、古い PIXI VideoSource が
            // 新動画上で生き続けて GL texture サイズ不整合 → 画面が真っ黒、
            // という症状になる。
            const meta = get().videoMeta;
            // [DEBUG black-screen] 一時ログ。原因切り分け後に削除する。
            // eslint-disable-next-line no-console
            console.log("[remove-debug] onCanPlay", {
              oldMetaWidth:  meta?.width,
              oldMetaHeight: meta?.height,
              videoElementWidth:  videoElement.videoWidth,
              videoElementHeight: videoElement.videoHeight,
              readyState: videoElement.readyState,
              paused: videoElement.paused,
              duration: videoElement.duration,
              currentTime: videoElement.currentTime,
              srcUrl: videoElement.src,
            });
            if (meta) {
              const newW = videoElement.videoWidth  || meta.width;
              const newH = videoElement.videoHeight || meta.height;
              if (newW !== meta.width || newH !== meta.height) {
                // eslint-disable-next-line no-console
                console.log("[remove-debug] updating videoMeta", { newW, newH });
                set({ videoMeta: { ...meta, width: newW, height: newH } });
              } else {
                // eslint-disable-next-line no-console
                console.log("[remove-debug] videoMeta unchanged, useEffect won't fire");
              }
            }
            try { videoElement.currentTime = restoreTime; } catch { /* ignore */ }
            if (!wasPaused) void videoElement.play();
            resolve();
          };
          videoElement.addEventListener("canplay", onCanPlay, { once: true });
          setTimeout(() => {
            videoElement.removeEventListener("canplay", onCanPlay);
            resolve();
          }, 5000);
        });

        // 前景削除後はマスクが無効化され、元の base video を更新したのと同じ扱い
        set({ removeState: "idle", hasSegmentation: false });
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        // 失敗時は表示中の合成 mp4 をそのまま維持し、再試行可能にする
        set({ removeState: "error", removeError: message });
      }
    },

    reset: () => {
      videoElement.pause();
      videoElement.removeAttribute("src");
      videoElement.load();
      if (videoObjectUrl) {
        URL.revokeObjectURL(videoObjectUrl);
        videoObjectUrl = null;
      }
      if (videoFrameCallbackId !== null) {
        type VFCEnabledVideoElement = HTMLVideoElement & {
          cancelVideoFrameCallback?: (id: number) => void;
        };
        const v = videoElement as VFCEnabledVideoElement;
        v.cancelVideoFrameCallback?.(videoFrameCallbackId);
        videoFrameCallbackId = null;
      }
      set({ ...initialState, videoElement });
    },
  };
});
