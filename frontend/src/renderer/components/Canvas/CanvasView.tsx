import { useEffect, useRef } from "react";
import { useVideoStore } from "../../store/videoStore";
import { VideoCanvas } from "./VideoCanvas";

export function CanvasView() {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<VideoCanvas | null>(null);

  const setBbox       = useVideoStore((s) => s.setBbox);
  const videoElement  = useVideoStore((s) => s.videoElement);
  const isPlaying     = useVideoStore((s) => s.isPlaying);
  const segmentState  = useVideoStore((s) => s.segmentState);
  const removeState   = useVideoStore((s) => s.removeState);
  const bbox          = useVideoStore((s) => s.bbox);
  const videoMeta     = useVideoStore((s) => s.videoMeta);

  // mount / unmount
  useEffect(() => {
    if (!containerRef.current) return;
    const canvas = new VideoCanvas({
      container: containerRef.current,
      onBboxChange: (b) => setBbox(b),
    });
    canvasRef.current = canvas;
    return () => {
      canvas.destroy();
      canvasRef.current = null;
    };
  }, [setBbox]);

  // 動画の差し替え (loadVideo / runSegment 後の合成動画)
  // dims は bbox 座標系をバックエンドの cv2 寸法に揃えるために必須
  // （HTMLVideoElement.videoWidth は SAR 補正後の DAR 寸法で食い違うことがある）
  useEffect(() => {
    const dims =
      videoMeta && videoMeta.width > 0 && videoMeta.height > 0
        ? { width: videoMeta.width, height: videoMeta.height }
        : null;
    // [DEBUG black-screen] 一時ログ。原因切り分け後に削除する。
    // eslint-disable-next-line no-console
    console.log("[canvas-debug] useEffect fired -> setVideo", {
      hasVideoElement: !!videoElement,
      videoElementWidth:  videoElement?.videoWidth,
      videoElementHeight: videoElement?.videoHeight,
      dims,
    });
    canvasRef.current?.setVideo(videoElement, dims);
  }, [videoElement, videoMeta?.width, videoMeta?.height]);

  // BBox 操作可否
  useEffect(() => {
    const interactive =
      !isPlaying &&
      segmentState !== "running" &&
      removeState !== "running" &&
      videoMeta != null;
    canvasRef.current?.setBboxInteractive(interactive);
  }, [isPlaying, segmentState, removeState, videoMeta]);

  // 外部からの BBox 表示同期
  useEffect(() => {
    canvasRef.current?.setBboxDisplay(bbox);
  }, [bbox]);

  return <div ref={containerRef} className="canvas-host" />;
}
