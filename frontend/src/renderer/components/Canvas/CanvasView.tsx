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
  useEffect(() => {
    canvasRef.current?.setVideo(videoElement);
  }, [videoElement, videoMeta?.width, videoMeta?.height]);

  // BBox 操作可否
  useEffect(() => {
    const interactive = !isPlaying && segmentState !== "running" && videoMeta != null;
    canvasRef.current?.setBboxInteractive(interactive);
  }, [isPlaying, segmentState, videoMeta]);

  // 外部からの BBox 表示同期
  useEffect(() => {
    canvasRef.current?.setBboxDisplay(bbox);
  }, [bbox]);

  return <div ref={containerRef} className="canvas-host" />;
}
