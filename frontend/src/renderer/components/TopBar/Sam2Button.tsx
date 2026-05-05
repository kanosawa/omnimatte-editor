import { useVideoStore } from "../../store/videoStore";

export function Sam2Button() {
  const videoMeta = useVideoStore((s) => s.videoMeta);
  const isPlaying = useVideoStore((s) => s.isPlaying);
  const bbox = useVideoStore((s) => s.bbox);
  const segmentState = useVideoStore((s) => s.segmentState);
  const segmentError = useVideoStore((s) => s.segmentError);
  const removeState = useVideoStore((s) => s.removeState);
  const runSegment = useVideoStore((s) => s.runSegment);

  const isLoaded = videoMeta !== null;
  const isRunning = segmentState === "running";
  const isRemoving = removeState === "running";

  const enabled = isLoaded && !isPlaying && bbox !== null && !isRunning && !isRemoving;

  const label = isRunning ? "物体検出中…" : "物体検出";

  return (
    <div className="topbar-group">
      <button className="btn btn-primary" disabled={!enabled} onClick={() => void runSegment()}>
        {label}
      </button>
      {segmentState === "error" && segmentError && (
        <span className="topbar-error">{segmentError}</span>
      )}
    </div>
  );
}
