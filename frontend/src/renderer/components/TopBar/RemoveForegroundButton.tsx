import { useVideoStore } from "../../store/videoStore";

export function RemoveForegroundButton() {
  const videoMeta = useVideoStore((s) => s.videoMeta);
  const isPlaying = useVideoStore((s) => s.isPlaying);
  const hasSegmentation = useVideoStore((s) => s.hasSegmentation);
  const segmentState = useVideoStore((s) => s.segmentState);
  const removeState = useVideoStore((s) => s.removeState);
  const removeError = useVideoStore((s) => s.removeError);
  const runRemoveForeground = useVideoStore((s) => s.runRemoveForeground);

  const isLoaded = videoMeta !== null;
  const isSegmenting = segmentState === "running";
  const isRemoving = removeState === "running";

  const enabled =
    isLoaded && !isPlaying && hasSegmentation && !isSegmenting && !isRemoving;

  const label = isRemoving ? "処理中…" : "前景を削除";

  return (
    <div className="topbar-group">
      <button
        className="btn"
        disabled={!enabled}
        onClick={() => void runRemoveForeground()}
      >
        {label}
      </button>
      {removeState === "error" && removeError && (
        <span className="topbar-error">{removeError}</span>
      )}
    </div>
  );
}
