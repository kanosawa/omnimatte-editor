import { useVideoStore } from "../../store/videoStore";

export function PlaybackControls() {
  const videoMeta = useVideoStore((s) => s.videoMeta);
  const isPlaying = useVideoStore((s) => s.isPlaying);
  const currentFrame = useVideoStore((s) => s.currentFrame);
  const segmentState = useVideoStore((s) => s.segmentState);
  const removeState = useVideoStore((s) => s.removeState);
  const togglePlay = useVideoStore((s) => s.togglePlay);
  const stepFrame = useVideoStore((s) => s.stepFrame);

  const isLoaded = videoMeta !== null;
  const numFrames = videoMeta?.numFrames ?? 0;
  const isBusy = segmentState === "running" || removeState === "running";
  const canPlay = isLoaded && !isBusy;
  const canStep = canPlay && !isPlaying;

  return (
    <div className="playback-controls">
      <button
        className="btn btn-icon"
        disabled={!canStep || currentFrame <= 0}
        onClick={() => stepFrame(-1)}
        title="コマ戻し"
      >
        ⏮
      </button>
      <button
        className="btn btn-icon"
        disabled={!canPlay}
        onClick={togglePlay}
        title={isPlaying ? "停止" : "再生"}
      >
        {isPlaying ? "⏸" : "▶"}
      </button>
      <button
        className="btn btn-icon"
        disabled={!canStep || currentFrame >= numFrames - 1}
        onClick={() => stepFrame(1)}
        title="コマ送り"
      >
        ⏭
      </button>
    </div>
  );
}
