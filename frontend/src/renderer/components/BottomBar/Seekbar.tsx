import { useVideoStore } from "../../store/videoStore";

export function Seekbar() {
  const videoMeta = useVideoStore((s) => s.videoMeta);
  const currentFrame = useVideoStore((s) => s.currentFrame);
  const segmentState = useVideoStore((s) => s.segmentState);
  const removeState = useVideoStore((s) => s.removeState);
  const seekTo = useVideoStore((s) => s.seekTo);

  const isLoaded = videoMeta !== null;
  const max = (videoMeta?.numFrames ?? 1) - 1;
  const isBusy = segmentState === "running" || removeState === "running";
  const enabled = isLoaded && !isBusy;

  return (
    <div className="seekbar">
      <input
        type="range"
        min={0}
        max={Math.max(0, max)}
        step={1}
        value={isLoaded ? currentFrame : 0}
        disabled={!enabled}
        onChange={(e) => seekTo(Number(e.target.value))}
      />
    </div>
  );
}
