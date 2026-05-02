import { useVideoStore } from "../../store/videoStore";

export function Seekbar() {
  const videoMeta = useVideoStore((s) => s.videoMeta);
  const currentFrame = useVideoStore((s) => s.currentFrame);
  const seekTo = useVideoStore((s) => s.seekTo);

  const isLoaded = videoMeta !== null;
  const max = (videoMeta?.numFrames ?? 1) - 1;

  return (
    <div className="seekbar">
      <input
        type="range"
        min={0}
        max={Math.max(0, max)}
        step={1}
        value={isLoaded ? currentFrame : 0}
        disabled={!isLoaded}
        onChange={(e) => seekTo(Number(e.target.value))}
      />
    </div>
  );
}
