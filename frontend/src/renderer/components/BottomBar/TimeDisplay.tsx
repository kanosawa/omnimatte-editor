import { useVideoStore } from "../../store/videoStore";

function formatTime(sec: number): string {
  if (!isFinite(sec) || sec < 0) return "--:--.--";
  const m = Math.floor(sec / 60);
  const s = sec - m * 60;
  return `${String(m).padStart(2, "0")}:${s.toFixed(2).padStart(5, "0")}`;
}

export function TimeDisplay() {
  const videoMeta = useVideoStore((s) => s.videoMeta);
  const currentFrame = useVideoStore((s) => s.currentFrame);

  if (!videoMeta) {
    return (
      <div className="time-display">
        <span className="time">--:--.-- / --:--.--</span>
        <span className="separator">|</span>
        <span className="frame">- / -</span>
      </div>
    );
  }

  const fps = videoMeta.fps;
  const currentSec = fps > 0 ? currentFrame / fps : 0;
  const totalSec = videoMeta.durationSec;

  return (
    <div className="time-display">
      <span className="time">
        {formatTime(currentSec)} / {formatTime(totalSec)}
      </span>
      <span className="separator">|</span>
      <span className="frame">
        {currentFrame} / {videoMeta.numFrames - 1}
      </span>
    </div>
  );
}
