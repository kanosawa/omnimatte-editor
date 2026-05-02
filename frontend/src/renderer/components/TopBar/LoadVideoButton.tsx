import { useRef, useState } from "react";
import { useVideoStore } from "../../store/videoStore";

export function LoadVideoButton() {
  const inputRef = useRef<HTMLInputElement>(null);
  const loadVideo = useVideoStore((s) => s.loadVideo);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const onChange: React.ChangeEventHandler<HTMLInputElement> = async (e) => {
    const file = e.target.files?.[0];
    e.target.value = "";
    if (!file) return;
    setBusy(true);
    setError(null);
    try {
      await loadVideo(file);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="topbar-group">
      <input
        ref={inputRef}
        type="file"
        accept="video/mp4"
        style={{ display: "none" }}
        onChange={onChange}
      />
      <button className="btn" disabled={busy} onClick={() => inputRef.current?.click()}>
        {busy ? "読み込み中…" : "動画を読み込む"}
      </button>
      {error && <span className="topbar-error">{error}</span>}
    </div>
  );
}
