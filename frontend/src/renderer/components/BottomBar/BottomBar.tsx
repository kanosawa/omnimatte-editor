import { PlaybackControls } from "./PlaybackControls";
import { Seekbar } from "./Seekbar";
import { TimeDisplay } from "./TimeDisplay";

export function BottomBar() {
  return (
    <div className="bottombar">
      <PlaybackControls />
      <Seekbar />
      <TimeDisplay />
    </div>
  );
}
