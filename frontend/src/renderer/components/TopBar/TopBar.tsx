import { LoadVideoButton } from "./LoadVideoButton";
import { RemoveForegroundButton } from "./RemoveForegroundButton";
import { Sam2Button } from "./Sam2Button";

export function TopBar() {
  return (
    <div className="topbar">
      <LoadVideoButton />
      <Sam2Button />
      <RemoveForegroundButton />
    </div>
  );
}
