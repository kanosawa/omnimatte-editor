import { TopBar } from "./components/TopBar/TopBar";
import { CanvasView } from "./components/Canvas/CanvasView";
import { BottomBar } from "./components/BottomBar/BottomBar";

export function App() {
  return (
    <div className="app">
      <TopBar />
      <CanvasView />
      <BottomBar />
    </div>
  );
}
