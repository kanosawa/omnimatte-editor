import { app, BrowserWindow } from "electron";
import { join } from "node:path";

function createWindow(): void {
  const win = new BrowserWindow({
    width: 1280,
    height: 800,
    show: false,
    autoHideMenuBar: true,
    webPreferences: {
      preload: join(__dirname, "../preload/index.js"),
      sandbox: false,
    },
  });

  win.on("ready-to-show", () => win.show());

  if (process.env.ELECTRON_RENDERER_URL) {
    win.loadURL(process.env.ELECTRON_RENDERER_URL);
    // dev mode では DevTools を自動オープン
    win.webContents.openDevTools({ mode: "detach" });
  } else {
    win.loadFile(join(__dirname, "../renderer/index.html"));
  }

  // F12 / Ctrl+Shift+I で DevTools を toggle (本番ビルドでもデバッグできるように)
  win.webContents.on("before-input-event", (_event, input) => {
    if (input.type !== "keyDown") return;
    const isToggle =
      input.key === "F12" ||
      (input.control && input.shift && input.key.toLowerCase() === "i");
    if (isToggle) win.webContents.toggleDevTools();
  });
}

app.whenReady().then(() => {
  createWindow();
  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});
