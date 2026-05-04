import * as PIXI from "pixi.js";
import type { Bbox } from "../../types";

type Options = {
  container: HTMLElement;
  onBboxChange: (bbox: Bbox | null) => void;
};

type Point = { x: number; y: number };

export class VideoCanvas {
  private app: PIXI.Application | null = null;
  private container: HTMLElement;
  private onBboxChange: (bbox: Bbox | null) => void;

  // Containers
  private videoLayer = new PIXI.Container();
  private bboxLayer  = new PIXI.Container();

  // Sprites
  private videoSprite: PIXI.Sprite | null = null;
  private videoSource: PIXI.VideoSource | null = null;
  private bboxGfx = new PIXI.Graphics();

  // Interaction
  private bboxInteractive: boolean = false;
  private dragStart: Point | null = null;
  private dragCurrent: Point | null = null;
  private currentBboxDisplay: Bbox | null = null;

  // Layout
  private videoWidth  = 0;
  private videoHeight = 0;
  private resizeObserver: ResizeObserver | null = null;

  private destroyed  = false;
  private initialized = false;

  constructor(options: Options) {
    this.container   = options.container;
    this.onBboxChange = options.onBboxChange;
    void this.init();
  }

  private async init(): Promise<void> {
    const app = new PIXI.Application();
    await app.init({
      background: 0x000000,
      resizeTo: this.container,
      antialias: true,
      autoDensity: true,
      resolution: window.devicePixelRatio || 1,
    });
    if (this.destroyed) {
      app.destroy(true, { children: true, texture: true });
      return;
    }
    this.app = app;
    this.container.appendChild(app.canvas);

    app.stage.addChild(this.videoLayer);
    app.stage.addChild(this.bboxLayer);
    this.bboxLayer.addChild(this.bboxGfx);

    app.stage.eventMode = "static";
    app.stage.hitArea   = app.screen;
    app.stage.on("pointerdown",     this.handlePointerDown);
    app.stage.on("pointermove",     this.handlePointerMove);
    app.stage.on("pointerup",       this.handlePointerUp);
    app.stage.on("pointerupoutside", this.handlePointerUp);

    this.resizeObserver = new ResizeObserver(() => this.layout());
    this.resizeObserver.observe(this.container);
    this.initialized = true;
    this.layout();
  }

  // -------------- Public API --------------

  setVideo(
    video: HTMLVideoElement | null,
    dims?: { width: number; height: number } | null,
  ): void {
    if (!this.initialized) return;

    // 古い sprite を破棄する前に source の resize リスナーを外しておく
    // （sprite.destroy({ texture: true }) で source も連鎖破棄されるため）。
    if (this.videoSource) {
      this.videoSource.off("resize", this.handleSourceResize);
      this.videoSource = null;
    }
    if (this.videoSprite) {
      this.videoLayer.removeChild(this.videoSprite);
      this.videoSprite.destroy({ texture: true });
      this.videoSprite = null;
    }

    if (video) {
      // VideoSource を明示的に構築して autoPlay: false を実際に効かせる。
      const source = new PIXI.VideoSource({ resource: video, autoPlay: false });
      this.videoSource = source;
      // 動画のフレームデータが届いたタイミング（PIXI 内部で source.resize が
      // 走り、texture.orig.width が placeholder の 1 から実寸に切り替わる）で
      // 再 layout を行う。これがないと sprite.scale が 1×1 ベースで計算された
      // ままになり、本物のフレーム到着と同時にスプライトが何百倍にも巨大化して
      // 動画が見えなくなる race condition が発生する。
      source.on("resize", this.handleSourceResize);

      const tex = new PIXI.Texture({ source });
      this.videoSprite = new PIXI.Sprite(tex);
      this.videoLayer.addChildAt(this.videoSprite, 0);
      // dims（バックエンドが cv2 で返した寸法）を最優先で使う。
      // HTMLVideoElement.videoWidth は SAR を考慮した DAR 寸法を返すため、
      // SAR ≠ 1:1 のアナモルフィック動画ではバックエンドの寸法と食い違う。
      // 食い違ったままだと bbox 座標系がずれて SAM2 が誤った位置を切る。
      const applySize = () => {
        const w = dims?.width  ?? video.videoWidth;
        const h = dims?.height ?? video.videoHeight;
        this.videoWidth  = w || this.videoWidth;
        this.videoHeight = h || this.videoHeight;
        this.layout();
      };
      if (dims?.width && dims?.height) applySize();
      else if (video.readyState >= 1) applySize();
      else video.addEventListener("loadedmetadata", applySize, { once: true });
    } else {
      this.videoWidth  = 0;
      this.videoHeight = 0;
    }
    this.layout();
  }

  private handleSourceResize = (): void => {
    this.layout();
  };

  setBboxInteractive(enabled: boolean): void {
    this.bboxInteractive = enabled;
    if (!enabled) {
      this.dragStart   = null;
      this.dragCurrent = null;
      this.redrawBbox();
    }
    if (this.app) this.app.stage.cursor = enabled ? "crosshair" : "default";
  }

  setBboxDisplay(bbox: Bbox | null): void {
    this.currentBboxDisplay = bbox;
    this.redrawBbox();
  }

  clearBbox(): void {
    this.currentBboxDisplay = null;
    this.dragStart   = null;
    this.dragCurrent = null;
    this.redrawBbox();
  }

  resize(): void { this.layout(); }

  destroy(): void {
    this.destroyed = true;
    if (this.videoSource) {
      this.videoSource.off("resize", this.handleSourceResize);
      this.videoSource = null;
    }
    if (this.videoSprite) {
      this.videoLayer.removeChild(this.videoSprite);
      this.videoSprite.destroy({ texture: true });
      this.videoSprite = null;
    }
    this.resizeObserver?.disconnect();
    this.resizeObserver = null;
    if (this.app) {
      this.app.stage.off("pointerdown",      this.handlePointerDown);
      this.app.stage.off("pointermove",      this.handlePointerMove);
      this.app.stage.off("pointerup",        this.handlePointerUp);
      this.app.stage.off("pointerupoutside", this.handlePointerUp);
      try { this.app.destroy(true, { children: true, texture: true }); } catch { /* ignore */ }
      this.app = null;
    }
  }

  // -------------- Layout --------------

  private layout(): void {
    if (!this.app) return;
    const cw = this.container.clientWidth;
    const ch = this.container.clientHeight;
    this.app.renderer.resize(cw, ch);

    const vw = this.videoWidth;
    const vh = this.videoHeight;
    if (vw <= 0 || vh <= 0) {
      this.videoLayer.position.set(0, 0);
      this.videoLayer.scale.set(1);
      this.bboxLayer.position.set(0, 0);
      this.bboxLayer.scale.set(1);
      return;
    }

    const scale   = Math.min(cw / vw, ch / vh);
    const offsetX = (cw - vw * scale) / 2;
    const offsetY = (ch - vh * scale) / 2;

    this.videoLayer.position.set(offsetX, offsetY);
    this.videoLayer.scale.set(scale);
    this.bboxLayer.position.set(offsetX, offsetY);
    this.bboxLayer.scale.set(scale);

    if (this.videoSprite) {
      this.videoSprite.width  = vw;
      this.videoSprite.height = vh;
    }
    this.redrawBbox();
  }

  // -------------- Coordinate helpers --------------

  private stageToVideo(p: Point): Point {
    const s = this.videoLayer.scale.x || 1;
    return {
      x: (p.x - this.videoLayer.position.x) / s,
      y: (p.y - this.videoLayer.position.y) / s,
    };
  }

  private clampToVideo(p: Point): Point {
    return {
      x: Math.min(this.videoWidth,  Math.max(0, p.x)),
      y: Math.min(this.videoHeight, Math.max(0, p.y)),
    };
  }

  private isInsideVideo(p: Point): boolean {
    const v = this.stageToVideo(p);
    return v.x >= 0 && v.y >= 0 && v.x <= this.videoWidth && v.y <= this.videoHeight;
  }

  // -------------- BBox drawing --------------

  private redrawBbox(): void {
    const g = this.bboxGfx;
    g.clear();

    let bbox: Bbox | null = null;
    if (this.dragStart && this.dragCurrent) {
      bbox = {
        x1: Math.min(this.dragStart.x, this.dragCurrent.x),
        y1: Math.min(this.dragStart.y, this.dragCurrent.y),
        x2: Math.max(this.dragStart.x, this.dragCurrent.x),
        y2: Math.max(this.dragStart.y, this.dragCurrent.y),
      };
    } else if (this.currentBboxDisplay) {
      bbox = this.currentBboxDisplay;
    }

    if (!bbox || this.videoWidth <= 0) return;
    const w = bbox.x2 - bbox.x1;
    const h = bbox.y2 - bbox.y1;
    if (w <= 0 || h <= 0) return;

    g.rect(bbox.x1, bbox.y1, w, h);
    g.stroke({ width: 2 / (this.videoLayer.scale.x || 1), color: 0x00e0ff, alpha: 1 });
    g.fill({ color: 0x00e0ff, alpha: 0.1 });
  }

  // -------------- Pointer handlers --------------

  private handlePointerDown = (e: PIXI.FederatedPointerEvent): void => {
    if (!this.bboxInteractive || this.videoWidth <= 0 || this.videoHeight <= 0) return;
    const sp = { x: e.global.x, y: e.global.y };
    if (!this.isInsideVideo(sp)) return;
    const v = this.clampToVideo(this.stageToVideo(sp));
    this.dragStart          = v;
    this.dragCurrent        = v;
    this.currentBboxDisplay = null;
    this.redrawBbox();
  };

  private handlePointerMove = (e: PIXI.FederatedPointerEvent): void => {
    if (!this.bboxInteractive || !this.dragStart) return;
    this.dragCurrent = this.clampToVideo(this.stageToVideo({ x: e.global.x, y: e.global.y }));
    this.redrawBbox();
  };

  private handlePointerUp = (_e: PIXI.FederatedPointerEvent): void => {
    if (!this.bboxInteractive || !this.dragStart || !this.dragCurrent) {
      this.dragStart = this.dragCurrent = null;
      return;
    }
    const x1 = Math.min(this.dragStart.x, this.dragCurrent.x);
    const y1 = Math.min(this.dragStart.y, this.dragCurrent.y);
    const x2 = Math.max(this.dragStart.x, this.dragCurrent.x);
    const y2 = Math.max(this.dragStart.y, this.dragCurrent.y);
    this.dragStart = this.dragCurrent = null;

    if (x2 - x1 < 5 || y2 - y1 < 5) {
      this.currentBboxDisplay = null;
      this.redrawBbox();
      this.onBboxChange(null);
      return;
    }

    const bbox: Bbox = { x1, y1, x2, y2 };
    this.currentBboxDisplay = bbox;
    this.redrawBbox();
    this.onBboxChange(bbox);
  };
}
