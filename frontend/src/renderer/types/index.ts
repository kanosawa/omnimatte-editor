export type SegmentState = "idle" | "running" | "error";

export type Bbox = {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
};

export type VideoMeta = {
  width: number;
  height: number;
  fps: number;
  numFrames: number;
  durationSec: number;
};

export type SessionResponse = {
  videoMeta: VideoMeta;
};

export type SegmentRequest = {
  frame_idx: number;
  bbox: [number, number, number, number];
};
