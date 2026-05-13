import threading
from dataclasses import dataclass

import numpy as np


@dataclass
class MaskRecord:
    """Casper に渡す trimask とその生成時メタ情報。

    `trimask`: (T, H, W) uint8。値は {0, 128, 255} の 3 値:
      - 0   = remove（対象前景）
      - 128 = neutral（背景）
      - 255 = keep（他の前景）
    """
    trimask: np.ndarray
    base_video_path: str           # マスク生成時点の base video パス（整合性チェック用）
    fps: float                     # マスク生成時点の fps


class MaskStore:
    """per-session スロット。`Session` が 1 個ずつ所有する。

    `/segment` 完了時に `set` で trimask を入れ、`/remove` で読み出す。
    新しい `/session` / `/remove` 時には旧 Session ごと GC されるので、
    明示的な clear は不要（メソッドは API として残してある）。
    """

    def __init__(self) -> None:
        self._current: MaskRecord | None = None
        self._lock = threading.Lock()

    def set(self, trimask: np.ndarray, base_video_path: str, fps: float) -> MaskRecord:
        record = MaskRecord(
            trimask=trimask,
            base_video_path=base_video_path,
            fps=fps,
        )
        with self._lock:
            self._current = record
        return record

    def current(self) -> MaskRecord | None:
        with self._lock:
            return self._current

    def clear(self) -> None:
        with self._lock:
            self._current = None
