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
    def __init__(self) -> None:
        self._current: MaskRecord | None = None

    def set(self, trimask: np.ndarray, base_video_path: str, fps: float) -> MaskRecord:
        record = MaskRecord(
            trimask=trimask,
            base_video_path=base_video_path,
            fps=fps,
        )
        self._current = record
        return record

    def current(self) -> MaskRecord | None:
        return self._current
