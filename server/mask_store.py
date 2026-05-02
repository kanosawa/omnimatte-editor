import threading
import time
from dataclasses import dataclass

import numpy as np


@dataclass
class MaskRecord:
    masks: np.ndarray              # (T, H, W) bool
    base_video_path: str           # マスク生成時点の base video パス（整合性チェック用）
    fps: float                     # マスク生成時点の fps（マスクの mp4 化に使う）
    created_at: float


class MaskStore:
    """直近 1 件の SAM2 マスクを保持する単一スロット。

    `/segment` 完了時に `set` で上書きし、`/remove` 成功時 / `/session`
    差し替え時に `clear` する。`/remove` の入力に使うため、サーバ側に
    保持してフロント⇔サーバ間でマスクを往復させない。
    """

    def __init__(self) -> None:
        self._current: MaskRecord | None = None
        self._lock = threading.Lock()

    def set(self, masks: np.ndarray, base_video_path: str, fps: float) -> MaskRecord:
        record = MaskRecord(
            masks=masks,
            base_video_path=base_video_path,
            fps=fps,
            created_at=time.time(),
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


mask_store = MaskStore()
