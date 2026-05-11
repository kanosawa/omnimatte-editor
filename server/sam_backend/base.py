"""SAM backend abstraction.

`server.routes.*` および `server.full_foreground_store` などの呼び出し側は
このインターフェースのみを介して SAM を扱う。SAM2 のロード処理・autocast などは
`sam2_backend.py` に閉じ込める。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterator, Literal

import numpy as np


ModelState = Literal["loading", "ready", "failed"]


PropagateItem = tuple[int, list[int], list[np.ndarray]]
"""propagate() の yield 単位。

- frame_idx: フレーム番号
- obj_ids: そのフレームで結果が得られた object_id のリスト
- masks: 各 obj_id に対応する bool[H, W] マスク（base video 解像度）
"""


class SamBackend(ABC):
    """SAM2 video predictor を扱うための backend ABC."""

    # ---------- ロード状態 ----------

    @property
    @abstractmethod
    def state(self) -> ModelState: ...

    @property
    @abstractmethod
    def error(self) -> str | None: ...

    @abstractmethod
    async def load(self) -> None:
        """非同期にモデルをロードする。lifespan から create_task で呼ばれる。"""

    @abstractmethod
    async def wait_ready(self, timeout: float | None = None) -> None:
        """ロード完了を待ち合わせる。timeout 超過は TimeoutError、失敗時は RuntimeError。"""

    # ---------- セッション ----------

    @abstractmethod
    def init_state(self, video_path: str) -> Any:
        """動画ごとの inference_state を構築する。Session に保存される opaque 値。"""

    @abstractmethod
    def reset_state(self, state: Any) -> None:
        """inference_state をクリアし、新しいプロンプトを受け付け可能な状態にする。"""

    # ---------- マスク登録 ----------

    @abstractmethod
    def add_mask(
        self,
        state: Any,
        frame_idx: int,
        obj_id: int,
        mask: np.ndarray,
    ) -> None:
        """事前計算済みの bool マスクを登録する（R-CNN 由来など）。"""

    @abstractmethod
    def add_bbox_prompt(
        self,
        state: Any,
        frame_idx: int,
        obj_id: int,
        bbox: list[float],
        base_video_path: str,
        height: int,
        width: int,
    ) -> np.ndarray:
        """bbox プロンプトを登録し、初期フレームの bool[H, W] マスクを返す。

        - bbox: `[x1, y1, x2, y2]` の動画ピクセル座標
        - base_video_path: 一部 backend が frame を読むのに使う（現状の SAM2 では未使用）
        - height/width: 出力マスク解像度（base video の解像度）

        video predictor の `add_new_points_or_box` に bbox を直接渡す。
        """

    # ---------- 伝播 ----------

    @abstractmethod
    def propagate(
        self,
        state: Any,
        start_frame_idx: int,
        num_frames: int,
        reverse: bool = False,
    ) -> Iterator[PropagateItem]:
        """`start_frame_idx` から順方向 / 逆方向に伝播し、各フレームのマスクを yield する。"""
