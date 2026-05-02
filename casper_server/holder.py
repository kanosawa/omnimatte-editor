import asyncio
import logging
import os
from typing import Any, Literal

from server.model import CASPER_TRANSFORMER_PATH


logger = logging.getLogger(__name__)


CasperState = Literal["loading", "ready", "failed"]


class CasperHolder:
    """Casper パイプラインのロード状態管理。SAM2 の `ModelHolder` と同パターン。"""

    def __init__(self) -> None:
        self._state: CasperState = "loading"
        self._pipeline: Any = None
        self._vae: Any = None
        self._generator: Any = None
        self._cfg: Any = None
        self._error: str | None = None
        self._ready_event = asyncio.Event()

    @property
    def state(self) -> CasperState:
        return self._state

    @property
    def pipeline(self) -> Any:
        return self._pipeline

    @property
    def vae(self) -> Any:
        return self._vae

    @property
    def generator(self) -> Any:
        return self._generator

    @property
    def cfg(self) -> Any:
        return self._cfg

    @property
    def error(self) -> str | None:
        return self._error

    def _load_sync(self):
        # 重みファイルの存在確認を最初に行い、明確なエラーで即 failed にする
        if not os.path.exists(CASPER_TRANSFORMER_PATH):
            raise FileNotFoundError(
                f"casper model not found: {CASPER_TRANSFORMER_PATH}"
            )

        from casper_server.pipeline import build_default_config, load_pipeline

        cfg = build_default_config()
        pipeline, vae, generator = load_pipeline(cfg)
        return cfg, pipeline, vae, generator

    async def load(self) -> None:
        try:
            cfg, pipeline, vae, generator = await asyncio.to_thread(self._load_sync)
            self._cfg = cfg
            self._pipeline = pipeline
            self._vae = vae
            self._generator = generator
            self._state = "ready"
            logger.info("casper pipeline loaded")
        except Exception as exc:
            logger.exception("casper pipeline load failed")
            self._error = str(exc)
            self._state = "failed"
        self._ready_event.set()

    async def wait_ready(self, timeout: float | None = None) -> None:
        if self._state == "ready":
            return
        if self._state == "failed":
            raise RuntimeError(f"casper failed to load: {self._error}")
        try:
            await asyncio.wait_for(self._ready_event.wait(), timeout=timeout)
        except asyncio.TimeoutError as exc:
            raise TimeoutError("casper not ready (timeout)") from exc
        if self._state == "failed":
            raise RuntimeError(f"casper failed to load: {self._error}")


casper_holder = CasperHolder()
