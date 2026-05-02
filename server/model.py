import asyncio
import logging
import os
from typing import Literal


logger = logging.getLogger(__name__)

_here = os.path.dirname(os.path.abspath(__file__))
_sam2_dir = os.path.join(os.path.dirname(_here), "vendor", "sam2")
SAM2_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
SAM2_CKPT = os.path.join(_sam2_dir, "checkpoints/sam2.1_hiera_large.pt")
SAM2_DEVICE = "cuda"


ModelState = Literal["loading", "ready", "failed"]


class ModelHolder:
    def __init__(self) -> None:
        self._state: ModelState = "loading"
        self._predictor = None
        self._error: str | None = None
        self._ready_event = asyncio.Event()

    @property
    def state(self) -> ModelState:
        return self._state

    @property
    def predictor(self):
        return self._predictor

    @property
    def error(self) -> str | None:
        return self._error

    def _load_sync(self):
        from sam2.build_sam import build_sam2_video_predictor

        logger.info(
            "loading SAM2 model: cfg=%s ckpt=%s device=%s",
            SAM2_CFG,
            SAM2_CKPT,
            SAM2_DEVICE,
        )
        predictor = build_sam2_video_predictor(
            SAM2_CFG,
            SAM2_CKPT,
            device=SAM2_DEVICE,
        )
        logger.info("SAM2 model loaded")
        return predictor

    async def load(self) -> None:
        try:
            self._predictor = await asyncio.to_thread(self._load_sync)
            self._state = "ready"
        except Exception as exc:
            logger.exception("SAM2 model load failed")
            self._error = str(exc)
            self._state = "failed"
        self._ready_event.set()

    async def wait_ready(self, timeout: float | None = None) -> None:
        if self._state == "ready":
            return
        if self._state == "failed":
            raise RuntimeError(f"model failed to load: {self._error}")
        try:
            await asyncio.wait_for(self._ready_event.wait(), timeout=timeout)
        except asyncio.TimeoutError as exc:
            raise TimeoutError("model not ready (timeout)") from exc
        if self._state == "failed":
            raise RuntimeError(f"model failed to load: {self._error}")


model_holder = ModelHolder()
