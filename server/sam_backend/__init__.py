"""SAM backend selection layer.

Exposes a single `sam_backend` instance that implements `SamBackend`.
Selected at import time via `OMNIMATTE_SAM_VERSION` env var (`2` or `3`, default `2`).
"""
import logging
import os
from typing import Literal

from server.sam_backend.base import SamBackend


logger = logging.getLogger(__name__)


SamVersion = Literal["2", "3"]


def _build_sam_backend() -> SamBackend:
    version = os.environ.get("OMNIMATTE_SAM_VERSION", "2").strip()
    if version == "2":
        from server.sam_backend.sam2_backend import Sam2Backend
        logger.info("using SAM2 backend (OMNIMATTE_SAM_VERSION=2)")
        return Sam2Backend()
    if version == "3":
        from server.sam_backend.sam3_backend import Sam3Backend
        logger.info("using SAM3 backend (OMNIMATTE_SAM_VERSION=3)")
        return Sam3Backend()
    raise RuntimeError(
        f"unknown OMNIMATTE_SAM_VERSION={version!r} (expected '2' or '3')"
    )


sam_backend: SamBackend = _build_sam_backend()
