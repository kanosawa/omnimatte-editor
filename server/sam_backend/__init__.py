"""SAM backend.

Exposes a single `sam_backend` instance that wraps the SAM2 video predictor.
"""
from server.sam_backend.base import SamBackend
from server.sam_backend.sam2_backend import Sam2Backend


sam_backend: SamBackend = Sam2Backend()
