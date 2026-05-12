from fastapi import APIRouter

from backend.ml.detector import detectron2
from backend.ml.sam import sam2
from backend.ml.casper import get_casper_state
from backend.schemas import HealthResponse
from backend.stores.full_foreground_store import full_foreground_store
from backend.stores.session import session_slot


router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        detector_state=detectron2.state,
        sam_state=sam2.state,
        casper_state=get_casper_state(),
        full_fg_state=full_foreground_store.state,
        session_active=session_slot.is_active(),
    )
