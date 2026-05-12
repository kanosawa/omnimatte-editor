from fastapi import APIRouter

from backend.casper import get_casper_state
from backend.detector import detectron2
from backend.full_foreground_store import full_foreground_store
from backend.sam import sam2
from backend.schemas import HealthResponse
from backend.session import session_slot


router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        model_state=sam2.state,
        casper_state=get_casper_state(),
        detector_state=detectron2.state,
        full_fg_state=full_foreground_store.state,
        session_active=session_slot.is_active(),
    )
