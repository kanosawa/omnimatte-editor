from fastapi import APIRouter

from server.casper import get_casper_state
from server.detector import detector_holder
from server.full_foreground_store import full_foreground_store
from server.sam_backend import sam_backend
from server.schemas import HealthResponse
from server.session import session_slot


router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        model_state=sam_backend.state,
        casper_state=get_casper_state(),
        detector_state=detector_holder.state,
        full_fg_state=full_foreground_store.state,
        session_active=session_slot.is_active(),
    )
