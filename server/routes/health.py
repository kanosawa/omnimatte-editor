from fastapi import APIRouter

from server.casper_client import get_casper_state
from server.detector import detector_holder
from server.full_foreground_store import full_foreground_store
from server.model import model_holder
from server.schemas import HealthResponse
from server.session import session_slot


router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    casper_state = await get_casper_state()
    return HealthResponse(
        model_state=model_holder.state,
        casper_state=casper_state,
        detector_state=detector_holder.state,
        full_fg_state=full_foreground_store.state,
        session_active=session_slot.is_active(),
    )
