from fastapi import APIRouter

from server.model import model_holder
from server.schemas import HealthResponse
from server.session import session_slot


router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        model_state=model_holder.state,
        session_active=session_slot.is_active(),
    )
