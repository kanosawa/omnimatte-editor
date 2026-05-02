from fastapi import APIRouter

from casper_server.holder import casper_holder


router = APIRouter()


@router.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "casper_state": casper_holder.state,
    }
