import asyncio
import logging
from contextlib import asynccontextmanager

# `_ModuleLock` deadlock 対策
import torch  # noqa: F401
import torchvision  # noqa: F401
import torchvision.ops  # noqa: F401

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.ml.detector import detectron2
from backend.ml.sam import sam2
from backend.ml.casper import casper
from backend.routes import health, removal, segment, session


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(detectron2.load())
    asyncio.create_task(sam2.load())
    asyncio.create_task(casper.load())
    yield


app = FastAPI(title="Omnimatte Editor Backend", lifespan=lifespan)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(session.router)
app.include_router(segment.router)
app.include_router(removal.router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        port=8000,
    )
