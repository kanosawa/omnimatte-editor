import asyncio
import logging
from contextlib import asynccontextmanager

# `_ModuleLock` deadlock 対策
import torch
import torchvision  # noqa: F401
import torchvision.ops  # noqa: F401

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is required but not available. "
        "This backend supports CUDA-only execution."
    )

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.predictors.detectron2 import detectron2
from backend.predictors.sam2 import sam2
from backend.predictors.casper import casper
from backend.routes import removal, segment, session


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

app.include_router(session.router)
app.include_router(segment.router)
app.include_router(removal.router)


if __name__ == "__main__":
    import os
    import uvicorn

    port = int(os.environ.get("OMNIMATTE_PORT", "8000"))
    uvicorn.run(app, host="127.0.0.1", port=port)
