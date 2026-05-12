import asyncio
import logging
from contextlib import asynccontextmanager

# Sam2 / Detectron2 / Casper の各 _load_sync は
# asyncio.to_thread で並行に走り、各 thread が初回 import を走らせる。
# Python 3.12 の thread-safe import 強化により、同じ torchvision サブモジュール
# (e.g. torchvision.ops.roi_align) を複数 thread が同時に import すると
# `_ModuleLock` deadlock が発生する。メイン thread で先に import を完了させて
# おけば、後続 thread はキャッシュ hit で済むので deadlock しない。
import torch  # noqa: F401
import torchvision  # noqa: F401
import torchvision.ops  # noqa: F401

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.ml.casper import casper
from backend.ml.detector import detectron2
from backend.ml.sam import sam2
from backend.routes import health, removal, segment, session


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(sam2.load())
    asyncio.create_task(detectron2.load())
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
