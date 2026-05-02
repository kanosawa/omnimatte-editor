import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from server.model import model_holder
from server.routes import health, removal, segment, session


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(model_holder.load())
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
        "server.main:app",
        port=8000,
    )
