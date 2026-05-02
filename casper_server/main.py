import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from casper_server.holder import casper_holder
from casper_server.routes import health, run


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 起動と同時に Casper をプリロード
    asyncio.create_task(casper_holder.load())
    yield


app = FastAPI(title="Omnimatte Casper Sidecar", lifespan=lifespan)
app.include_router(health.router)
app.include_router(run.router)


def main() -> None:
    import uvicorn

    port = int(os.environ.get("OMNIMATTE_CASPER_PORT", "8765"))
    uvicorn.run(
        "casper_server.main:app",
        host="127.0.0.1",
        port=port,
    )


if __name__ == "__main__":
    main()
