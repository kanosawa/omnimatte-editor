import asyncio
import logging
import subprocess
import sys
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from server.detector import detector_holder
from server.model import (
    CASPER_PORT,
    SPAWN_CASPER,
)
from server.routes import health, removal, segment, session
from server.sam_backend import sam_backend


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def _pump_sidecar_output(stream, prefix: str) -> None:
    """sidecar の標準出力 / 標準エラーを本サーバログに [casper] プレフィクス付きで流す。"""
    try:
        for raw in iter(stream.readline, b""):
            line = raw.decode(errors="replace").rstrip()
            if line:
                print(f"{prefix}{line}", flush=True)
    except Exception:
        pass
    finally:
        try:
            stream.close()
        except Exception:
            pass


def _spawn_sidecar() -> subprocess.Popen | None:
    """Casper sidecar を別プロセスで起動し、stdout/stderr をログに流す。失敗時は None。"""
    env = {**__import__("os").environ}
    env.setdefault("OMNIMATTE_CASPER_PORT", str(CASPER_PORT))
    try:
        proc = subprocess.Popen(
            [sys.executable, "-m", "casper_server.main"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
        )
    except Exception:
        logger.exception("failed to spawn casper sidecar")
        return None

    threading.Thread(
        target=_pump_sidecar_output,
        args=(proc.stdout, "[casper] "),
        daemon=True,
    ).start()
    logger.info("casper sidecar spawned (pid=%s, port=%s)", proc.pid, CASPER_PORT)
    return proc


@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(sam_backend.load())
    asyncio.create_task(detector_holder.load())
    sidecar_proc = _spawn_sidecar() if SPAWN_CASPER else None
    if not SPAWN_CASPER:
        logger.info("OMNIMATTE_SPAWN_CASPER=0; skip spawning casper sidecar")
    try:
        yield
    finally:
        if sidecar_proc is not None:
            logger.info("terminating casper sidecar (pid=%s)", sidecar_proc.pid)
            sidecar_proc.terminate()
            try:
                sidecar_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning("casper sidecar did not terminate in 10s; killing")
                sidecar_proc.kill()


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
