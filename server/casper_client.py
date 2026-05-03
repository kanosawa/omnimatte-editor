import asyncio
import logging
import os
import tempfile

import httpx
import numpy as np

from server.model import (
    CASPER_DEFAULT_PROMPT,
    CASPER_SIDECAR_BASE,
    CASPER_STARTUP_TIMEOUT_SEC,
)
from server.video_io import write_trimask_mp4


logger = logging.getLogger(__name__)


class CasperUnreachableError(Exception):
    """sidecar に接続できない（spawn 失敗、別マシンが落ちている等）。"""


class CasperBusyError(Exception):
    """sidecar が他のジョブで処理中（409）。"""


class CasperRunError(Exception):
    """sidecar 内で Casper 推論が失敗した（500）。"""


async def get_casper_state() -> str:
    """sidecar の /health から casper_state を返す。

    返り値: "loading" | "ready" | "failed" | "unreachable"
    """
    url = f"{CASPER_SIDECAR_BASE}/health"
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            res = await client.get(url)
            if res.status_code != 200:
                return "unreachable"
            data = res.json()
            return data.get("casper_state", "unreachable")
    except (httpx.HTTPError, asyncio.TimeoutError, OSError):
        return "unreachable"


async def run_casper(
    base_video_path: str,
    trimask: np.ndarray,
    fps: float,
    width: int,
    height: int,
) -> bytes:
    """sidecar の POST /run を呼び、前景削除済み mp4 バイナリを返す。

    `trimask`: (T, H, W) uint8。値は {0, 128, 255} の 3 値で、それぞれ
      remove（対象前景） / neutral（背景） / keep（他の前景）を表す。
    `width / height` は base video の解像度（ピクセル）。sidecar 側で
    16 の倍数に丸めて推論サイズに使う。

    呼び出し側は接続失敗・busy・推論失敗を例外で受け取り、適切な HTTP ステータスに変換する。
    trimask mp4 の一時ファイルは関数内で作成・削除する。
    """
    fd, trimask_path = tempfile.mkstemp(prefix="casper_trimask_", suffix=".mp4")
    os.close(fd)
    try:
        # trimask を base video と同じ解像度・fps でロスレス mp4 化
        write_trimask_mp4(trimask=trimask, fps=fps, out_path=trimask_path)

        url = f"{CASPER_SIDECAR_BASE}/run"
        # 接続タイムアウトのみ設定。読み取りは Casper 推論時間に依存するため上限なし
        timeout = httpx.Timeout(connect=CASPER_STARTUP_TIMEOUT_SEC, read=None, write=60.0, pool=None)
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                with open(base_video_path, "rb") as fv, open(trimask_path, "rb") as ft:
                    files = {
                        "input_video": ("input_video.mp4", fv, "video/mp4"),
                        "trimask": ("trimask_00.mp4", ft, "video/mp4"),
                    }
                    data = {
                        "prompt": CASPER_DEFAULT_PROMPT,
                        "fps": str(fps),
                        "width": str(width),
                        "height": str(height),
                    }
                    res = await client.post(url, files=files, data=data)
        except (httpx.ConnectError, httpx.ConnectTimeout, OSError) as exc:
            raise CasperUnreachableError(f"casper sidecar unreachable: {exc}") from exc
        except httpx.HTTPError as exc:
            raise CasperUnreachableError(f"casper sidecar HTTP error: {exc}") from exc

        if res.status_code == 409:
            raise CasperBusyError(_extract_detail(res, "another removal is in progress"))
        if res.status_code in (503,):
            raise CasperUnreachableError(_extract_detail(res, "casper not ready"))
        if res.status_code != 200:
            raise CasperRunError(_extract_detail(res, f"casper run failed (HTTP {res.status_code})"))
        return res.content
    finally:
        try:
            os.unlink(trimask_path)
        except OSError:
            pass


def _extract_detail(res: httpx.Response, fallback: str) -> str:
    try:
        body = res.json()
        detail = body.get("detail")
        if isinstance(detail, str):
            return detail
    except Exception:
        pass
    text = res.text or ""
    return text[:500] if text else fallback
