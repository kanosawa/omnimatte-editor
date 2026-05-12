import os
import sys

# `cd backend && python run.py` で起動するための sys.path 調整。
# backend を package として import するためにリポジトリ root を sys.path に追加する。
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import uvicorn


def main() -> None:
    port = int(os.environ.get("OMNIMATTE_PORT", "8000"))
    uvicorn.run(
        "backend.main:app",
        host="127.0.0.1",
        port=port,
    )


if __name__ == "__main__":
    main()
