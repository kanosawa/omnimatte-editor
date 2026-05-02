import os

import uvicorn


def main() -> None:
    port = int(os.environ.get("OMNIMATTE_CASPER_PORT", "8001"))
    uvicorn.run(
        "casper_server.main:app",
        host="127.0.0.1",
        port=port,
    )


if __name__ == "__main__":
    main()
