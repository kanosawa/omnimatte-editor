import uvicorn


def main() -> None:
    uvicorn.run(
        "server.main:app",
        port=8000,
    )


if __name__ == "__main__":
    main()
