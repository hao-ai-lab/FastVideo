"""Launch the standalone WanTrack control server."""

import os

import uvicorn


def main() -> None:
    uvicorn.run(
        "apps.wantrack_control.server:app",
        host=os.getenv("WANTRACK_HOST", "127.0.0.1"),
        port=int(os.getenv("WANTRACK_PORT", "8010")),
        reload=False,
    )


if __name__ == "__main__":
    main()
