# pyright: reportMissingTypeArgument=false
from __future__ import annotations

from dreamverse._deps import require_dreamverse_runtime_deps


def cli() -> None:
    require_dreamverse_runtime_deps()

    try:
        from dreamverse.main import cli as main_cli
    except ModuleNotFoundError as exc:
        if exc.name in {"fastvideo", "torch", "safetensors"}:
            raise SystemExit(
                "dreamverse-server requires FastVideo runtime deps. "
                "Install `fastvideo[dreamverse]` or run `uv sync --extra dreamverse` from the FastVideo checkout."
            ) from exc
        raise

    main_cli()


if __name__ == "__main__":
    cli()
