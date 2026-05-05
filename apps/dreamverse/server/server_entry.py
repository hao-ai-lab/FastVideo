# pyright: reportMissingTypeArgument=false
from __future__ import annotations


def cli() -> None:
    try:
        from main import cli as main_cli
    except ModuleNotFoundError as exc:
        if exc.name in {"fastvideo", "torch", "safetensors"}:
            raise SystemExit("dreamverse-server requires the Python backend extra. "
                             "Run `uv sync --extra server` first.") from exc
        raise

    main_cli()
