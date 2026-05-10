from __future__ import annotations

DREAMVERSE_RUNTIME_DEPS_MESSAGE = (
    "Dreamverse runtime deps missing — install with pip install 'fastvideo[dreamverse]'.")


def require_dreamverse_runtime_deps() -> None:
    try:
        import cerebras.cloud.sdk  # noqa: F401
        import openai  # noqa: F401
    except ModuleNotFoundError as exc:
        missing_root = (exc.name or "").split(".", 1)[0]
        if missing_root in {"cerebras", "openai"}:
            raise SystemExit(DREAMVERSE_RUNTIME_DEPS_MESSAGE) from exc
        raise
