"""fastvideo2.serve — online serving (REST async jobs + per-step WS). See server.py."""
from fastvideo2.serve.server import build_app, main

__all__ = ["build_app", "main"]
