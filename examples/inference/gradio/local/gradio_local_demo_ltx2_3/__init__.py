"""FastLTX-2.3 Gradio local demo package.

Split from the monolithic gradio_local_demo_ltx2_3.py for maintainability.
Runtime entrypoint is `main` in .app.
"""

from .app import main

__all__ = ["main"]
