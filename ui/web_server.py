# SPDX-License-Identifier: Apache-2.0
"""
FastVideo Job Runner — Web server for static frontend.

Serves the static HTML/CSS/JS files and optionally proxies API requests
to a separate API server.

Usage:
    # Serve static files only (API must be on same origin or CORS-enabled)
    python -m ui.web_server

    # Serve static files with API proxy
    python -m ui.web_server --api-url http://localhost:8189
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("fastvideo.ui.web")

# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="FastVideo Job Runner Web",
    version="0.1.0",
    description="Web server for FastVideo UI frontend",
)

# API proxy URL (set via CLI args)
_api_url: str | None = None


class APIProxyMiddleware(BaseHTTPMiddleware):
    """Proxy /api/* requests to the API server."""

    async def dispatch(self, request: Request, call_next):
        if _api_url and request.url.path.startswith("/api/"):
            import httpx

            # Build the target URL
            target_url = f"{_api_url.rstrip('/')}{request.url.path}"
            if request.url.query:
                target_url += f"?{request.url.query}"

            # Forward the request
            async with httpx.AsyncClient() as client:
                try:
                    # Get request body if present
                    body = await request.body() if request.method in (
                        "POST", "PUT", "PATCH"
                    ) else None

                    # Forward headers (exclude host and connection)
                    headers = dict(request.headers)
                    headers.pop("host", None)
                    headers.pop("connection", None)

                    # Make the proxied request
                    response = await client.request(
                        method=request.method,
                        url=target_url,
                        headers=headers,
                        content=body,
                        follow_redirects=True,
                        timeout=300.0,  # Long timeout for video generation
                    )

                    # Return the response
                    return Response(
                        content=response.content,
                        status_code=response.status_code,
                        headers=dict(response.headers),
                        media_type=response.headers.get("content-type"),
                    )
                except Exception as e:
                    logger.error("API proxy error: %s", e)
                    return Response(
                        content=f'{{"detail": "API proxy error: {str(e)}"}}',
                        status_code=502,
                        media_type="application/json",
                    )

        # For non-API requests, serve static files
        return await call_next(request)


# Add proxy middleware if API URL is configured
# (will be added in main() after parsing args)


# ---- Static frontend -------------------------------------------------------
_static_dir = os.path.join(os.path.dirname(__file__), "static")


@app.get("/")
async def serve_index():
    """Serve index.html with API URL injected."""
    index_path = os.path.join(_static_dir, "index.html")
    if not os.path.isfile(index_path):
        raise HTTPException(status_code=404, detail="index.html not found")

    with open(index_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Inject API URL meta tag if API URL is configured
    if _api_url:
        # Ensure API URL includes /api path
        api_url_with_path = _api_url.rstrip('/')
        if not api_url_with_path.endswith('/api'):
            api_url_with_path = f"{api_url_with_path}/api"
        api_meta = f'<meta name="api-url" content="{api_url_with_path}" />'
        # Insert after the viewport meta tag
        content = content.replace(
            '<meta name="viewport"',
            f'{api_meta}\n  <meta name="viewport"',
        )

    from fastapi.responses import HTMLResponse
    return HTMLResponse(content=content)


# Serve other static files (CSS, JS, etc.) at root paths
@app.get("/style.css")
async def serve_css():
    css_path = os.path.join(_static_dir, "style.css")
    if os.path.isfile(css_path):
        return FileResponse(css_path, media_type="text/css")
    raise HTTPException(status_code=404)


@app.get("/app.js")
async def serve_js():
    js_path = os.path.join(_static_dir, "app.js")
    if os.path.isfile(js_path):
        return FileResponse(js_path, media_type="text/javascript")
    raise HTTPException(status_code=404)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def main() -> None:
    global _api_url  # noqa: PLW0603

    parser = argparse.ArgumentParser(
        description="FastVideo Job Runner web server"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Bind address (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8188,
        help="Port number (default: 8188)",
    )
    parser.add_argument(
        "--api-url",
        default=os.environ.get("FASTVIDEO_API_URL"),
        help=(
            "URL of the API server to proxy requests to "
            "Can also be set via FASTVIDEO_API_URL environment variable. "
            "If not set, API requests must be on the same origin or CORS-enabled."
        ),
    )
    args = parser.parse_args()

    _api_url = args.api_url

    if _api_url:
        # Add proxy middleware
        app.add_middleware(APIProxyMiddleware)
        logger.info("API proxy enabled: %s", _api_url)
    else:
        logger.info(
            "API proxy disabled. "
            "API requests must be on the same origin or CORS-enabled."
        )

    import uvicorn

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()

