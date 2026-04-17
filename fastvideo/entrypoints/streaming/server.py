# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from fastvideo.api.schema import ServeConfig
from fastvideo.logger import init_logger

logger = init_logger(__name__)


def run_server(serve_config: ServeConfig) -> None:
    """Launch the streaming (WebSocket / Dynamo) server."""
    if serve_config.streaming is None:
        raise ValueError("ServeConfig.streaming must be set to launch the streaming server; "
                         "got None. Add a `streaming:` block to your serve config.")
    raise NotImplementedError("streaming server is not implemented yet")
