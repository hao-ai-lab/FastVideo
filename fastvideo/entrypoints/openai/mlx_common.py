# SPDX-License-Identifier: Apache-2.0
"""Server configuration and worker lifecycle shared by native MLX servers."""

from concurrent.futures import ThreadPoolExecutor
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from fastvideo.api.schema import GenerationRequest


class MLXServerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    output_dir: str = "outputs/mlx"
    served_model_name: str = Field(default="mlx", min_length=1)


class MLXWorkerGenerator:
    """Load, generate, and release on the same single MLX worker thread."""
    thread_name = "mlx"

    def __init__(self, config: Any) -> None:
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix=self.thread_name)
        try:
            self._pipeline = self._worker.submit(self._load, config).result()
        except BaseException:
            self._worker.shutdown(wait=True)
            raise

    @staticmethod
    def _load(config: Any) -> Any:
        raise NotImplementedError

    def _generate(self, request: GenerationRequest) -> dict[str, Any]:
        raise NotImplementedError

    def generate(self, request: GenerationRequest) -> dict[str, Any]:
        return self._worker.submit(self._generate, request).result()

    @staticmethod
    def _cleanup() -> None:
        from fastvideo.mlx_runtime.memory import cleanup_mlx

        cleanup_mlx()

    def shutdown(self) -> None:

        def release():
            self._pipeline = None
            self._cleanup()

        try:
            self._worker.submit(release).result()
        finally:
            self._worker.shutdown(wait=True)


def run_mlx_server(config: Any) -> None:
    """Dispatch a validated MLX config without constructing a CUDA engine."""
    import uvicorn

    from fastvideo.entrypoints.openai.mlx_wan_server import MLXWanServeConfig, create_mlx_wan_app

    if isinstance(config, MLXWanServeConfig):
        app = create_mlx_wan_app(config)
    else:
        from fastvideo.entrypoints.openai.mlx_server import create_mlx_app

        app = create_mlx_app(config)
    uvicorn.run(app, host=config.server.host, port=config.server.port)
