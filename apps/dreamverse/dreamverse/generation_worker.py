"""Select and own one model-specific generation backend per GPU process."""

from __future__ import annotations

from dreamverse.config import MODEL_CONFIG
from dreamverse.generation_contracts import GenerationBackend, StepResult


def _create_generation_backend(backend_name: str, gpu_id: int) -> GenerationBackend:
    """Construct the backend that owns the selected model family's behavior."""
    if backend_name == "ltx2":
        from dreamverse.video_generation import LTX2GenerationBackend

        return LTX2GenerationBackend(gpu_id)
    if backend_name == "minimax_h3":
        from dreamverse.minimax_h3_generation import MiniMaxH3GenerationBackend

        return MiniMaxH3GenerationBackend(gpu_id)
    raise ValueError(f"Unsupported DreamVerse generation backend: {backend_name!r}")


class VideoGenerationWorker:
    """Delegate GPU lifecycle and generation calls to the active model backend."""

    def __init__(self, gpu_id: int):
        self.gpu_id = gpu_id
        self.model_config: dict = dict(MODEL_CONFIG)
        self.backend_name: str | None = None
        self.backend: GenerationBackend | None = None

    def initialize(self, model_config: dict | None = None) -> None:
        """Load the requested model through its generation backend.

        Model selection belongs here so the GPU process and streaming layers
        use one stable media contract without importing model-specific code.
        """
        requested_model_config = dict(model_config) if model_config is not None else dict(self.model_config)
        backend_name = requested_model_config.get("generation_backend")
        if not isinstance(backend_name, str) or not backend_name:
            raise ValueError("DreamVerse model configuration requires `generation_backend`.")

        candidate_backend = self.backend
        if candidate_backend is None or self.backend_name != backend_name:
            if candidate_backend is not None:
                candidate_backend.shutdown()
            candidate_backend = _create_generation_backend(backend_name, self.gpu_id)

        try:
            candidate_backend.initialize(requested_model_config)
        except Exception:
            try:
                candidate_backend.shutdown()
            except Exception as shutdown_error:
                print(f"[GPU {self.gpu_id}] Backend cleanup after initialization failure: {shutdown_error}")
            self.backend = None
            self.backend_name = None
            raise

        self.model_config = requested_model_config
        self.backend = candidate_backend
        self.backend_name = backend_name

    def _require_backend(self) -> GenerationBackend:
        """Return the initialized backend or fail before processing a command."""
        if self.backend is None:
            raise RuntimeError("Generation backend is not initialized.")
        return self.backend

    def shutdown(self) -> None:
        """Release model resources owned by the selected backend."""
        if self.backend is not None:
            self.backend.shutdown()

    def clear_conditioning(self) -> None:
        self._require_backend().clear_conditioning()

    def generate_step(
        self,
        prompt: str,
        segment_idx: int,
        image_path: str | None,
        reset_conditioning: bool,
    ) -> StepResult:
        """Generate one segment through the selected model backend."""
        return self._require_backend().generate_step(
            prompt,
            segment_idx,
            image_path,
            reset_conditioning,
        )

    def warmup(self, prompt: str) -> dict[str, float]:
        return self._require_backend().warmup(prompt)

    def apply_lora_stack(self, stack: list[tuple[str, float]]) -> tuple[str | None, str | None]:
        return self._require_backend().apply_lora_stack(stack)
