"""Cosmos Predict2.5 distilled bootstrap and DFD continuation for DreamVerse."""

from __future__ import annotations

import gc
import os
import time
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from dreamverse.generation_contracts import StepResult

if TYPE_CHECKING:
    from PIL.Image import Image

_SILENT_AUDIO_SAMPLE_RATE = 24_000


def _required_config_str(model_config: dict, field_name: str) -> str:
    value = model_config.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Cosmos Predict2.5 DFD model configuration requires `{field_name}`.")
    return value.strip()


class Cosmos25DFDGenerationBackend:
    """Own complementary Cosmos T2W and one-frame-conditioned DFD generators."""

    def __init__(self, gpu_id: int):
        self.gpu_id = gpu_id
        self.bootstrap_generator: Any | None = None
        self.continuation_generator: Any | None = None
        self.model_config: dict = {}
        self.continuation_image: Image | None = None

    def _gpu_mem(self) -> str:
        allocated_gib = torch.cuda.memory_allocated() / 1024**3
        reserved_gib = torch.cuda.memory_reserved() / 1024**3
        return f"alloc={allocated_gib:.2f}GiB, reserved={reserved_gib:.2f}GiB"

    @staticmethod
    def _configure_environment(attention_backend: str) -> None:
        os.environ["FASTVIDEO_ATTENTION_BACKEND"] = attention_backend
        os.environ.pop("FASTVIDEO_INFERENCE_TORCH_COMPILE", None)

    @staticmethod
    def _load_generator(model_path: str):
        from fastvideo import VideoGenerator

        return VideoGenerator.from_pretrained(
            model_path,
            num_gpus=1,
            use_fsdp_inference=False,
            dit_cpu_offload=False,
            vae_cpu_offload=False,
            text_encoder_cpu_offload=True,
            pin_cpu_memory=True,
            enable_torch_compile=False,
        )

    def initialize(self, model_config: dict | None = None) -> None:
        """Load both package roles so bootstrap and continuation are ready."""
        if model_config is not None:
            self.model_config = dict(model_config)
        if not self.model_config:
            raise ValueError("Cosmos Predict2.5 DFD initialization requires a model configuration.")

        self.shutdown()
        bootstrap_path = _required_config_str(self.model_config, "model_path")
        continuation_path = _required_config_str(self.model_config, "continuation_model_path")
        attention_backend = _required_config_str(self.model_config, "attention_backend")
        self._configure_environment(attention_backend)

        print(f"[GPU {self.gpu_id}] Loading Cosmos T2W bootstrap: {bootstrap_path}")
        print(f"[GPU {self.gpu_id}] Before bootstrap load: {self._gpu_mem()}")
        self.bootstrap_generator = self._load_generator(bootstrap_path)
        print(f"[GPU {self.gpu_id}] Loading Cosmos DFD continuation: {continuation_path}")
        self.continuation_generator = self._load_generator(continuation_path)
        print(f"[GPU {self.gpu_id}] Cosmos T2W + DFD loaded: {self._gpu_mem()} (warmup pending)")

    def shutdown(self) -> None:
        """Release both FastVideo generators and the retained terminal frame."""
        self.clear_conditioning()
        for attr_name in ("bootstrap_generator", "continuation_generator"):
            generator = getattr(self, attr_name)
            if generator is not None:
                try:
                    generator.shutdown()
                except Exception as exc:
                    print(f"[GPU {self.gpu_id}] Cosmos generator shutdown warning: {exc}")
                setattr(self, attr_name, None)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def clear_conditioning(self) -> None:
        if self.continuation_image is not None:
            self.continuation_image.close()
            self.continuation_image = None

    @staticmethod
    def _load_rgb_image(image_path: str) -> Image:
        from PIL import Image

        with Image.open(image_path) as image:
            return image.convert("RGB").copy()

    def _select_conditioning_image(
        self,
        segment_idx: int,
        image_path: str | None,
        reset_conditioning: bool,
    ) -> tuple[Image | None, bool]:
        if reset_conditioning:
            self.clear_conditioning()
        if segment_idx > 1 and self.continuation_image is not None:
            return self.continuation_image.copy(), True
        if segment_idx > 1 and not reset_conditioning:
            raise RuntimeError(f"Cosmos DFD segment {segment_idx} requires a retained continuation frame.")
        if segment_idx == 1 and image_path:
            return self._load_rgb_image(image_path), False
        return None, False

    def _sampling_param(self, *, conditioned: bool):
        # ``num_cond_frames`` is not yet exposed by the typed SamplingConfig,
        # so this backend uses the compatibility request until that field lands.
        from fastvideo.api.sampling_param import SamplingParam

        num_frames_key = "continuation_num_frames" if conditioned else "bootstrap_num_frames"
        return SamplingParam(
            negative_prompt="",
            save_video=False,
            return_frames=True,
            height=int(self.model_config["height"]),
            width=int(self.model_config["width"]),
            num_frames=int(self.model_config[num_frames_key]),
            fps=int(self.model_config["fps"]),
            num_inference_steps=int(self.model_config["num_inference_steps"]),
            guidance_scale=1.0,
            seed=int(self.model_config["seed"]),
            num_cond_frames=1 if conditioned else 0,
        )

    def _save_continuation_frame(self, frame: object) -> None:
        from PIL import Image

        self.clear_conditioning()
        if isinstance(frame, Image.Image):
            self.continuation_image = frame.convert("RGB").copy()
            return
        pixels = np.asarray(frame)
        self.continuation_image = Image.fromarray(np.ascontiguousarray(pixels)).convert("RGB")

    @staticmethod
    def _silent_audio(frame_count: int, fps: int) -> torch.Tensor:
        sample_count = max(1, int(round((frame_count / float(fps)) * _SILENT_AUDIO_SAMPLE_RATE)))
        return torch.zeros(sample_count, dtype=torch.float32)

    def generate_step(
        self,
        prompt: str,
        segment_idx: int,
        image_path: str | None,
        reset_conditioning: bool,
    ) -> StepResult:
        """Generate a T2W start or DFD continuation and retain its last frame."""
        if self.bootstrap_generator is None or self.continuation_generator is None:
            raise RuntimeError("Cosmos T2W + DFD generators are not initialized.")

        conditioning_image, uses_continuation = self._select_conditioning_image(
            segment_idx,
            image_path,
            reset_conditioning,
        )
        conditioned = conditioning_image is not None
        generator = self.continuation_generator if conditioned else self.bootstrap_generator
        sampling_param = self._sampling_param(conditioned=conditioned)
        started = time.perf_counter()
        try:
            if conditioned:
                sampling_param.pil_image = conditioning_image
            result = generator.generate_video(prompt, sampling_param=sampling_param)
        finally:
            if conditioning_image is not None:
                conditioning_image.close()
        torch.cuda.synchronize()
        generation_ms = (time.perf_counter() - started) * 1000.0

        if not isinstance(result, dict):
            raise RuntimeError("Cosmos generation did not return one result dictionary.")
        frames = result.get("frames")
        expected_frames = int(sampling_param.num_frames)
        if not isinstance(frames, list) or len(frames) != expected_frames:
            actual_frames = len(frames) if isinstance(frames, list) else None
            raise RuntimeError(f"Cosmos generation returned {actual_frames} frames; expected {expected_frames}.")

        save_started = time.perf_counter()
        self._save_continuation_frame(frames[-1])
        save_conditioning_ms = (time.perf_counter() - save_started) * 1000.0
        fps = int(sampling_param.fps)
        timings = {
            "generation_ms": generation_ms,
            "generation_time_ms": float(result.get("generation_time") or 0.0) * 1000.0,
            "save_conditioning_ms": save_conditioning_ms,
            "e2e_latency_ms": (time.perf_counter() - started) * 1000.0,
        }
        trim_frames = 1 if uses_continuation else 0
        mode = "DFD continuation" if conditioned else "T2W bootstrap"
        print(f"[GPU {self.gpu_id}] Cosmos {mode} segment {segment_idx}: "
              f"{len(frames)} frames, gen={generation_ms:.0f}ms, "
              f"save_conditioning={save_conditioning_ms:.0f}ms, "
              f"e2e={timings['e2e_latency_ms']:.0f}ms")
        return StepResult(
            frames=frames,
            audio=self._silent_audio(len(frames), fps),
            audio_sample_rate=_SILENT_AUDIO_SAMPLE_RATE,
            timings=timings,
            head_trim_frames=trim_frames,
            head_trim_audio_frames=trim_frames,
        )

    def warmup(self, prompt: str) -> dict[str, float]:
        """Exercise both T2W bootstrap and retained-frame DFD request shapes."""
        warmup_prompt = (prompt or "").strip()
        if not warmup_prompt:
            raise RuntimeError("Startup warmup prompt must be non-empty.")
        print(f"[GPU {self.gpu_id}] Cosmos startup warmup starting "
              "(synthetic segments: T2W bootstrap, DFD continuation)")
        started = time.perf_counter()
        bootstrap_result = self.generate_step(warmup_prompt, 1, None, True)
        continuation_result = self.generate_step(warmup_prompt, 2, None, False)
        total_ms = (time.perf_counter() - started) * 1000.0
        self.clear_conditioning()
        bootstrap_ms = float(bootstrap_result.timings.get("e2e_latency_ms", 0.0))
        continuation_ms = float(continuation_result.timings.get("e2e_latency_ms", 0.0))
        print(f"[GPU {self.gpu_id}] Cosmos startup warmup complete: "
              f"bootstrap={bootstrap_ms:.0f}ms, continuation={continuation_ms:.0f}ms, total={total_ms:.0f}ms")
        return {
            "warmup_bootstrap_ms": bootstrap_ms,
            "warmup_continuation_ms": continuation_ms,
            "warmup_total_ms": total_ms,
        }

    def apply_lora_stack(self, stack: list[tuple[str, float]]) -> tuple[str | None, str | None]:
        del stack
        raise RuntimeError("Cosmos Predict2.5 DFD does not support DreamVerse runtime LoRA changes.")
