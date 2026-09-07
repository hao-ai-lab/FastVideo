"""FastH3 model lifecycle and first-frame continuation for DreamVerse."""

from __future__ import annotations

import gc
import os
import time
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from dreamverse.config import DREAMVERSE_SP_SIZE
from dreamverse.generation_contracts import StepResult

if TYPE_CHECKING:
    from PIL.Image import Image


def _required_config_str(model_config: dict, field_name: str) -> str:
    """Read one required non-empty string from a DreamVerse model profile."""
    value = model_config.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"FastH3 model configuration requires `{field_name}`.")
    return value.strip()


class MiniMaxH3GenerationBackend:
    """Run the VSA data-free FastH3 adapter and retain one continuation frame."""

    def __init__(self, gpu_id: int):
        self.gpu_id = gpu_id
        self.generator: Any | None = None
        self.model_config: dict = {}
        self.continuation_image: Image | None = None

    def _gpu_mem(self) -> str:
        allocated_gib = torch.cuda.memory_allocated() / 1024**3
        reserved_gib = torch.cuda.memory_reserved() / 1024**3
        return f"alloc={allocated_gib:.2f}GiB, reserved={reserved_gib:.2f}GiB"

    @staticmethod
    def _configure_environment(attention_backend: str) -> None:
        """Apply the fixed boot-time switches from the FastH3 reference recipe."""
        os.environ.update({
            "FASTVIDEO_ATTENTION_BACKEND": attention_backend,
            "FASTVIDEO_FA4": "1",
            "FASTVIDEO_MINIMAX_H3_FUSIONS": "all",
            "FASTVIDEO_VSA_SM100A": "0",
        })
        os.environ.pop("FASTVIDEO_INFERENCE_TORCH_COMPILE", None)

    def initialize(self, model_config: dict | None = None) -> None:
        """Download the fixed Preview adapter and load the FastH3 generator.

        The model profile owns the base checkpoint, adapter file, attention
        backend, and generation geometry. The backend translates that profile
        into FastVideo's typed generator configuration.
        """
        if model_config is not None:
            self.model_config = dict(model_config)
        if not self.model_config:
            raise ValueError("FastH3 initialization requires a model configuration.")

        if self.generator is not None:
            self.generator.shutdown()
            self.generator = None
            gc.collect()
            torch.cuda.empty_cache()

        self.clear_conditioning()
        model_path = _required_config_str(self.model_config, "model_path")
        adapter_repo = _required_config_str(self.model_config, "adapter_repo")
        adapter_filename = _required_config_str(self.model_config, "adapter_filename")
        attention_backend = _required_config_str(self.model_config, "attention_backend")
        self._configure_environment(attention_backend)

        from huggingface_hub import hf_hub_download

        from fastvideo import VideoGenerator
        from fastvideo.api import (
            CompileConfig,
            ComponentConfig,
            EngineConfig,
            GeneratorConfig,
            OffloadConfig,
            ParallelismConfig,
            PipelineSelection,
        )

        adapter_path = hf_hub_download(repo_id=adapter_repo, filename=adapter_filename)
        experimental = {
            "attention_backend": attention_backend,
            "inference_torch_compile": attention_backend == "FLASH_ATTN",
            "vae_parallel_decode": True,
            "vae_parallel_decode_strategy": "gather",
        }
        if attention_backend == "VIDEO_SPARSE_ATTN_H3":
            experimental.update({
                "VSA_sparsity": 0.9,
                "VSA_tile_size": 64,
            })
        generator_config = GeneratorConfig(
            model_path=model_path,
            pipeline=PipelineSelection(
                components=ComponentConfig(lora_path=adapter_path, lora_strength=1.0),
                experimental=experimental,
            ),
            engine=EngineConfig(
                num_gpus=DREAMVERSE_SP_SIZE,
                parallelism=ParallelismConfig(tp_size=1, sp_size=DREAMVERSE_SP_SIZE),
                offload=OffloadConfig(
                    dit=False,
                    dit_layerwise=False,
                    text_encoder=True,
                    image_encoder=True,
                    vae=True,
                    pin_cpu_memory=True,
                ),
                compile=CompileConfig(enabled=False, vae_enabled=True),
                use_fsdp_inference=False,
            ),
        )

        print(f"[GPU {self.gpu_id}] Loading FastH3 model: {model_path}")
        print(f"[GPU {self.gpu_id}] FastH3 adapter: {adapter_repo}/{adapter_filename}")
        print(f"[GPU {self.gpu_id}] Before model load: {self._gpu_mem()}")
        self.generator = VideoGenerator.from_config(generator_config)
        print(f"[GPU {self.gpu_id}] FastH3 loaded: {self._gpu_mem()} (warmup pending)")

    def shutdown(self) -> None:
        """Release the FastVideo generator and cached continuation image."""
        self.clear_conditioning()
        if self.generator is not None:
            self.generator.shutdown()
            self.generator = None

    def clear_conditioning(self) -> None:
        """Release the first-frame image retained for the next segment."""
        if self.continuation_image is not None:
            self.continuation_image.close()
            self.continuation_image = None

    @staticmethod
    def _load_rgb_image(image_path: str) -> Image:
        """Load an image into an independent RGB buffer with no open file handle."""
        from PIL import Image

        with Image.open(image_path) as image:
            return image.convert("RGB").copy()

    def _select_conditioning_image(
        self,
        segment_idx: int,
        image_path: str | None,
        reset_conditioning: bool,
    ) -> tuple[Image | None, bool]:
        """Select the initial upload or retained last frame for one segment."""
        if reset_conditioning:
            self.clear_conditioning()
        if segment_idx > 1 and self.continuation_image is not None:
            return self.continuation_image.copy(), True
        if segment_idx > 1 and not reset_conditioning:
            raise RuntimeError(f"FastH3 segment {segment_idx} requires a retained continuation frame.")
        if segment_idx == 1 and image_path:
            return self._load_rgb_image(image_path), False
        return None, False

    def _build_request(self, prompt: str, conditioning_image: Image | None):
        """Build the typed FastVideo request owned by the FastH3 profile."""
        from fastvideo.api import GenerationRequest, InputConfig, OutputConfig, SamplingConfig

        return GenerationRequest(
            prompt=prompt,
            negative_prompt="",
            inputs=InputConfig(pil_image=conditioning_image),
            sampling=SamplingConfig(
                height=int(self.model_config["height"]),
                width=int(self.model_config["width"]),
                num_frames=int(self.model_config["num_frames"]),
                fps=24,
                num_inference_steps=int(self.model_config["num_inference_steps"]),
                guidance_scale=1.0,
                batch_cfg=False,
                seed=int(self.model_config["seed"]),
            ),
            output=OutputConfig(save_video=False, return_frames=True),
        )

    def _save_continuation_frame(self, frames: list) -> None:
        """Retain the last decoded frame as first-frame conditioning."""
        from PIL import Image

        self.clear_conditioning()
        self.continuation_image = Image.fromarray(np.ascontiguousarray(frames[-1])).convert("RGB")

    def generate_step(
        self,
        prompt: str,
        segment_idx: int,
        image_path: str | None,
        reset_conditioning: bool,
    ) -> StepResult:
        """Generate one synchronized FastH3 segment and retain its last frame.

        Later segments use MiniMax H3's first-frame-to-video path. The first
        conditioned frame and its matching audio duration are trimmed before
        streaming so adjacent segments do not duplicate media.
        """
        if self.generator is None:
            raise RuntimeError("FastH3 generator is not initialized.")
        conditioning_image, uses_continuation = self._select_conditioning_image(
            segment_idx,
            image_path,
            reset_conditioning,
        )
        request = self._build_request(prompt, conditioning_image)
        started = time.perf_counter()
        try:
            result = self.generator.generate(request)
        finally:
            if conditioning_image is not None:
                conditioning_image.close()
        torch.cuda.synchronize()
        generation_ms = (time.perf_counter() - started) * 1000.0

        if isinstance(result, list):
            raise RuntimeError("FastH3 returned multiple results for one DreamVerse segment.")
        frames = result.frames
        if not isinstance(frames, list) or not frames:
            raise RuntimeError("FastH3 generation did not return decoded frames.")
        audio = result.audio
        audio_sample_rate = result.audio_sample_rate
        if audio is not None and audio_sample_rate is None:
            raise RuntimeError("FastH3 returned audio without an audio sample rate.")

        save_started = time.perf_counter()
        self._save_continuation_frame(frames)
        save_conditioning_ms = (time.perf_counter() - save_started) * 1000.0
        timings = {
            "generation_ms": generation_ms,
            "generation_time_ms": float(result.generation_time or 0.0) * 1000.0,
            "save_conditioning_ms": save_conditioning_ms,
            "e2e_latency_ms": (time.perf_counter() - started) * 1000.0,
        }
        trim_frames = 1 if uses_continuation else 0
        print(f"[GPU {self.gpu_id}] FastH3 segment {segment_idx}: "
              f"{len(frames)} frames, gen={generation_ms:.0f}ms, "
              f"save_conditioning={save_conditioning_ms:.0f}ms, "
              f"e2e={timings['e2e_latency_ms']:.0f}ms")
        return StepResult(
            frames=frames,
            audio=audio,
            audio_sample_rate=audio_sample_rate,
            timings=timings,
            head_trim_frames=trim_frames,
            head_trim_audio_frames=trim_frames,
        )

    def warmup(self, prompt: str) -> dict[str, float]:
        """Compile the FastH3 text and first-frame paths before readiness."""
        warmup_prompt = (prompt or "").strip()
        if not warmup_prompt:
            raise RuntimeError("Startup warmup prompt must be non-empty.")
        print(f"[GPU {self.gpu_id}] FastH3 startup warmup starting "
              "(synthetic segments: text-to-video, first-frame-to-video)")
        started = time.perf_counter()
        text_result = self.generate_step(
            warmup_prompt,
            segment_idx=1,
            image_path=None,
            reset_conditioning=True,
        )
        first_frame_result = self.generate_step(
            warmup_prompt,
            segment_idx=2,
            image_path=None,
            reset_conditioning=False,
        )
        total_ms = (time.perf_counter() - started) * 1000.0
        self.clear_conditioning()
        text_ms = float(text_result.timings.get("e2e_latency_ms", 0.0))
        first_frame_ms = float(first_frame_result.timings.get("e2e_latency_ms", 0.0))
        print(f"[GPU {self.gpu_id}] FastH3 startup warmup complete: "
              f"text_to_video={text_ms:.0f}ms, "
              f"first_frame_to_video={first_frame_ms:.0f}ms, "
              f"total={total_ms:.0f}ms")
        return {
            "warmup_text_to_video_ms": text_ms,
            "warmup_first_frame_to_video_ms": first_frame_ms,
            "warmup_total_ms": total_ms,
        }

    def apply_lora_stack(self, stack: list[tuple[str, float]]) -> tuple[str | None, str | None]:
        """Reject runtime LoRA mutation because FastH3 uses one startup adapter."""
        del stack
        raise RuntimeError("FastH3 uses its fixed startup adapter and does not support runtime LoRA changes.")
