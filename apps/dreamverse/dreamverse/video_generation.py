"""LTX2 model lifecycle and continuation conditioning.

Runs inside a GPU worker subprocess.  Owns the model, the audio
encoder, and the per-session continuation state carried across
segments.  Callers must set ``os.environ["CUDA_VISIBLE_DEVICES"]``
before constructing ``VideoGenerationWorker`` — all ``fastvideo.*``
imports are deferred to method bodies so nothing touches CUDA at
module import time.
"""
# pyright: reportArgumentType=false, reportMissingImports=false, reportMissingTypeArgument=false, reportOptionalMemberAccess=false
# ruff: noqa: SIM105
# mypy: ignore-errors
import gc
import os
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from dreamverse.config import (
    FRAME_HEIGHT,
    FRAME_WIDTH,
    MODEL_CONFIG,
    NUM_FRAMES,
    NUM_INFERENCE_STEPS,
)

# Multi-frame decoded continuation defaults from
# examples/inference/basic/basic_ltx2_distilled_video_continuation.py.
# Overridable via environment variables.
LTX2_VIDEO_CONDITIONING_NUM_FRAMES = int(os.getenv("LTX2_VIDEO_CONDITIONING_NUM_FRAMES", "9"))
LTX2_VIDEO_CONDITIONING_END_OFFSET = int(os.getenv("LTX2_VIDEO_CONDITIONING_END_OFFSET", "0"))
LTX2_VIDEO_CONDITIONING_FRAME_IDX = int(os.getenv("LTX2_VIDEO_CONDITIONING_FRAME_IDX", "0"))
LTX2_VIDEO_CONDITIONING_STRENGTH = float(os.getenv("LTX2_VIDEO_CONDITIONING_STRENGTH", "1.0"))

if (LTX2_VIDEO_CONDITIONING_NUM_FRAMES - 1) % 8 != 0:
    raise ValueError("LTX2_VIDEO_CONDITIONING_NUM_FRAMES must satisfy "
                     "(frames - 1) % 8 == 0; got "
                     f"{LTX2_VIDEO_CONDITIONING_NUM_FRAMES}")

# Audio conditioning: reuse denoised audio latents from the previous
# segment as initial latents for the next segment.
ENABLE_AUDIO_COND = os.getenv("ENABLE_AUDIO_COND", "1").lower() in ("1", "true", "yes")
# Number of decoded video frames worth of audio to condition on.
# Not subject to the (n-1)%8==0 constraint since this is audio-only.
AUDIO_CONDITIONING_NUM_FRAMES = int(os.getenv("AUDIO_CONDITIONING_NUM_FRAMES", '49'))
AUDIO_CONDITIONING_STRENGTH = float(os.getenv("AUDIO_CONDITIONING_STRENGTH", "1.0"))

# Noise injected into conditioning context to prevent error
# accumulation across segments.  0 = no noise (default).
VIDEO_CONTEXT_NOISE = float(os.getenv("VIDEO_CONTEXT_NOISE", "0"))
AUDIO_CONTEXT_NOISE = float(os.getenv("AUDIO_CONTEXT_NOISE", "0"))

# Re-encode audio latents through decode→encode round-trip to
# regularize noise accumulation (mirrors the video PIL→VAE path).
ENABLE_AUDIO_RE_ENCODE = os.getenv("ENABLE_AUDIO_RE_ENCODE", "").lower() in ("1", "true", "yes")

DEFAULT_LTX2_AUDIO_SAMPLE_RATE = 16000
DEFAULT_LTX2_AUDIO_HOP_LENGTH = 160
DEFAULT_LTX2_AUDIO_DOWNSAMPLE = 4


@dataclass
class StepResult:
    """Output of one generation step.

    ``head_trim_frames`` / ``head_trim_audio_frames`` are derived here
    so downstream AV streaming never needs to import conditioning
    constants.
    """
    frames: list
    audio: Any
    audio_sample_rate: int | None
    timings: dict
    head_trim_frames: int
    head_trim_audio_frames: int


class ContinuationState:
    """Per-session video + audio conditioning carried across segments."""

    def __init__(self):
        self.video_images: list | None = None
        self.audio_latents: torch.Tensor | None = None

    def clear(self) -> None:
        if self.video_images:
            for old_image in self.video_images:
                try:
                    old_image.close()
                except Exception:
                    pass
        self.video_images = None
        self.audio_latents = None

    def apply_video(self, request_kwargs: dict, segment_idx: int) -> None:
        """Seed next-segment kwargs with the cached tail frames."""
        if segment_idx <= 1 or not self.video_images:
            return
        from PIL import Image

        cond_images = list(self.video_images)
        if VIDEO_CONTEXT_NOISE > 0:
            noisy = []
            for img in cond_images:
                arr = np.array(img, dtype=np.float32)
                arr += np.random.normal(0, VIDEO_CONTEXT_NOISE * 255, arr.shape)
                arr = np.clip(arr, 0, 255).astype(np.uint8)
                noisy.append(Image.fromarray(arr))
            cond_images = noisy
        request_kwargs["ltx2_video_conditions"] = [(
            cond_images,
            LTX2_VIDEO_CONDITIONING_FRAME_IDX,
            LTX2_VIDEO_CONDITIONING_STRENGTH,
        )]
        request_kwargs["ltx2_images"] = None
        request_kwargs["image_path"] = None

    def apply_audio(
        self,
        request_kwargs: dict,
        segment_idx: int,
        audio_lps: float,
    ) -> None:
        """Seed next-segment kwargs with clean audio latents + denoise mask.

        When audio conditioning is longer than video, extend audio
        generation and shift video RoPE forward so the audio prefix
        sits before video t=0.  ``audio_lps`` (audio latent frames per
        second) is passed in so this class never imports fastvideo.
        """
        if not (ENABLE_AUDIO_COND and segment_idx > 1 and self.audio_latents is not None):
            return
        cached = self.audio_latents  # [B,C,T,mel]
        cond_duration = float(AUDIO_CONDITIONING_NUM_FRAMES) / 24.0
        audio_cond_T = max(1, round(cond_duration * audio_lps))
        audio_cond_T = min(audio_cond_T, cached.shape[2])

        audio_extra = max(0, AUDIO_CONDITIONING_NUM_FRAMES - LTX2_VIDEO_CONDITIONING_NUM_FRAMES)
        if audio_extra > 0:
            audio_num_frames = NUM_FRAMES + audio_extra
            request_kwargs["audio_num_frames"] = (audio_num_frames)
            prefix_sec = float(audio_extra) / 24.0
            request_kwargs["video_position_offset_sec"] = prefix_sec

        new_duration = float(NUM_FRAMES + audio_extra) / 24.0
        total_T = max(
            round(new_duration * audio_lps),
            audio_cond_T + 1,
        )

        B, C, _, mel = cached.shape
        clean = torch.zeros((B, C, total_T, mel), dtype=cached.dtype)
        clean[:, :, :audio_cond_T, :] = (cached[:, :, -audio_cond_T:, :])
        if AUDIO_CONTEXT_NOISE > 0:
            noise = torch.randn_like(clean[:, :, :audio_cond_T, :])
            clean[:, :, :audio_cond_T, :] += (AUDIO_CONTEXT_NOISE * noise)

        mask = torch.ones((B, 1, total_T, 1), dtype=torch.float32)
        mask[:, :, :audio_cond_T, :] = (1.0 - AUDIO_CONDITIONING_STRENGTH)

        request_kwargs["ltx2_audio_clean_latent"] = clean
        request_kwargs["ltx2_audio_denoise_mask"] = mask

    def save_video(self, frames: list) -> None:
        """Snapshot trailing N frames as PIL images for next-segment conditioning."""
        from PIL import Image
        num_cond_frames = LTX2_VIDEO_CONDITIONING_NUM_FRAMES
        end_offset = LTX2_VIDEO_CONDITIONING_END_OFFSET
        if end_offset + num_cond_frames > len(frames):
            raise RuntimeError(f"Cannot extract {num_cond_frames} conditioning frames with "
                               f"end_offset={end_offset} from {len(frames)} generated frames.")
        start_idx = len(frames) - end_offset - num_cond_frames
        self.video_images = [
            Image.fromarray(np.ascontiguousarray(frames[start_idx + i])) for i in range(num_cond_frames)
        ]

    def save_audio_latents(self, latents: torch.Tensor | None) -> None:
        if latents is None:
            self.audio_latents = None
            return
        self.audio_latents = latents.detach().clone().cpu()


class VideoGenerationWorker:
    """Single-GPU LTX2 generator with continuation state.

    Caller must set ``os.environ["CUDA_VISIBLE_DEVICES"]`` before
    instantiating, and call ``initialize()`` before any
    ``generate_step()`` / ``warmup()``.
    """

    def __init__(self, gpu_id: int):
        self.gpu_id = gpu_id
        self.generator = None
        self.current_model_config: dict = dict(MODEL_CONFIG)
        self.continuation = ContinuationState()
        self.audio_encoder_module = None
        self.audio_processor_module = None

    def _gpu_mem(self) -> str:
        a = torch.cuda.memory_allocated() / 1024**3
        r = torch.cuda.memory_reserved() / 1024**3
        return f"alloc={a:.2f}GiB, reserved={r:.2f}GiB"

    @staticmethod
    def _resolve_refine_upsampler_path(model_root: str) -> str:
        candidates = (
            os.path.join(model_root, "spatial_upscaler"),
            os.path.join(model_root, "spatial_upsampler"),
        )
        for candidate in candidates:
            config_path = os.path.join(candidate, "config.json")
            if os.path.isfile(config_path):
                return candidate
        raise FileNotFoundError("Could not find an LTX2 refine upsampler directory under "
                                f"{model_root}. Checked: {', '.join(candidates)}")

    def initialize(self, model_config: dict | None = None) -> None:
        """Load (or reload) the LTX2 generator on the visible GPU."""
        if model_config is not None:
            self.current_model_config = model_config

        if self.generator is not None:
            print(f"[GPU {self.gpu_id}] Freeing old model...")
            try:
                self.generator.shutdown()
            except Exception:
                pass
            del self.generator
            self.generator = None
            gc.collect()
            torch.cuda.empty_cache()
            print(f"[GPU {self.gpu_id}] After cleanup: {self._gpu_mem()}")

        print(f"[GPU {self.gpu_id}] Loading model: "
              f"{self.current_model_config['model_path']}")
        print(f"[GPU {self.gpu_id}] Before model load: {self._gpu_mem()}")

        from fastvideo.api.schema import (
            CompileConfig,
            ComponentConfig,
            EngineConfig,
            GeneratorConfig,
            OffloadConfig,
            PipelineSelection,
            QuantizationConfig,
        )
        from fastvideo.entrypoints.video_generator import VideoGenerator
        from fastvideo.utils import maybe_download_model

        model_root = maybe_download_model(self.current_model_config["model_path"])
        refine_upsampler_path = self._resolve_refine_upsampler_path(model_root)
        config_model_path = (self.current_model_config.get("config_model_path")
                             or self.current_model_config["model_path"])

        enable_compile = os.getenv("ENABLE_TORCH_COMPILE", "1") == "1"

        components = ComponentConfig(
            config_root=config_model_path,
            upsampler_weights=refine_upsampler_path,
        )
        init_weights = self.current_model_config.get("init_weights_from_safetensors")
        if init_weights:
            components.transformer_weights = init_weights

        generator_config = GeneratorConfig(
            model_path=model_root,
            engine=EngineConfig(
                num_gpus=1,
                offload=OffloadConfig(
                    dit=False,
                    dit_layerwise=False,
                    text_encoder=False,
                    vae=False,
                    pin_cpu_memory=True,
                ),
                compile=CompileConfig(
                    enabled=enable_compile,
                    text_encoder_enabled=enable_compile,
                    backend="inductor",
                    fullgraph=True,
                    mode="max-autotune-no-cudagraphs",
                    dynamic=False,
                ),
                use_fsdp_inference=False,
                quantization=QuantizationConfig(transformer_quant="NVFP4"),
            ),
            pipeline=PipelineSelection(
                components=components,
                vae_tiling=False,
                preset_overrides={
                    "refine": {
                        "enabled": True,
                        "num_inference_steps": 2,
                        "guidance_scale": 1.0,
                        "add_noise": True,
                    },
                },
            ),
        )

        self.generator = VideoGenerator.from_pretrained(config=generator_config)
        print(f"[GPU {self.gpu_id}] After model load: {self._gpu_mem()}")
        self._load_audio_encoder(model_root)
        print(f"[GPU {self.gpu_id}] LTX2 model loaded (warmup pending)")

    def _load_audio_encoder(self, model_root: str) -> None:
        if not ENABLE_AUDIO_RE_ENCODE:
            return
        from fastvideo.models.audio.ltx2_audio_processing import AudioProcessor
        from fastvideo.models.loader.component_loader import ComponentLoader

        audio_vae_path = os.path.join(model_root, "audio_vae")
        if not os.path.isdir(audio_vae_path):
            print(f"[GPU {self.gpu_id}] audio_vae dir not found at "
                  f"{audio_vae_path}; disabling re-encode")
            return

        loader = ComponentLoader.for_module_type("audio_encoder", "diffusers")
        enc = loader.load(audio_vae_path, self.generator.fastvideo_args)
        target = getattr(enc, "model", enc)

        proc = AudioProcessor(
            sample_rate=target.sample_rate,
            mel_bins=target.mel_bins,
            mel_hop_length=target.mel_hop_length,
            n_fft=target.n_fft,
        ).to(torch.device("cuda"))

        self.audio_encoder_module = target
        self.audio_processor_module = proc
        print(f"[GPU {self.gpu_id}] Audio encoder loaded for "
              f"re-encode conditioning ({self._gpu_mem()})")

    def _re_encode_audio(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
    ) -> torch.Tensor | None:
        """Waveform → mel → encoder → latents."""
        if (self.audio_encoder_module is None or self.audio_processor_module is None):
            return None
        device = torch.device("cuda")
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)  # [channels, samples]
        waveform = waveform.unsqueeze(0).to(device=device, dtype=torch.float32)  # [1, ch, samples]
        with torch.no_grad():
            mel = self.audio_processor_module.waveform_to_mel(
                waveform,
                waveform_sample_rate=sample_rate,
            ).to(device=device, dtype=torch.float32)
            latents = self.audio_encoder_module(mel)
        return latents.detach()

    def shutdown(self) -> None:
        if self.generator is not None:
            try:
                self.generator.shutdown()
            except Exception:
                pass

    def clear_conditioning(self) -> None:
        self.continuation.clear()

    def generate_step(
        self,
        prompt: str,
        segment_idx: int,
        image_path: str | None,
        reset_conditioning: bool,
    ) -> StepResult:
        """Execute one generation step; snapshot state for the next segment."""
        timings: dict = {}

        request_kwargs = dict(
            prompt=prompt,
            negative_prompt="",
            save_video=False,
            height=FRAME_HEIGHT,
            width=FRAME_WIDTH,
            num_frames=NUM_FRAMES,
            fps=24,
            num_inference_steps=NUM_INFERENCE_STEPS,
            guidance_scale=1.0,
            seed=10,
            ltx2_image_crf=0.0,
            image_path=image_path if segment_idx == 1 else None,
            return_continuation_state=False,
        )

        if reset_conditioning:
            self.continuation.clear()

        audio_lps = (DEFAULT_LTX2_AUDIO_SAMPLE_RATE / DEFAULT_LTX2_AUDIO_HOP_LENGTH / DEFAULT_LTX2_AUDIO_DOWNSAMPLE)

        # Phase 1: seed kwargs with prior-segment conditioning.
        self.continuation.apply_video(request_kwargs, segment_idx)
        self.continuation.apply_audio(request_kwargs, segment_idx, audio_lps)

        # Phase 2: generate.
        t0 = time.perf_counter()
        result = self.generator.generate_video(**request_kwargs)
        torch.cuda.synchronize()
        timings["generation_ms"] = (time.perf_counter() - t0) * 1000

        if not isinstance(result, dict):
            raise RuntimeError("Expected dictionary output from generate_video.")
        frames = result.get("frames")
        if not isinstance(frames, list) or len(frames) == 0:
            raise RuntimeError("Generation did not return frames.")
        audio = result.get("audio")
        audio_sample_rate = result.get("audio_sample_rate")
        if audio is not None and audio_sample_rate is None:
            # LTX2 audio decoding stage uses 24kHz output by default.
            audio_sample_rate = 24000
            print(f"[GPU {self.gpu_id}] audio_sample_rate missing from result; "
                  f"defaulting to {audio_sample_rate}Hz")

        timings["generation_time_ms"] = result.get("generation_time", 0.0) * 1000

        # Phase 3: snapshot continuation state for the next segment.
        t_save_start = time.perf_counter()
        self.continuation.clear()
        self.continuation.save_video(frames)
        next_audio_latents = self._derive_next_audio_latents(audio, audio_sample_rate, result, segment_idx)
        self.continuation.save_audio_latents(next_audio_latents)

        timings["save_conditioning_ms"] = (time.perf_counter() - t_save_start) * 1000
        timings["e2e_latency_ms"] = (time.perf_counter() - t0) * 1000

        print(f"[GPU {self.gpu_id}] LTX2 segment {segment_idx}: "
              f"{len(frames)} frames, gen={timings['generation_ms']:.0f}ms, "
              f"save_conditioning={timings['save_conditioning_ms']:.0f}ms, "
              f"e2e={timings['e2e_latency_ms']:.0f}ms")

        # Head-trim values for downstream AV streaming — computed here so
        # the streaming layer never needs to know conditioning constants.
        is_continuation = segment_idx > 1 and not reset_conditioning
        head_trim_frames = (LTX2_VIDEO_CONDITIONING_NUM_FRAMES if is_continuation else 0)
        audio_extra = (max(0, AUDIO_CONDITIONING_NUM_FRAMES -
                           LTX2_VIDEO_CONDITIONING_NUM_FRAMES) if ENABLE_AUDIO_COND else 0)
        head_trim_audio_frames = (head_trim_frames + audio_extra if is_continuation else 0)

        return StepResult(
            frames=frames,
            audio=audio,
            audio_sample_rate=audio_sample_rate,
            timings=timings,
            head_trim_frames=head_trim_frames,
            head_trim_audio_frames=head_trim_audio_frames,
        )

    def _derive_next_audio_latents(
        self,
        audio: object,
        audio_sample_rate: int | None,
        result: dict,
        segment_idx: int,
    ) -> torch.Tensor | None:
        """Pick which tensor to cache for next-segment audio conditioning."""
        if not ENABLE_AUDIO_COND:
            return None
        if (ENABLE_AUDIO_RE_ENCODE and audio is not None and audio_sample_rate is not None):
            re_encoded = self._re_encode_audio(audio, audio_sample_rate)
            if re_encoded is not None:
                print(f"[GPU {self.gpu_id}] Re-encoded audio "
                      f"latents shape="
                      f"{tuple(re_encoded.shape)} "
                      f"for segment {segment_idx + 1}")
                return re_encoded
            return None
        audio_latents = result.get("ltx2_audio_latents")
        if audio_latents is not None:
            print(f"[GPU {self.gpu_id}] Cached audio latents "
                  f"shape={tuple(audio_latents.shape)} "
                  f"for segment {segment_idx + 1}")
            return audio_latents
        return None

    def warmup(self, prompt: str) -> dict[str, float]:
        warmup_prompt = (prompt or "").strip()
        if not warmup_prompt:
            raise RuntimeError("Startup warmup prompt must be non-empty.")

        print(f"[GPU {self.gpu_id}] Startup warmup starting "
              "(synthetic segments: seg1, seg2, seg1-post-LoRA)")
        warmup_t0 = time.perf_counter()

        r1 = self.generate_step(
            warmup_prompt,
            segment_idx=1,
            image_path=None,
            reset_conditioning=True,
        )
        r2 = self.generate_step(
            warmup_prompt,
            segment_idx=2,
            image_path=None,
            reset_conditioning=False,
        )
        # r1 stage 1 compiled BEFORE LoRA wrapping (which happens during
        # r1 stage 2 via ltx2_refine_lora_stage), so the resulting graph
        # is keyed off pre-LoRA module identity and is stale once r1 r2
        # finish. The first real user seg=1 then re-compiles stage 1
        # ("stage1-LoRA-nocont"), wasting ~90s on the user's first
        # request. Run a 3rd warmup pass with seg=1 reset=True after r2
        # so this graph is compiled while no client is waiting. r3
        # stage 2 hits r1 stage 2's cache (shape match, both LoRA-nocont)
        # so the only real work is the missing stage 1 graph.
        self.continuation.clear()
        r3 = self.generate_step(
            warmup_prompt,
            segment_idx=1,
            image_path=None,
            reset_conditioning=True,
        )
        warmup_total_ms = (time.perf_counter() - warmup_t0) * 1000.0
        self.continuation.clear()

        segment1_ms = float(r1.timings.get("e2e_latency_ms", 0.0))
        segment2_ms = float(r2.timings.get("e2e_latency_ms", 0.0))
        segment3_ms = float(r3.timings.get("e2e_latency_ms", 0.0))
        print(f"[GPU {self.gpu_id}] Startup warmup complete: "
              f"segment1={segment1_ms:.0f}ms, "
              f"segment2={segment2_ms:.0f}ms, "
              f"segment3={segment3_ms:.0f}ms, total={warmup_total_ms:.0f}ms")
        return {
            "warmup_segment1_ms": segment1_ms,
            "warmup_segment2_ms": segment2_ms,
            "warmup_segment3_ms": segment3_ms,
            "warmup_total_ms": warmup_total_ms,
        }
