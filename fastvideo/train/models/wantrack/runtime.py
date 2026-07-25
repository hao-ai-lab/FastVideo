# SPDX-License-Identifier: Apache-2.0
"""Reusable causal WanTrack interactive-inference runtime and session."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import io
import json
import os
from pathlib import Path
import threading
import time
from typing import Any, Literal

import numpy as np
from PIL import Image
import torch

from fastvideo.pipelines import TrainingBatch
from fastvideo.train.models.wantrack.control import StableGridController
from fastvideo.train.models.wantrack.inference import (
    clear_wantrack_caches,
    sample_wantrack_block,
)


@dataclass(frozen=True, slots=True)
class WanTrackSamplingSettings:
    seed: int = 0
    num_inference_steps: int = 30
    text_guidance_scale: float = 3.0
    motion_guidance_scale: float = 1.5
    motion_cfg: bool = True

    @classmethod
    def from_value(
        cls,
        value: WanTrackSamplingSettings | dict[str, Any] | None,
    ) -> WanTrackSamplingSettings:
        if value is None:
            result = cls()
        elif isinstance(value, cls):
            result = value
        elif isinstance(value, dict):
            aliases = dict(value)
            if "steps" in aliases:
                aliases["num_inference_steps"] = aliases.pop("steps")
            if "text_guidance" in aliases:
                aliases["text_guidance_scale"] = aliases.pop("text_guidance")
            if "motion_guidance" in aliases:
                aliases["motion_guidance_scale"] = aliases.pop("motion_guidance")
            result = cls(**aliases)
        else:
            raise TypeError("sampling must be a mapping or settings object")
        if result.num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be positive")
        if result.text_guidance_scale < 0 or result.motion_guidance_scale < 0:
            raise ValueError("guidance scales must be non-negative")
        return result


@dataclass(slots=True)
class PreparedWanTrackInput:
    image: Image.Image
    prompt: str
    batch: TrainingBatch
    latent_channels: int
    latent_height: int
    latent_width: int


@dataclass(frozen=True, slots=True)
class GeneratedWanTrackBlock:
    block_index: int
    latent_start: int
    latent_frames: int
    pixel_frames: np.ndarray
    applied_revision: int
    radius: float
    active_handle_ids: tuple[str, ...]


class WanTrackInferenceRuntime:
    """Own heavyweight components shared by one active causal session."""

    def __init__(
        self,
        *,
        model: Any,
        vae: Any,
        text_encoder: Any,
        tokenizer: Any,
        image_encoder: Any,
        image_processor: Any,
        training_config: Any,
        raw_config: dict[str, Any],
        model_dir: str,
        fps: float,
        chunk_size: int,
    ) -> None:
        self.model = model
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.image_encoder = image_encoder
        self.image_processor = image_processor
        self.training_config = training_config
        self.raw_config = raw_config
        self.model_dir = str(model_dir)
        self.fps = float(fps)
        self.chunk_size = int(chunk_size)
        self.temporal_compression = int(
            training_config.pipeline_config.vae_config.arch_config.temporal_compression_ratio)
        self.height = int(training_config.data.num_height)
        self.width = int(training_config.data.num_width)
        if self.height <= 0 or self.width <= 0:
            raise ValueError("WanTrack YAML must define positive training.data.num_height "
                             "and num_width")
        if self.chunk_size <= 0:
            raise ValueError("causal WanTrack chunk size must be positive")
        if self.fps <= 0:
            raise ValueError("WanTrack FPS must be positive")

    @classmethod
    def from_export(
        cls,
        model_dir: str | os.PathLike[str],
        yaml_path: str | os.PathLike[str],
    ) -> WanTrackInferenceRuntime:
        """Load a Diffusers-format causal export and its training YAML."""
        model_dir = str(Path(model_dir).expanduser().resolve())
        yaml_path = str(Path(yaml_path).expanduser().resolve())
        model_index_path = Path(model_dir) / "model_index.json"
        if not Path(model_dir).is_dir():
            raise FileNotFoundError(f"WanTrack model directory not found: {model_dir}")
        if not Path(yaml_path).is_file():
            raise FileNotFoundError(f"WanTrack YAML not found: {yaml_path}")
        if not model_index_path.is_file():
            raise ValueError(f"Diffusers export is missing {model_index_path.name}: {model_dir}")

        with model_index_path.open(encoding="utf-8") as handle:
            model_index = json.load(handle)
        transformer_info = model_index.get("transformer")
        if (not isinstance(transformer_info, list | tuple) or len(transformer_info) < 2
                or "CausalTrackWan" not in str(transformer_info[1])):
            raise ValueError("WanTrack interactive inference requires a causal "
                             "CausalTrackWanTransformer3DModel export")

        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("PyYAML is required to load the WanTrack YAML") from exc
        with open(yaml_path, encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        if not isinstance(raw, dict):
            raise ValueError("WanTrack YAML root must be a mapping")
        models = deepcopy(raw.get("models"))
        if not isinstance(models, dict) or not isinstance(models.get("student"), dict):
            raise ValueError("WanTrack YAML must define models.student")
        target = str(models["student"].get("_target_", ""))
        if "WanTrackCausalModel" not in target:
            raise ValueError("WanTrack YAML models.student must target "
                             "WanTrackCausalModel")
        models["student"]["init_from"] = model_dir

        from fastvideo.train.utils.config import (
            _build_training_config,
            _parse_pipeline_config,
        )

        pipeline_config = _parse_pipeline_config(raw, models=models)
        training_raw = raw.get("training")
        if not isinstance(training_raw, dict):
            raise ValueError("WanTrack YAML must define training")
        training_config = _build_training_config(
            training_raw,
            models=models,
            pipeline_config=pipeline_config,
        )
        training_config.model_path = model_dir
        # Standalone inference is single-rank and does not initialize a
        # training dataloader.
        training_config.distributed.num_gpus = 1
        training_config.distributed.tp_size = 1
        training_config.distributed.sp_size = 1
        training_config.distributed.hsdp_replicate_dim = 1
        training_config.distributed.hsdp_shard_dim = 1

        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        from fastvideo.distributed import (
            maybe_init_distributed_environment_and_model_parallel, )

        maybe_init_distributed_environment_and_model_parallel(1, 1)

        from fastvideo.train.models.wantrack.wantrack_causal import (
            WanTrackCausalModel, )
        from fastvideo.train.utils.moduleloader import (
            load_module_from_path,
            make_inference_args,
        )
        from fastvideo.models.loader.component_loader import (
            PipelineComponentLoader, )

        model = WanTrackCausalModel(
            init_from=model_dir,
            training_config=training_config,
            trainable=False,
        )
        model.transformer.eval()
        vae = load_module_from_path(
            model_path=model_dir,
            module_type="vae",
            training_config=training_config,
        ).eval()
        model.vae = vae

        inference_args = make_inference_args(
            training_config,
            model_path=model_dir,
        )

        def load_component(name: str, required: bool = True) -> Any:
            info = model_index.get(name)
            if info is None:
                if required:
                    raise ValueError(f"WanTrack export is missing required {name!r} component")
                return None
            if not isinstance(info, list | tuple) or len(info) < 2:
                raise ValueError(f"Invalid {name!r} component entry in model_index.json")
            component = PipelineComponentLoader.load_module(
                module_name=name,
                component_model_path=str(Path(model_dir) / name),
                transformers_or_diffusers=str(info[0]),
                fastvideo_args=inference_args,
            )
            if component is None and required:
                raise RuntimeError(f"Failed to load WanTrack component {name!r}")
            return component

        text_encoder = load_component("text_encoder")
        tokenizer = load_component("tokenizer")
        image_encoder = load_component("image_encoder")
        image_processor_name = ("image_processor"
                                if model_index.get("image_processor") is not None else "feature_extractor")
        image_processor = load_component(image_processor_name)

        method = raw.get("method", {})
        chunk_size = int(
            method.get(
                "chunk_size",
                getattr(
                    model.transformer,
                    "num_frame_per_block",
                    getattr(
                        model.transformer.config.arch_config,
                        "num_frames_per_block",
                        3,
                    ),
                ),
            ))
        fps = cls._fps_from_yaml(raw)
        return cls(
            model=model,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            image_encoder=image_encoder,
            image_processor=image_processor,
            training_config=training_config,
            raw_config=raw,
            model_dir=model_dir,
            fps=fps,
            chunk_size=chunk_size,
        )

    @staticmethod
    def _fps_from_yaml(raw: dict[str, Any]) -> float:
        candidates = [
            raw.get("fps"),
            (raw.get("pipeline") or {}).get("fps") if isinstance(raw.get("pipeline"), dict) else None,
            ((raw.get("training") or {}).get("data") or {}).get("fps")
            if isinstance(raw.get("training"), dict) else None,
            ((raw.get("callbacks") or {}).get("track_validation") or {}).get("fps") if isinstance(
                raw.get("callbacks"), dict) else None,
        ]
        for value in candidates:
            if value is not None and float(value) > 0:
                return float(value)
        return 16.0

    @staticmethod
    def preprocess_image(
        image: Image.Image,
        *,
        width: int,
        height: int,
    ) -> Image.Image:
        if width <= 0 or height <= 0:
            raise ValueError("target image dimensions must be positive")
        image = image.convert("RGB")
        source_ratio = image.width / image.height
        target_ratio = width / height
        if source_ratio > target_ratio:
            crop_width = round(image.height * target_ratio)
            left = (image.width - crop_width) // 2
            box = (left, 0, left + crop_width, image.height)
        else:
            crop_height = round(image.width / target_ratio)
            top = (image.height - crop_height) // 2
            box = (0, top, image.width, top + crop_height)
        return image.crop(box).resize((width, height), Image.Resampling.LANCZOS)

    @torch.no_grad()
    def prepare(
        self,
        image: Image.Image | bytes,
        prompt: str = "",
    ) -> PreparedWanTrackInput:
        if isinstance(image, bytes):
            image = Image.open(io.BytesIO(image))
        if not isinstance(image, Image.Image):
            raise TypeError("image must be PIL.Image.Image or encoded bytes")
        prompt = str(prompt or "")
        processed = self.preprocess_image(
            image,
            width=self.width,
            height=self.height,
        )
        device = self.model.device
        dtype = torch.bfloat16

        from fastvideo.forward_context import set_forward_context
        from fastvideo.pipelines.stages.text_encoding import TextEncodingStage

        inference_args = self._inference_args()
        text_stage = TextEncodingStage(
            text_encoders=[self.text_encoder],
            tokenizers=[self.tokenizer],
        )
        embeddings, masks = text_stage.encode_text(
            prompt,
            inference_args,
            return_attention_mask=True,
            device=device,
            dtype=dtype,
        )
        text_embedding = embeddings[0]
        text_mask = masks[0].to(device=device, dtype=dtype)

        image_inputs = self.image_processor(
            images=processed,
            return_tensors="pt",
        )
        image_encoder = self.image_encoder.to(device)
        image_inputs = {key: value.to(device) for key, value in image_inputs.items() if torch.is_tensor(value)}
        with set_forward_context(current_timestep=0, attn_metadata=None):
            image_output = image_encoder(**image_inputs)
        clip_feature = image_output.last_hidden_state.to(dtype=dtype)

        array = np.asarray(processed, dtype=np.float32) / 127.5 - 1.0
        pixels = torch.from_numpy(array).permute(2, 0, 1)
        pixels = pixels.unsqueeze(0).unsqueeze(2).to(
            device=device,
            dtype=torch.float32,
        )
        vae = self.vae.to(device)
        encoded = vae.encode(pixels)
        first_latent = encoded.mean
        shift = getattr(vae, "shift_factor", None)
        scale = getattr(vae, "scaling_factor", None)
        if shift is not None:
            first_latent = first_latent - torch.as_tensor(
                shift,
                device=device,
                dtype=first_latent.dtype,
            )
        if scale is not None:
            first_latent = first_latent * torch.as_tensor(
                scale,
                device=device,
                dtype=first_latent.dtype,
            )
        first_latent = first_latent.to(dtype=dtype)
        image_condition = self.model._build_i2v_condition(first_latent)
        latent_channels = int(self.training_config.pipeline_config.vae_config.arch_config.z_dim)
        latent_height = int(first_latent.shape[-2])
        latent_width = int(first_latent.shape[-1])
        latents = torch.zeros(
            1,
            1,
            latent_channels,
            latent_height,
            latent_width,
            device=device,
            dtype=dtype,
        )
        conditional = {
            "encoder_hidden_states": text_embedding,
            "encoder_attention_mask": text_mask,
            "encoder_hidden_states_image": clip_feature,
            "image_latents": image_condition,
            "track_points": None,
            "track_visibility": None,
            "track_ids": None,
            "track_map": None,
        }
        batch = TrainingBatch(
            latents=latents,
            encoder_hidden_states=text_embedding,
            encoder_attention_mask=text_mask,
            image_embeds=clip_feature,
            image_latents=image_condition,
            conditional_dict=conditional,
            unconditional_dict=None,
            attn_metadata=None,
            attn_metadata_vsa=None,
        )
        return PreparedWanTrackInput(
            image=processed,
            prompt=prompt,
            batch=batch,
            latent_channels=latent_channels,
            latent_height=latent_height,
            latent_width=latent_width,
        )

    def _inference_args(self) -> Any:
        from fastvideo.train.utils.moduleloader import make_inference_args

        return make_inference_args(
            self.training_config,
            model_path=self.model_dir,
        )

    def create_session(self) -> CausalWanTrackSession:
        return CausalWanTrackSession(self)

    @torch.no_grad()
    def encode_track_window(
        self,
        *,
        points: np.ndarray,
        visibility: np.ndarray,
        pixel_start: int,
        latent_start: int,
        latent_t: int,
        latent_h: int,
        latent_w: int,
        track_ids: torch.Tensor,
    ) -> torch.Tensor:
        encoder = self.model.transformer.track_encoder
        device = self.model.device
        return encoder.forward_window(
            torch.from_numpy(np.ascontiguousarray(points)).unsqueeze(0).to(
                device=device,
                dtype=torch.float32,
            ),
            torch.from_numpy(np.ascontiguousarray(visibility)).unsqueeze(0).to(
                device=device,
                dtype=torch.float32,
            ),
            latent_start=latent_start,
            latent_t=latent_t,
            latent_h=latent_h,
            latent_w=latent_w,
            pixel_start=pixel_start,
            track_ids=track_ids.to(device),
        )

    def new_vae_cache(self) -> list[torch.Tensor | None] | None:
        getter = getattr(self.vae, "get_streaming_cache", None)
        return getter() if callable(getter) else None

    @torch.no_grad()
    def decode_block(
        self,
        latents: torch.Tensor,
        *,
        cache: list[torch.Tensor | None] | None,
        first: bool,
    ) -> tuple[np.ndarray, list[torch.Tensor | None] | None]:
        normalized = latents.permute(0, 2, 1, 3, 4).float()
        if bool(getattr(self.vae, "handles_latent_denorm", False)):
            denormalized = normalized
        else:
            mean = torch.tensor(
                self.vae.latents_mean,
                device=normalized.device,
                dtype=normalized.dtype,
            ).view(1, -1, 1, 1, 1)
            std = torch.tensor(
                self.vae.latents_std,
                device=normalized.device,
                dtype=normalized.dtype,
            ).view(1, -1, 1, 1, 1)
            denormalized = normalized * std + mean
        if cache is not None and hasattr(self.vae, "streaming_decode"):
            media, cache = self.vae.streaming_decode(
                denormalized,
                cache,
                is_first_chunk=first,
            )
        else:
            media = self.vae.decode(denormalized)
        frames = ((media / 2 + 0.5).clamp(0, 1) * 255).round()
        frames = frames[0].permute(1, 2, 3, 0).byte().cpu().numpy()
        return np.ascontiguousarray(frames), cache

    def clear_state(self) -> None:
        clear_wantrack_caches(self.model)
        clear_vae = getattr(self.vae, "clear_cache", None)
        if callable(clear_vae):
            clear_vae()


class CausalWanTrackSession:
    """One causal rollout with immutable committed control history."""

    def __init__(self, runtime: WanTrackInferenceRuntime) -> None:
        self.runtime = runtime
        self._lock = threading.RLock()
        self._state: Literal["created", "running", "closed", "failed"] = ("created")
        self._prepared: PreparedWanTrackInput | None = None
        self._sampling: WanTrackSamplingSettings | None = None
        self._controller: StableGridController | None = None
        self._noise_generator: torch.Generator | None = None
        self._track_ids: torch.Tensor | None = None
        self._batch: TrainingBatch | None = None
        self._vae_cache: list[torch.Tensor | None] | None = None
        self._history: list[np.ndarray] = []
        self._block_index = 0
        self._latent_start = 0
        self._session_started_ms = 0.0
        self._block_started_ms = 0.0
        self._close_reason: str | None = None

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    @property
    def committed_control_history(self) -> tuple[np.ndarray, ...]:
        with self._lock:
            return tuple(item.copy() for item in self._history)

    @property
    def block_index(self) -> int:
        with self._lock:
            return self._block_index

    def start(
        self,
        image: PreparedWanTrackInput | Image.Image | bytes,
        prompt: str,
        handles: list[dict[str, Any]],
        sampling: WanTrackSamplingSettings | dict[str, Any] | None = None,
        *,
        radius: float = 0.15,
    ) -> None:
        with self._lock:
            if self._state != "created":
                raise RuntimeError("WanTrack session can only be started once")
        prepared = (image if isinstance(image, PreparedWanTrackInput) else self.runtime.prepare(image, prompt))
        settings = WanTrackSamplingSettings.from_value(sampling)
        controller = StableGridController(handles, radius=radius)
        generator = torch.Generator(device="cpu").manual_seed(settings.seed)
        track_ids = torch.arange(
            controller.grid.shape[0],
            dtype=torch.long,
            device="cpu",
        ).unsqueeze(0)
        batch = prepared.batch
        batch.track_ids = track_ids.to(self.runtime.model.device)
        if batch.conditional_dict is None:
            raise RuntimeError("prepared WanTrack input is missing conditions")
        batch.conditional_dict["track_ids"] = batch.track_ids
        self.runtime.clear_state()
        now = time.monotonic() * 1000.0
        with self._lock:
            self._prepared = prepared
            self._sampling = settings
            self._controller = controller
            self._noise_generator = generator
            self._track_ids = batch.track_ids
            self._batch = batch
            self._vae_cache = self.runtime.new_vae_cache()
            self._session_started_ms = now
            self._block_started_ms = now
            self._state = "running"

    def apply_control_revision(
        self,
        revision: int,
        *,
        samples: list[dict[str, Any]] | None = None,
        add: list[dict[str, Any]] | None = None,
        remove: list[str] | None = None,
        handles: list[dict[str, Any]] | None = None,
        radius: float | None = None,
        received_at_ms: float | None = None,
    ) -> bool:
        with self._lock:
            if self._state != "running" or self._controller is None:
                raise RuntimeError("WanTrack session is not running")
            controller = self._controller
            if received_at_ms is None:
                received_at_ms = (time.monotonic() * 1000.0 - self._session_started_ms)
        return controller.queue_revision(
            revision,
            samples=samples,
            add=add,
            remove=remove,
            handles=handles,
            radius=radius,
            received_at_ms=received_at_ms,
        )

    @torch.no_grad()
    def generate_next_block(self) -> GeneratedWanTrackBlock:
        with self._lock:
            if self._state != "running":
                raise RuntimeError("WanTrack session is not running")
            assert self._prepared is not None
            assert self._sampling is not None
            assert self._controller is not None
            assert self._noise_generator is not None
            assert self._track_ids is not None
            assert self._batch is not None
            prepared = self._prepared
            settings = self._sampling
            controller = self._controller
            generator = self._noise_generator
            track_ids = self._track_ids
            batch = self._batch
            block_index = self._block_index
            latent_start = self._latent_start
            previous_block_started_ms = self._block_started_ms
            block_started_ms = (time.monotonic() * 1000.0 - self._session_started_ms)
            self._block_started_ms = block_started_ms

        latent_frames = 1 if block_index == 0 else self.runtime.chunk_size
        pixel_frames = (1 if block_index == 0 else latent_frames * self.runtime.temporal_compression)
        if block_index == 0:
            control = controller.render_constant(pixel_frames)
        else:
            control = controller.apply_pending(
                pixel_frames,
                interval_start_ms=previous_block_started_ms,
                interval_end_ms=block_started_ms,
            )

        candidate_history = self._history + [control.tracks.copy()]
        all_points = np.concatenate(candidate_history, axis=0)
        all_visibility = np.ones(all_points.shape[:2], dtype=np.float32)
        ratio = self.runtime.temporal_compression
        required_start = max(0, latent_start * ratio - (ratio - 1))
        required_end = (latent_start + latent_frames - 1) * ratio + 1
        if required_end > all_points.shape[0]:
            raise RuntimeError("control history does not cover the next causal block")
        window_points = all_points[required_start:required_end]
        window_visibility = all_visibility[required_start:required_end]

        try:
            track_map = self.runtime.encode_track_window(
                points=window_points,
                visibility=window_visibility,
                pixel_start=required_start,
                latent_start=latent_start,
                latent_t=latent_frames,
                latent_h=prepared.latent_height,
                latent_w=prepared.latent_width,
                track_ids=track_ids,
            )
            batch.track_map = track_map
            batch.track_points = None
            batch.track_visibility = None
            assert batch.conditional_dict is not None
            batch.conditional_dict["track_map"] = track_map
            batch.conditional_dict["track_points"] = None
            batch.conditional_dict["track_visibility"] = None
            noise = torch.randn(
                (
                    1,
                    latent_frames,
                    prepared.latent_channels,
                    prepared.latent_height,
                    prepared.latent_width,
                ),
                generator=generator,
                dtype=torch.float32,
                device="cpu",
            ).to(
                device=self.runtime.model.device,
                dtype=torch.bfloat16,
            )
            latents = sample_wantrack_block(
                self.runtime.model,
                batch,
                noise,
                start_frame=latent_start,
                num_inference_steps=settings.num_inference_steps,
                text_guidance_scale=settings.text_guidance_scale,
                motion_guidance_scale=settings.motion_guidance_scale,
                motion_cfg=settings.motion_cfg,
                commit=True,
            )
            decoded, vae_cache = self.runtime.decode_block(
                latents,
                cache=self._vae_cache,
                first=(block_index == 0),
            )
        except Exception:
            with self._lock:
                self._state = "failed"
                self._close_reason = "error"
            self.runtime.clear_state()
            raise

        with self._lock:
            self._history.append(control.tracks.copy())
            self._vae_cache = vae_cache
            self._block_index += 1
            self._latent_start += latent_frames
        return GeneratedWanTrackBlock(
            block_index=block_index,
            latent_start=latent_start,
            latent_frames=latent_frames,
            pixel_frames=decoded,
            applied_revision=control.revision,
            radius=control.radius,
            active_handle_ids=control.active_handle_ids,
        )

    def close(self, reason: str = "stop") -> None:
        with self._lock:
            if self._state == "closed":
                return
            self._state = "closed"
            self._close_reason = str(reason)
            self._vae_cache = None
        self.runtime.clear_state()
