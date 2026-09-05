# SPDX-License-Identifier: Apache-2.0
"""Component-disaggregated MiniMax H3 stage runners and wire contracts."""

from __future__ import annotations

from dataclasses import dataclass, replace
from uuid import uuid4

import torch

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.pipelines.basic.minimax_h3.minimax_h3_pipeline import (
    MiniMaxH3BasePipeline,
    _apply_h3_checkpoint_arch_configs,
    _use_taeh3_t2va,
)
from fastvideo.pipelines.basic.minimax_h3.packing import MiniMaxH3PackedLayout
from fastvideo.pipelines.basic.minimax_h3.stages import (
    MiniMaxH3AudioDecodingStage,
    MiniMaxH3DenoisingStage,
    MiniMaxH3LatentPreparationStage,
    MiniMaxH3VideoDecodingStage,
)
from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_latent_preparation import MINIMAX_H3_LAYOUT_KEY
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch, PipelineLoggingInfo

MINIMAX_H3_WIRE_SCHEMA_VERSION = 1


def _cuda_tensor(value: torch.Tensor, name: str, dimensions: int) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.ndim != dimensions:
        shape = None if not isinstance(value, torch.Tensor) else tuple(value.shape)
        raise ValueError(f"MiniMax-H3 wire field {name!r} must be a {dimensions}D tensor, got {shape}.")
    return value.detach().to(device="cuda", non_blocking=True).contiguous()


def _cuda_layout(layout: MiniMaxH3PackedLayout) -> MiniMaxH3PackedLayout:
    if not isinstance(layout, MiniMaxH3PackedLayout):
        raise TypeError("MiniMax-H3 encode output is missing its packed layout.")
    return replace(
        layout,
        position_ids=_cuda_tensor(layout.position_ids, "layout.position_ids", 2),
        token_tags=_cuda_tensor(layout.token_tags, "layout.token_tags", 1),
        video_indices=_cuda_tensor(layout.video_indices, "layout.video_indices", 1),
        audio_indices=_cuda_tensor(layout.audio_indices, "layout.audio_indices", 1),
        text_indices=_cuda_tensor(layout.text_indices, "layout.text_indices", 1),
    )


def _validate_schema(schema_version: int) -> None:
    if schema_version != MINIMAX_H3_WIRE_SCHEMA_VERSION:
        raise ValueError("Unsupported MiniMax-H3 wire schema version "
                         f"{schema_version}; expected {MINIMAX_H3_WIRE_SCHEMA_VERSION}.")


def _raw_latent_shape(batch: ForwardBatch, stage: str) -> tuple[int, int, int, int, int]:
    shape = batch.raw_latent_shape
    if shape is None or len(shape) != 5:
        raise ValueError(f"MiniMax-H3 {stage} must produce a five-dimensional raw latent shape.")
    return (int(shape[0]), int(shape[1]), int(shape[2]), int(shape[3]), int(shape[4]))


def _validate_cuda_layout(layout: MiniMaxH3PackedLayout) -> None:
    if not isinstance(layout, MiniMaxH3PackedLayout):
        raise TypeError("MiniMax-H3 wire payload requires a packed layout.")
    expected_shapes = {
        "position_ids": (layout.sequence_length, 3),
        "token_tags": (layout.sequence_length, ),
    }
    for name, tensor in (
        ("position_ids", layout.position_ids),
        ("token_tags", layout.token_tags),
        ("video_indices", layout.video_indices),
        ("audio_indices", layout.audio_indices),
        ("text_indices", layout.text_indices),
    ):
        if not tensor.is_cuda or not tensor.is_contiguous():
            raise ValueError(f"MiniMax-H3 wire field layout.{name} must be a contiguous CUDA tensor.")
        expected = expected_shapes.get(name)
        if expected is not None and tuple(tensor.shape) != expected:
            raise ValueError(
                f"MiniMax-H3 wire field layout.{name} has shape {tuple(tensor.shape)}, expected {expected}.")
        if expected is None and tensor.ndim != 1:
            raise ValueError(f"MiniMax-H3 wire field layout.{name} must be one-dimensional.")


@dataclass(frozen=True)
class MiniMaxH3EncodedState:
    """Minimal CUDA payload sent from the encoder/VAE node to the DiT node."""

    request_id: str
    prompt_embeds: torch.Tensor
    video_latents: torch.Tensor
    audio_latents: torch.Tensor
    layout: MiniMaxH3PackedLayout
    raw_latent_shape: tuple[int, int, int, int, int]
    num_inference_steps: int
    vsa_sparsity: float
    vsa_mode: str = "exempt"
    vsa_dense_first_n_steps: int = 0
    vsa_dense_layers: tuple[int, ...] = ()
    logging_info: PipelineLoggingInfo | None = None
    schema_version: int = MINIMAX_H3_WIRE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_schema(self.schema_version)
        if not self.request_id:
            raise ValueError("MiniMax-H3 wire payload requires a non-empty request_id.")
        if len(self.raw_latent_shape) != 5 or min(self.raw_latent_shape) <= 0:
            raise ValueError(f"Invalid MiniMax-H3 raw latent shape: {self.raw_latent_shape}.")
        if self.num_inference_steps <= 0:
            raise ValueError("MiniMax-H3 wire payload requires at least one denoising step.")
        if self.vsa_mode not in {"exempt", "compete"}:
            raise ValueError(f"vsa_mode must be 'exempt' or 'compete', got {self.vsa_mode!r}.")
        for name, tensor, dimensions in (
            ("prompt_embeds", self.prompt_embeds, 3),
            ("video_latents", self.video_latents, 2),
            ("audio_latents", self.audio_latents, 2),
        ):
            if not tensor.is_cuda or not tensor.is_contiguous() or tensor.ndim != dimensions:
                raise ValueError(f"MiniMax-H3 wire field {name!r} must be a contiguous CUDA {dimensions}D tensor.")
        _validate_cuda_layout(self.layout)
        if self.video_latents.shape[0] != self.layout.video_indices.numel():
            raise ValueError("MiniMax-H3 video latent rows do not match layout.video_indices.")
        if self.audio_latents.shape[0] != self.layout.audio_indices.numel():
            raise ValueError("MiniMax-H3 audio latent rows do not match layout.audio_indices.")
        if self.prompt_embeds.shape[1] != self.layout.text_indices.numel():
            raise ValueError("MiniMax-H3 prompt embedding rows do not match layout.text_indices.")

    @classmethod
    def from_batch(cls, batch: ForwardBatch, request_id: str | None = None) -> MiniMaxH3EncodedState:
        if len(batch.prompt_embeds) != 1 or batch.latents is None or batch.audio_latents is None:
            raise ValueError("MiniMax-H3 encode must produce one prompt embedding and both latent streams.")
        layout = batch.extra.get(MINIMAX_H3_LAYOUT_KEY)
        return cls(
            request_id=request_id if request_id is not None else uuid4().hex,
            prompt_embeds=_cuda_tensor(batch.prompt_embeds[0], "prompt_embeds", 3),
            video_latents=_cuda_tensor(batch.latents, "video_latents", 2),
            audio_latents=_cuda_tensor(batch.audio_latents, "audio_latents", 2),
            layout=_cuda_layout(layout),
            raw_latent_shape=_raw_latent_shape(batch, "encode"),
            num_inference_steps=int(batch.num_inference_steps),
            vsa_sparsity=float(batch.VSA_sparsity),
            vsa_mode=str(batch.extra.get("vsa_mode", "exempt")),
            vsa_dense_first_n_steps=int(batch.extra.get("vsa_dense_first_n_steps", 0)),
            vsa_dense_layers=tuple(int(layer) for layer in batch.extra.get("vsa_dense_layers", ())),
            logging_info=batch.logging_info,
        )

    def to_batch(self) -> ForwardBatch:
        # Pickle/Ray reconstruction does not invoke dataclass __post_init__.
        # Revalidate the schema and CUDA tensor contract at the receiving boundary.
        self.__post_init__()
        return ForwardBatch(
            data_type="video",
            prompt_embeds=[self.prompt_embeds],
            latents=self.video_latents,
            audio_latents=self.audio_latents,
            raw_latent_shape=self.raw_latent_shape,
            num_inference_steps=self.num_inference_steps,
            VSA_sparsity=self.vsa_sparsity,
            extra={
                MINIMAX_H3_LAYOUT_KEY: self.layout,
                "request_id": self.request_id,
                "vsa_mode": self.vsa_mode,
                "vsa_dense_first_n_steps": self.vsa_dense_first_n_steps,
                "vsa_dense_layers": self.vsa_dense_layers,
            },
            logging_info=self.logging_info or PipelineLoggingInfo(),
        )


@dataclass(frozen=True)
class MiniMaxH3DenoisedState:
    """Minimal CUDA payload returned by the DiT node for VAE decoding."""

    request_id: str
    video_latents: torch.Tensor
    audio_latents: torch.Tensor
    layout: MiniMaxH3PackedLayout
    raw_latent_shape: tuple[int, int, int, int, int]
    logging_info: PipelineLoggingInfo | None = None
    schema_version: int = MINIMAX_H3_WIRE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_schema(self.schema_version)
        if not self.request_id:
            raise ValueError("MiniMax-H3 wire payload requires a non-empty request_id.")
        if len(self.raw_latent_shape) != 5 or min(self.raw_latent_shape) <= 0:
            raise ValueError(f"Invalid MiniMax-H3 raw latent shape: {self.raw_latent_shape}.")
        for name, tensor in (("video_latents", self.video_latents), ("audio_latents", self.audio_latents)):
            if not tensor.is_cuda or not tensor.is_contiguous() or tensor.ndim != 2:
                raise ValueError(f"MiniMax-H3 wire field {name!r} must be a contiguous CUDA 2D tensor.")
        _validate_cuda_layout(self.layout)
        if self.video_latents.shape[0] != self.layout.video_indices.numel():
            raise ValueError("MiniMax-H3 video latent rows do not match layout.video_indices.")
        if self.audio_latents.shape[0] != self.layout.audio_indices.numel():
            raise ValueError("MiniMax-H3 audio latent rows do not match layout.audio_indices.")

    @classmethod
    def from_batch(cls, batch: ForwardBatch, request_id: str) -> MiniMaxH3DenoisedState:
        if batch.latents is None or batch.audio_latents is None:
            raise ValueError("MiniMax-H3 denoise must return both latent streams.")
        return cls(
            request_id=request_id,
            video_latents=_cuda_tensor(batch.latents, "video_latents", 2),
            audio_latents=_cuda_tensor(batch.audio_latents, "audio_latents", 2),
            layout=_cuda_layout(batch.extra.get(MINIMAX_H3_LAYOUT_KEY)),
            raw_latent_shape=_raw_latent_shape(batch, "denoise"),
            logging_info=batch.logging_info,
        )

    def to_batch(self) -> ForwardBatch:
        # Revalidate after transport; unpickling itself skips __post_init__.
        self.__post_init__()
        return ForwardBatch(
            data_type="video",
            latents=self.video_latents,
            audio_latents=self.audio_latents,
            raw_latent_shape=self.raw_latent_shape,
            extra={
                MINIMAX_H3_LAYOUT_KEY: self.layout,
                "request_id": self.request_id,
            },
            logging_info=self.logging_info or PipelineLoggingInfo(),
        )


class _MiniMaxH3ResidentRolePipeline(MiniMaxH3BasePipeline):
    """Base that disables H3's single-worker release/reload lifecycle."""

    _lazy_module_names: tuple[str, ...] = ()

    def _defer_denoise_modules(self, fastvideo_args: FastVideoArgs) -> bool:
        del fastvideo_args
        return False

    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        del batch, fastvideo_args
        raise RuntimeError("Component-disaggregated H3 pipelines expose encode(), denoise(), and decode().")


class MiniMaxH3EncoderDecoderPipeline(_MiniMaxH3ResidentRolePipeline):
    """Persistent Qwen + video/audio VAE worker for T2VA and FL2VA."""

    _required_config_modules = ["text_encoder", "tokenizer", "processor", "vae", "audio_vae", "scheduler"]

    @classmethod
    def get_hf_download_allow_patterns(cls) -> list[str]:
        patterns = super().get_hf_download_allow_patterns() or []
        # Packing uses the checkpoint's DiT patch geometry, but this role must
        # never download or instantiate DiT weights. The text-encoder loader
        # also consults transformer/config.json for connector RoPE metadata.
        geometry_dir = cls._extra_config_module_map.get("transformer", "transformer")
        metadata = {"transformer/config.json", f"{geometry_dir}/config.json"}
        return [*patterns, *(pattern for pattern in sorted(metadata) if pattern not in patterns)]

    def initialize_pipeline(self, fastvideo_args: FastVideoArgs) -> None:
        _apply_h3_checkpoint_arch_configs(self.model_path, fastvideo_args, self._extra_config_module_map)
        shift = getattr(self.get_module("scheduler"), "shift", None)
        if shift is None or float(shift) != 12.0:
            raise ValueError(f"MiniMax-H3 video scheduler must expose shift=12, got {shift}.")

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs) -> None:
        ref2va = self._ref2va
        self._add_condition_stages(fastvideo_args, ref2va=ref2va)
        vae = self.get_module("vae")
        audio_vae = self.get_module("audio_vae")
        scheduler = self.get_module("scheduler")
        if vae is None or audio_vae is None or scheduler is None:
            raise RuntimeError("MiniMax-H3 encoder/decoder worker requires both VAEs and the video scheduler.")
        self.add_stage(
            "latent_preparation_stage",
            MiniMaxH3LatentPreparationStage(vae=vae, audio_vae=audio_vae, scheduler=scheduler, ref2va=ref2va),
        )
        use_taeh3 = _use_taeh3_t2va(fastvideo_args, ref2va=ref2va)
        self.add_stage("video_decoding_stage", MiniMaxH3VideoDecodingStage(vae=None if use_taeh3 else vae))
        self.add_stage("audio_decoding_stage", MiniMaxH3AudioDecodingStage(audio_vae=audio_vae))

    def encode(self, batch: ForwardBatch, request_id: str | None = None) -> MiniMaxH3EncodedState:
        if not self.post_init_called:
            self.post_init()
        for name in ("input_preparation_stage", "conditioning_stage", "latent_preparation_stage"):
            batch = self._stage_name_mapping[name](batch, self.fastvideo_args)
        return MiniMaxH3EncodedState.from_batch(batch, request_id=request_id)

    def decode(self, state: MiniMaxH3DenoisedState) -> ForwardBatch:
        if not self.post_init_called:
            self.post_init()
        batch = state.to_batch()
        for name in ("video_decoding_stage", "audio_decoding_stage"):
            batch = self._stage_name_mapping[name](batch, self.fastvideo_args)
        batch.extra["request_id"] = state.request_id
        return batch


class MiniMaxH3RefEncoderDecoderPipeline(MiniMaxH3EncoderDecoderPipeline):
    """Persistent encoder/decoder worker for Ref2VA requests."""

    _extra_config_module_map = {"transformer": "transformer_ref"}
    _ref2va_default = True


class MiniMaxH3DiTPipeline(_MiniMaxH3ResidentRolePipeline):
    """Persistent DiT-only worker for T2VA and FL2VA."""

    _required_config_modules = ["transformer", "scheduler", "audio_scheduler"]

    def initialize_pipeline(self, fastvideo_args: FastVideoArgs) -> None:
        _apply_h3_checkpoint_arch_configs(self.model_path, fastvideo_args, self._extra_config_module_map)
        for module_name, modality, expected_shift in (
            ("scheduler", "video", 12.0),
            ("audio_scheduler", "audio", 3.0),
        ):
            shift = getattr(self.get_module(module_name), "shift", None)
            if shift is None or float(shift) != expected_shift:
                raise ValueError(f"MiniMax-H3 {modality} scheduler must expose shift={expected_shift:g}, got {shift}.")

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs) -> None:
        del fastvideo_args
        transformer = self.get_module("transformer")
        scheduler = self.get_module("scheduler")
        audio_scheduler = self.get_module("audio_scheduler")
        if transformer is None or scheduler is None or audio_scheduler is None:
            raise RuntimeError("MiniMax-H3 DiT worker requires transformer and both schedulers.")
        self.add_stage(
            "denoising_stage",
            MiniMaxH3DenoisingStage(
                transformer=transformer,
                scheduler=scheduler,
                audio_scheduler=audio_scheduler,
            ),
        )

    def denoise(self, state: MiniMaxH3EncodedState) -> MiniMaxH3DenoisedState:
        if not self.post_init_called:
            self.post_init()
        batch = self._stage_name_mapping["denoising_stage"](state.to_batch(), self.fastvideo_args)
        return MiniMaxH3DenoisedState.from_batch(batch, request_id=state.request_id)


class MiniMaxH3RefDiTPipeline(MiniMaxH3DiTPipeline):
    """Persistent Ref2VA DiT worker using the checkpoint transformer_ref partition."""

    _extra_config_module_map = {"transformer": "transformer_ref"}
    _ref2va_default = True


__all__ = [
    "MINIMAX_H3_WIRE_SCHEMA_VERSION",
    "MiniMaxH3DenoisedState",
    "MiniMaxH3DiTPipeline",
    "MiniMaxH3EncodedState",
    "MiniMaxH3EncoderDecoderPipeline",
    "MiniMaxH3RefDiTPipeline",
    "MiniMaxH3RefEncoderDecoderPipeline",
]
